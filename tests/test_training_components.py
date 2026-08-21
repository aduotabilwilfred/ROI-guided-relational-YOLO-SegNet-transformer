from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.models import train, train_folds
from src.models.train_handoff import FrozenYOLOFeatures
from src.utils.bilra_attention import BiLevelRoutingAttention
from src.utils.losses import segmentation_classification_loss
from src.utils.relational_head import HeadConfig, RelationalSegHead
from src.utils.roi_partition import build_roi_masks
from src.utils.rtrb import RelationalTransformerBlock


class BiLRATests(unittest.TestCase):
    def test_forward_shape_finite_and_backward(self):
        module = BiLevelRoutingAttention(dim=16, num_heads=4, n_regions=2, top_k=2)
        inputs = torch.randn(2, 16, 8, 8, requires_grad=True)

        output = module(inputs)

        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(any(p.grad is not None for p in module.parameters()))


class ROIMaskTests(unittest.TestCase):
    def test_one_box_produces_tumour_and_disjoint_context(self):
        masks = build_roi_masks(
            [torch.tensor([[0.25, 0.25, 0.75, 0.75]])], 8, 8, dilation=1.5
        )
        self.assertTrue(masks.has_roi.item())
        self.assertEqual(masks.m_mask.sum().item(), 16)
        self.assertGreater(masks.w_mask.sum().item(), 0)
        self.assertEqual((masks.m_mask * masks.w_mask).sum().item(), 0)

    def test_multiple_boxes_are_unioned(self):
        boxes = torch.tensor(
            [[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75]]
        )
        masks = build_roi_masks([boxes], 8, 8)
        self.assertTrue(masks.has_roi.item())
        self.assertEqual(masks.m_mask.sum().item(), 28)

    def test_zero_boxes_are_safe(self):
        masks = build_roi_masks([torch.empty(0, 4)], 8, 8)
        self.assertFalse(masks.has_roi.item())
        self.assertEqual(masks.m_mask.sum().item(), 0)
        self.assertEqual(masks.w_mask.sum().item(), 0)


class RelationalModelTests(unittest.TestCase):
    def test_rtrb_shape_and_backward_with_zero_box_sample(self):
        block = RelationalTransformerBlock(in_channels=8, embed_dim=8, num_heads=2)
        features = torch.randn(2, 8, 8, 8, requires_grad=True)
        masks = build_roi_masks(
            [torch.tensor([[0.2, 0.2, 0.7, 0.7]]), torch.empty(0, 4)], 8, 8
        )

        output = block(features, masks.m_mask, masks.w_mask)

        self.assertEqual(output.shape, (2, 16, 8, 8))
        self.assertTrue(torch.isfinite(output).all())
        output.mean().backward()
        self.assertIsNotNone(features.grad)

    def test_head_outputs_and_combined_loss_backward(self):
        cfg = HeadConfig(
            image_size=32,
            backbone_channels=(8, 16, 32),
            embed_dim=16,
            num_heads=4,
            n_regions=2,
            top_k=2,
        )
        head = RelationalSegHead(cfg)
        features = [
            torch.randn(3, 8, 8, 8),
            torch.randn(3, 16, 4, 4),
            torch.randn(3, 32, 2, 2),
        ]
        boxes = [
            torch.tensor([[0.2, 0.2, 0.7, 0.7]]),
            torch.tensor([[0.0, 0.0, 0.4, 0.4], [0.5, 0.5, 1.0, 1.0]]),
            torch.empty(0, 4),
        ]

        seg_logits, cls_logits = head(features, boxes)

        self.assertEqual(seg_logits.shape, (3, 1, 32, 32))
        self.assertEqual(cls_logits.shape, (3, 1))
        self.assertTrue(torch.isfinite(seg_logits).all())
        self.assertTrue(torch.isfinite(cls_logits).all())

        seg_target = torch.randint(0, 2, seg_logits.shape).float()
        cls_target = torch.tensor([[1.0], [1.0], [0.0]])
        total, seg_loss, cls_loss = segmentation_classification_loss(
            seg_logits,
            seg_target,
            cls_logits,
            cls_target,
            seg_weight=1.0,
            cls_weight=0.2,
        )
        self.assertTrue(torch.isfinite(torch.stack([total, seg_loss, cls_loss])).all())
        total.backward()
        self.assertIsNotNone(head.classifier[-1].weight.grad)
        self.assertTrue(any(p.grad is not None for p in head.stages.parameters()))

    def test_yolov8n_and_yolov8s_feature_channels_both_support_backward(self):
        detector_shapes = {
            "yolov8n": ((64, 8, 8), (128, 4, 4), (256, 2, 2)),
            "yolov8s": ((128, 8, 8), (256, 4, 4), (512, 2, 2)),
        }

        class ShapeOnlyDetector:
            def __init__(self, shapes):
                self.shapes = shapes

            def infer_feature_shapes(self, _image_size):
                return self.shapes

        for model_name, shapes in detector_shapes.items():
            with self.subTest(model=model_name):
                base_cfg = HeadConfig(
                    image_size=32,
                    embed_dim=16,
                    num_heads=4,
                    n_regions=2,
                    top_k=2,
                )
                cfg, measured_shapes = train.configure_head_for_detector(
                    base_cfg, ShapeOnlyDetector(shapes)
                )
                head = RelationalSegHead(cfg)
                features = [
                    torch.randn(1, channels, height, width)
                    for channels, height, width in shapes
                ]

                seg_logits, cls_logits = head(features, [torch.empty(0, 4)])

                self.assertEqual(measured_shapes, shapes)
                self.assertEqual(cfg.backbone_channels, tuple(s[0] for s in shapes))
                self.assertEqual(seg_logits.shape, (1, 1, 32, 32))
                self.assertEqual(cls_logits.shape, (1, 1))
                (seg_logits.mean() + cls_logits.mean()).backward()
                self.assertTrue(
                    all(projection.weight.grad is not None for projection in head.scale_proj)
                )


class HandoffContractTests(unittest.TestCase):
    def test_hooks_boxes_and_detector_freezing(self):
        predict_calls = []

        class Segment(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = [0, 1, 2]

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.ones(1))
                self.model = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])
                self.segment = Segment()

            def forward(self, inputs):
                output = inputs
                for layer in self.model:
                    output = layer(output)
                return output

        class FakeBoxes:
            def __init__(self, coords):
                self.xyxyn = coords

            def __len__(self):
                return len(self.xyxyn)

        class FakeResult:
            def __init__(self, boxes):
                self.boxes = boxes

        class FakeYOLO:
            def __init__(self, _weights):
                self.model = FakeModel()

            def predict(self, paths, **kwargs):
                predict_calls.append(kwargs)
                results = [
                    FakeResult(
                        FakeBoxes(
                            torch.tensor(
                                [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9]]
                            )
                        )
                    ),
                    FakeResult(None),
                ]
                return results[: len(paths)]

        fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)
        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            wrapper = FrozenYOLOFeatures("fake.pt", imgsz=32)
            features = wrapper.extract_features(torch.randn(2, 3, 32, 32))
            inferred_shapes = wrapper.infer_feature_shapes()
            boxes = wrapper.detect_boxes(["one.png", "two.png"])

        self.assertEqual(len(features), 3)
        self.assertTrue(all(feature.shape == (2, 3, 32, 32) for feature in features))
        self.assertFalse(any(parameter.requires_grad for parameter in wrapper.model.parameters()))
        self.assertEqual(boxes[0].shape, (2, 4))
        self.assertEqual(boxes[1].shape, (0, 4))
        self.assertTrue(all(not feature.requires_grad for feature in features))
        self.assertEqual(inferred_shapes, ((3, 32, 32),) * 3)
        self.assertEqual(wrapper.detector_conf, 0.25)
        self.assertEqual(predict_calls[-1]["conf"], 0.25)

        with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            overridden = FrozenYOLOFeatures(
                "fake.pt", detector_conf=0.025, imgsz=32
            )
            overridden.detect_boxes(["one.png"])

        self.assertEqual(overridden.detector_conf, 0.025)
        self.assertEqual(predict_calls[-1]["conf"], 0.025)

    def test_training_handoff_receives_detector_confidence(self):
        class StopBeforeTraining(Exception):
            pass

        with patch.object(
            train, "FrozenYOLOFeatures", side_effect=StopBeforeTraining
        ) as detector_factory:
            with self.assertRaises(StopBeforeTraining):
                train.train_one_fold(
                    fold_dir=Path("fold_0"),
                    weights=Path("best.pt"),
                    cfg=HeadConfig(),
                    device="cpu",
                    epochs=1,
                    batch_size=1,
                    lr=1e-4,
                    detector_conf=0.025,
                )

        detector_factory.assert_called_once_with(
            "best.pt", detector_conf=0.025, imgsz=512
        )


class TrainingCLITests(unittest.TestCase):
    def test_relational_smoke_options_are_explicit(self):
        argv = [
            "train",
            "--only-fold",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--image-size",
            "512",
            "--device",
            "cuda:0",
            "--amp",
            "--workers",
            "0",
            "--detector-conf",
            "0.025",
            "--patience",
            "7",
            "--skip-eval",
        ]
        with patch.object(sys, "argv", argv):
            args = train.parse_args()
        self.assertEqual(args.only_fold, 0)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.device, "cuda:0")
        self.assertTrue(args.amp)
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.detector_conf, 0.025)
        self.assertEqual(args.patience, 7)
        self.assertTrue(args.skip_eval)

    def test_relational_detector_confidence_defaults_to_existing_value(self):
        with patch.object(sys, "argv", ["train"]):
            args = train.parse_args()

        self.assertEqual(args.detector_conf, 0.25)
        self.assertEqual(args.patience, 7)

    def test_detector_smoke_options_are_explicit(self):
        argv = [
            "train_folds",
            "--only-fold",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--imgsz",
            "512",
            "--device",
            "0",
            "--amp",
            "--workers",
            "0",
        ]
        with patch.object(sys, "argv", argv):
            args = train_folds.parse_args()
        self.assertEqual(args.only_fold, 0)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.device, "0")
        self.assertTrue(args.amp)
        self.assertEqual(args.workers, 0)

    def test_each_cv_fold_is_accepted_by_both_clis(self):
        for fold in range(5):
            with self.subTest(fold=fold):
                with patch.object(
                    sys, "argv", ["train_folds", "--only-fold", str(fold)]
                ):
                    detector_args = train_folds.parse_args()
                with patch.object(
                    sys, "argv", ["train", "--only-fold", str(fold)]
                ):
                    relational_args = train.parse_args()

                self.assertEqual(detector_args.only_fold, fold)
                self.assertEqual(relational_args.only_fold, fold)
                self.assertEqual(
                    train.detector_checkpoint_for_fold(
                        Path("runs/final_yolov8s_cv"), fold
                    ),
                    Path(f"runs/final_yolov8s_cv/fold_{fold}/weights/best.pt"),
                )


class CrossValidationIsolationTests(unittest.TestCase):
    def test_detector_checkpoint_metadata_must_match_requested_fold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for fold in range(5):
                checkpoint = (
                    root / "detectors" / f"fold_{fold}" / "weights" / "best.pt"
                )
                checkpoint.parent.mkdir(parents=True)
                torch.save(
                    {
                        "train_args": {
                            "name": f"fold_{fold}",
                            "data": str(
                                root
                                / "folds"
                                / f"fold_{fold}"
                                / "dataset.yaml"
                            ),
                        }
                    },
                    checkpoint,
                )
                train.verify_detector_fold_alignment(
                    checkpoint, root / "folds" / f"fold_{fold}", fold
                )

            with self.assertRaisesRegex(RuntimeError, "not aligned"):
                train.verify_detector_fold_alignment(
                    root / "detectors" / "fold_0" / "weights" / "best.pt",
                    root / "folds" / "fold_1",
                    1,
                )

    def test_five_individual_reports_are_preserved_and_aggregated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            for fold in range(5):
                test_metrics = {
                    "dice": 0.2 + fold * 0.1,
                    "positive_dice": 0.25 + fold * 0.1,
                    "iou": 0.1 + fold * 0.05,
                    "positive_iou": 0.12 + fold * 0.05,
                    "segmentation_precision": 0.3 + fold * 0.05,
                    "segmentation_recall": 0.4 + fold * 0.05,
                    "segmentation_f1": 0.2 + fold * 0.1,
                    "classification_accuracy": 0.9,
                    "classification_precision": 0.9,
                    "classification_recall": 0.9,
                    "classification_f1": 0.9,
                    "classification_specificity": 0.9,
                }
                train.write_fold_metrics(
                    output,
                    fold,
                    {
                        "fold": fold,
                        "detector_project": "detectors",
                        "detector_conf": 0.025,
                        "image_size": 512,
                        "best_epoch": fold + 1,
                        "validation_positive_dice": 0.3 + fold * 0.1,
                        "training_duration_seconds": 100 + fold,
                        "test_metrics": test_metrics,
                    },
                )

            aggregate = train.aggregate_completed_fold_metrics(output, 5)

            self.assertIsNotNone(aggregate)
            self.assertAlmostEqual(aggregate["summary"]["test_dice"]["mean"], 0.4)
            self.assertGreater(aggregate["summary"]["test_dice"]["std"], 0)
            self.assertTrue((output / "aggregate_metrics.json").exists())
            for fold in range(5):
                path = output / f"fold_{fold}_metrics.json"
                self.assertTrue(path.exists())
                self.assertEqual(json.loads(path.read_text())["fold"], fold)


class ScientificTrainingWorkflowTests(unittest.TestCase):
    @staticmethod
    def _validation_metrics(positive_dice: float) -> dict:
        return {
            "total_loss": 0.5,
            "segmentation_loss": 0.4,
            "classification_loss": 0.5,
            "dice": positive_dice,
            "positive_dice": positive_dice,
            "classification_accuracy": 1.0,
            "intersection": 1,
            "pred_area": 1,
            "gt_area": 1,
            "positive_intersection": 1,
            "positive_pred_area": 1,
            "positive_gt_area": 1,
            "correct": 1,
            "total": 1,
        }

    def test_validation_drives_best_last_early_stopping_and_history(self):
        class TinyDataset:
            def __init__(self, _fold_dir, role, _image_size):
                self.role = role

            def __len__(self):
                return 1

            def __getitem__(self, _index):
                mask = torch.zeros(4, 4)
                mask[1:3, 1:3] = 1
                return {
                    "image": torch.zeros(3, 4, 4),
                    "mask": mask,
                    "path": "train.png",
                }

        class TinyDetector:
            def __init__(self, *_args, **_kwargs):
                self.model = nn.Identity()

            def extract_features(self, images):
                return [images]

            def infer_feature_shapes(self, _image_size):
                return ((3, 4, 4), (3, 2, 2), (3, 1, 1))

            def detect_boxes(self, paths):
                return [torch.empty(0, 4) for _ in paths]

        class TinyHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(0.0))

            def forward(self, features, _boxes):
                images = features[0]
                batch, _, height, width = images.shape
                seg = self.bias.reshape(1, 1, 1, 1).expand(
                    batch, 1, height, width
                )
                cls = self.bias.reshape(1, 1).expand(batch, 1)
                return seg, cls

        validation_sequence = [
            self._validation_metrics(0.6),
            self._validation_metrics(0.5),
            self._validation_metrics(0.4),
            self._validation_metrics(0.3),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            with (
                patch.object(train, "FoldImageDataset", TinyDataset),
                patch.object(train, "FrozenYOLOFeatures", TinyDetector),
                patch.object(train, "buildHead", return_value=TinyHead()),
                patch.object(
                    train, "evaluate_split", side_effect=validation_sequence
                ) as evaluate,
            ):
                train.train_one_fold(
                    fold_dir=Path("fold_0"),
                    weights=Path("detector.pt"),
                    cfg=HeadConfig(image_size=4),
                    device="cpu",
                    epochs=4,
                    batch_size=1,
                    lr=1e-4,
                    fold_idx=0,
                    workers=0,
                    amp=False,
                    detector_conf=0.025,
                    output_dir=output,
                    patience=2,
                )

            validation_roles = [call.kwargs["role"] for call in evaluate.call_args_list]
            self.assertEqual(validation_roles, ["val", "val", "val"])

            best_path = output / "best_head_fold0.pth"
            last_path = output / "last_head_fold0.pth"
            history_path = output / "history_fold0.csv"
            self.assertTrue(best_path.exists())
            self.assertTrue(last_path.exists())
            self.assertTrue(history_path.exists())

            best = torch.load(best_path, map_location="cpu", weights_only=True)
            last = torch.load(last_path, map_location="cpu", weights_only=True)
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(last["epoch"], 3)
            self.assertEqual(best["validation_metrics"]["positive_dice"], 0.6)
            self.assertEqual(last["validation_metrics"]["positive_dice"], 0.4)
            self.assertEqual(best["checkpoint_selection_metric"], "validation_positive_dice")
            self.assertIn("optimizer_state_dict", best)
            self.assertIn("training_config", best)
            self.assertEqual(best["detector_conf"], 0.025)
            restored = TinyHead()
            structured = train.load_head_checkpoint(best_path, restored)
            self.assertFalse(structured["legacy_bare_state_dict"])
            self.assertEqual(structured["epoch"], 1)

            with history_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["epoch"] for row in rows], ["1", "2", "3"])
            self.assertEqual(rows[0]["best_checkpoint"], "True")
            self.assertEqual(rows[2]["epochs_without_improvement"], "2")
            self.assertIn("validation_positive_dice", rows[0])
            self.assertIn("validation_segmentation_loss", rows[0])

    def test_old_bare_state_dict_still_loads_strictly(self):
        source = nn.Linear(2, 1)
        target = nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "legacy.pth"
            torch.save(source.state_dict(), checkpoint)
            metadata = train.load_head_checkpoint(checkpoint, target)

        self.assertTrue(metadata["legacy_bare_state_dict"])
        for source_parameter, target_parameter in zip(
            source.parameters(), target.parameters()
        ):
            self.assertTrue(torch.equal(source_parameter, target_parameter))

    def test_main_loads_best_before_single_test_evaluation(self):
        test_metrics = self._validation_metrics(0.7)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detector_project = root / "detector"
            weights = detector_project / "fold_0" / "weights" / "best.pt"
            weights.parent.mkdir(parents=True)
            weights.touch()
            args = SimpleNamespace(
                folds_root=root / "folds",
                detector_project=detector_project,
                detector_conf=0.025,
                output=root / "output",
                n_folds=5,
                epochs=30,
                batch_size=1,
                workers=0,
                lr=1e-4,
                device="cpu",
                image_size=4,
                only_fold=0,
                amp=False,
                seg_weight=1.0,
                cls_weight=0.2,
                patience=7,
                skip_eval=False,
                no_mlflow=True,
                mlflow_experiment="test",
                mlflow_run_name="test",
                mlflow_uri="",
            )
            head = TinyHeadForMain()
            events = []

            def load_best(path, model, **_kwargs):
                events.append(("load", path, model))
                return {"epoch": 4}

            def evaluate_test(**kwargs):
                events.append(("evaluate", kwargs["role"], kwargs["head"]))
                return test_metrics

            with (
                patch.object(train, "parse_args", return_value=args),
                patch.object(
                    train,
                    "train_one_fold",
                    return_value=(head, object()),
                ) as fit,
                patch.object(train, "verify_detector_fold_alignment"),
                patch.object(train, "load_head_checkpoint", side_effect=load_best),
                patch.object(train, "evaluate_split", side_effect=evaluate_test),
            ):
                result = train.main()

            self.assertEqual(result, 0)
            self.assertEqual(events[0][0], "load")
            self.assertEqual(events[0][1], args.output / "best_head_fold0.pth")
            self.assertEqual(events[1], ("evaluate", "test", head))
            self.assertEqual(fit.call_args.kwargs["output_dir"], args.output)
            self.assertEqual(fit.call_args.kwargs["patience"], 7)


class TinyHeadForMain(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
