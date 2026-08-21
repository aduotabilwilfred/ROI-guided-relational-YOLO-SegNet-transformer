from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .train_handoff import FrozenYOLOFeatures
from ..utils.losses import segmentation_classification_loss
from ..utils.relational_head import HeadConfig, buildHead


CLASSIFICATION_METRIC_NOTE = (
    "Classification accuracy is diagnostic only: the current dataset contains a "
    "strong aspect-ratio/acquisition shortcut. It is not used for checkpoint "
    "selection, early stopping, or model ranking."
)


def detector_checkpoint_for_fold(detector_project: Path, fold: int) -> Path:
    """Return the only detector checkpoint valid for a relational CV fold."""
    return detector_project / f"fold_{fold}" / "weights" / "best.pt"


def verify_detector_fold_alignment(
    detector_checkpoint: Path, fold_dir: Path, fold: int
) -> None:
    """Reject detector/data fold mismatches before constructing either model."""
    expected_fold = f"fold_{fold}"
    if fold_dir.name != expected_fold:
        raise RuntimeError(
            f"fold data mismatch: requested {expected_fold}, got {fold_dir}"
        )
    if detector_checkpoint.parent.parent.name != expected_fold:
        raise RuntimeError(
            "detector checkpoint path is not aligned with the requested fold: "
            f"expected .../{expected_fold}/weights/best.pt, got "
            f"{detector_checkpoint}"
        )

    payload = torch.load(
        detector_checkpoint, map_location="cpu", weights_only=False
    )
    train_args = payload.get("train_args") if isinstance(payload, dict) else None
    if not isinstance(train_args, dict):
        raise RuntimeError(
            f"detector checkpoint has no Ultralytics train_args: {detector_checkpoint}"
        )

    recorded_name = str(train_args.get("name", ""))
    recorded_data = str(train_args.get("data", "")).replace("\\", "/")
    expected_data_suffix = f"/{expected_fold}/dataset.yaml"
    if recorded_name != expected_fold or not (
        f"/{recorded_data.lstrip('/')}"
    ).endswith(expected_data_suffix):
        raise RuntimeError(
            "detector checkpoint metadata is not aligned with the requested fold: "
            f"requested={expected_fold}, recorded name={recorded_name!r}, "
            f"recorded data={recorded_data!r}"
        )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


class MLflowRun:
    def __init__(
        self, enabled: bool, experiment_name: str, run_name: str, tracking_uri: str = ""
    ):
        self.enabled = enabled
        self.mlflow = None
        self.experiments = experiment_name
        self.run_name = run_name

        if not enabled:
            return
        try:
            import mlflow

            self.mlflow = mlflow

            uri = tracking_uri or "sqlite:///mlflow.db"
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self.experiments)
            mlflow.start_run(run_name=self.run_name)
        except Exception as exc:  # noqa: BLE001
            print(f"MLflow disabled (init failed): {exc}")
            self.enabled = False
            self.mlflow = None

    def log_params(self, params: dict) -> None:
        if self.enabled and self.mlflow:
            try:
                self.mlflow.log_params(params)
            except Exception as exc:  # noqa: BLE001
                print(f"MLflow log_params failed: {exc}")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self.enabled and self.mlflow:
            try:
                self.mlflow.log_metric(key, value, step=step)
            except Exception as exc:  # noqa: BLE001
                print(f"MLflow log_metric failed: {exc}")

    def end(self) -> None:
        if self.enabled and self.mlflow:
            try:
                self.mlflow.end_run()
            except Exception:  # noqa: BLE001, S110
                pass


class FoldImageDataset(Dataset):
    """
    Serves denoised images and box-derived masks from a materialised fold.

    Reads the images YOLO trained on (fold_k/role/images) and their labels
    (fold_k/role/labels), so detection and segmentation share identical input.
    Returns the image as a 3-channel tensor (YOLO expects RGB), the path (the
    detector predicts from files), and the ground-truth mask.
    """

    def __init__(self, fold_dir: Path, role: str, image_size: int = 512):
        self.image_dir = fold_dir / role / "images"
        self.labels_dir = fold_dir / role / "labels"
        self.size = image_size
        self.paths = sorted(self.image_dir.glob("*.png")) + sorted(
            self.image_dir.glob("*.jpg")
        )

    def __len__(self):
        return len(self.paths)

    def _mask_from_labels(self, label_path: Path) -> np.ndarray:
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if not label_path.exists():
            return mask

        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            coords = np.array([float(v) for v in parts[1:]], dtype=np.float64)
            pts = (coords.reshape(-1, 2) * self.size).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def __getitem__(self, index: int) -> dict:
        """
        YOLOv8's first conv expects 3 channels so the single
        grayscale channels is replicated to R=G=B
        """
        path = self.paths[index]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        rgb = np.repeat(img[None], 3, axis=0).astype(np.float32) / 255.0
        mask = self._mask_from_labels(self.labels_dir / f"{path.stem}.txt")
        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(mask).float(),
            "path": str(path),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "path": [b["path"] for b in batch],
    }


def write_history(path: Path, rows: list[dict]) -> None:
    """Persist epoch metrics independently of console and MLflow output."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_training_checkpoint(
    path: Path,
    *,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    training_config: dict,
    detector_checkpoint: Path,
    detector_conf: float,
    validation_metrics: dict,
    seg_weight: float,
    cls_weight: float,
    image_size: int,
    batch_size: int,
    random_seed: int | None,
) -> None:
    """Save a resumable, self-describing relational-head checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "training_config": training_config,
            "detector_checkpoint": str(detector_checkpoint.resolve()),
            "detector_conf": detector_conf,
            "validation_metrics": validation_metrics,
            "seg_weight": seg_weight,
            "cls_weight": cls_weight,
            "image_size": image_size,
            "batch_size": batch_size,
            "random_seed": random_seed,
            "checkpoint_selection_metric": "validation_positive_dice",
            "classification_metric_note": CLASSIFICATION_METRIC_NOTE,
        },
        path,
    )


def load_head_checkpoint(
    path: Path,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load structured checkpoints and older bare state_dict files strictly."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        head.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return {**payload, "legacy_bare_state_dict": False}

    head.load_state_dict(payload, strict=True)
    return {
        "model_state_dict": payload,
        "legacy_bare_state_dict": True,
    }


@torch.no_grad()
def evaluate_split(
    head,
    detector,
    fold_dir: Path,
    cfg: HeadConfig,
    device: str,
    batch_size: int,
    role: str,
    workers: int = 0,
    amp: bool = True,
    seg_weight: float = 1.0,
    cls_weight: float = 0.2,
) -> dict:
    """Evaluate one named split at the unchanged 0.5 segmentation threshold."""
    was_training = head.training
    head.eval()
    ds = FoldImageDataset(fold_dir, role, cfg.image_size)
    use_amp = amp and torch.device(device).type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=workers,
        pin_memory=use_amp,
    )
    if len(loader) == 0:
        raise ValueError(f"{role} split is empty: {fold_dir / role}")

    inter = pred_area = gt_area = 0
    positive_inter = positive_pred_area = positive_gt_area = 0
    total_loss = segmentation_loss = classification_loss = 0.0
    correct = total = 0
    true_positive = true_negative = false_positive = false_negative = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].unsqueeze(1).to(device)
        cls_targets = (targets.flatten(1).sum(dim=1) > 0).float().unsqueeze(1)

        feats = detector.extract_features(images)
        boxes = detector.detect_boxes(batch["path"])
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
            logits, cls_logits = head(feats, boxes)
            loss, seg_loss, cls_loss = segmentation_classification_loss(
                logits,
                targets,
                cls_logits,
                cls_targets,
                seg_weight=seg_weight,
                cls_weight=cls_weight,
            )

        batch_size_actual = targets.shape[0]
        total_loss += loss.item() * batch_size_actual
        segmentation_loss += seg_loss.item() * batch_size_actual
        classification_loss += cls_loss.item() * batch_size_actual
        pred = (logits.sigmoid() > 0.5).float()

        inter += int((pred * targets).sum().item())
        pred_area += int(pred.sum().item())
        gt_area += int(targets.sum().item())

        positive = cls_targets.flatten().bool()
        if positive.any():
            positive_pred = pred[positive]
            positive_target = targets[positive]
            positive_inter += int((positive_pred * positive_target).sum().item())
            positive_pred_area += int(positive_pred.sum().item())
            positive_gt_area += int(positive_target.sum().item())

        cls_pred = (cls_logits > 0).flatten()
        cls_true = cls_targets.bool().flatten()
        correct += int((cls_pred == cls_true).sum().item())
        true_positive += int((cls_pred & cls_true).sum().item())
        true_negative += int((~cls_pred & ~cls_true).sum().item())
        false_positive += int((cls_pred & ~cls_true).sum().item())
        false_negative += int((~cls_pred & cls_true).sum().item())
        total += batch_size_actual

    if was_training:
        head.train()

    eps = 1e-7
    union = pred_area + gt_area - inter
    positive_union = positive_pred_area + positive_gt_area - positive_inter
    classification_precision = _ratio(
        true_positive, true_positive + false_positive
    )
    classification_recall = _ratio(
        true_positive, true_positive + false_negative
    )
    return {
        "total_loss": total_loss / total,
        "segmentation_loss": segmentation_loss / total,
        "classification_loss": classification_loss / total,
        "dice": (2 * inter + eps) / (pred_area + gt_area + eps),
        "positive_dice": (2 * positive_inter + eps)
        / (positive_pred_area + positive_gt_area + eps),
        "iou": _ratio(inter, union),
        "jaccard": _ratio(inter, union),
        "segmentation_precision": _ratio(inter, pred_area),
        "segmentation_recall": _ratio(inter, gt_area),
        "segmentation_f1": (2 * inter + eps) / (pred_area + gt_area + eps),
        "positive_iou": _ratio(positive_inter, positive_union),
        "positive_segmentation_precision": _ratio(
            positive_inter, positive_pred_area
        ),
        "positive_segmentation_recall": _ratio(positive_inter, positive_gt_area),
        "positive_segmentation_f1": (2 * positive_inter + eps)
        / (positive_pred_area + positive_gt_area + eps),
        "classification_accuracy": correct / total,
        "classification_precision": classification_precision,
        "classification_recall": classification_recall,
        "classification_f1": _ratio(
            2 * classification_precision * classification_recall,
            classification_precision + classification_recall,
        ),
        "classification_specificity": _ratio(
            true_negative, true_negative + false_positive
        ),
        "classification_true_positive": true_positive,
        "classification_true_negative": true_negative,
        "classification_false_positive": false_positive,
        "classification_false_negative": false_negative,
        "intersection": inter,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "positive_intersection": positive_inter,
        "positive_pred_area": positive_pred_area,
        "positive_gt_area": positive_gt_area,
        "correct": correct,
        "total": total,
    }


def configure_head_for_detector(
    cfg: HeadConfig, detector: FrozenYOLOFeatures
) -> tuple[HeadConfig, tuple[tuple[int, int, int], ...]]:
    """Bind only the head's input projections to the selected detector shapes."""
    feature_shapes = detector.infer_feature_shapes(cfg.image_size)
    if len(feature_shapes) != 3:
        raise RuntimeError(
            f"expected detector P3/P4/P5 features, received {len(feature_shapes)} maps"
        )
    channels = tuple(shape[0] for shape in feature_shapes)
    return replace(cfg, backbone_channels=channels), feature_shapes


def train_one_fold(
    fold_dir: Path,
    weights: Path,
    cfg: HeadConfig,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    mlrun: MLflowRun = None,
    fold_idx: int = 0,
    workers: int = 0,
    amp: bool = True,
    seg_weight: float = 1.0,
    cls_weight: float = 0.2,
    detector_conf: float = 0.25,
    output_dir: Path | None = None,
    patience: int | None = None,
    random_seed: int | None = None,
) -> tuple[torch.nn.Module, FrozenYOLOFeatures]:
    detector = FrozenYOLOFeatures(
        str(weights), detector_conf=detector_conf, imgsz=cfg.image_size
    )
    detector.model.to(device)
    cfg, feature_shapes = configure_head_for_detector(cfg, detector)
    print(
        f" detector feature shapes at {cfg.image_size}px: "
        f"{feature_shapes}; head input channels: {cfg.backbone_channels}"
    )
    head = buildHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    train_ds = FoldImageDataset(fold_dir, "train", cfg.image_size)
    use_amp = amp and torch.device(device).type == "cuda"
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=use_amp,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = []
    best_positive_dice = float("-inf")
    best_epoch = None
    epochs_without_improvement = 0
    training_started = time.perf_counter()
    stopped_early = False
    training_config = {
        "fold": fold_idx,
        "fold_dir": str(fold_dir),
        "max_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": device,
        "workers": workers,
        "amp": amp,
        "patience": patience,
        "seg_weight": seg_weight,
        "cls_weight": cls_weight,
        "detector_checkpoint": str(weights.resolve()),
        "detector_conf": detector_conf,
        "image_size": cfg.image_size,
        "random_seed": random_seed,
        "head_config": asdict(cfg),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "fold": fold_idx,
            "split_usage": {
                "train": "gradient updates only",
                "validation": (
                    "evaluated after every epoch; selects best checkpoint and "
                    "controls early stopping"
                ),
                "test": "evaluated once after training using the best checkpoint",
            },
            "checkpoint_selection_metric": "validation_positive_dice",
            "segmentation_threshold": 0.5,
            "classification_metric_note": CLASSIFICATION_METRIC_NOTE,
        }
        (output_dir / f"training_metadata_fold{fold_idx}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    print(CLASSIFICATION_METRIC_NOTE)
    for epoch_index in range(epochs):
        epoch_started = time.perf_counter()
        head.train()
        running = 0.0
        running_seg = 0.0
        running_cls = 0.0
        train_inter = train_pred = train_gt = 0
        train_correct = train_total = 0
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].unsqueeze(1).to(device)
            cls_targets = (targets.flatten(1).sum(dim=1) > 0).float().unsqueeze(1)

            """
            frozen detector runs twice per image
            once on the tensor to hook neck features, once on the file paths so ultralytics predict
            does its own letterbox + NMS for boxes. Both see identical content
            """
            feats = detector.extract_features(images)
            boxes = detector.detect_boxes(batch["path"])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(device).type, enabled=use_amp
            ):
                logits, cls_logits = head(feats, boxes)
                loss, seg_loss, cls_loss = segmentation_classification_loss(
                    logits,
                    targets,
                    cls_logits,
                    cls_targets,
                    seg_weight=seg_weight,
                    cls_weight=cls_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            batch_size_actual = images.shape[0]
            running += loss.item() * batch_size_actual
            running_seg += seg_loss.item() * batch_size_actual
            running_cls += cls_loss.item() * batch_size_actual

            with torch.no_grad():
                pred = (logits > 0).float()
                train_inter += int((pred * targets).sum().item())
                train_pred += int(pred.sum().item())
                train_gt += int(targets.sum().item())

                predicted_classes = cls_logits > 0
                train_correct += int(
                    (predicted_classes == cls_targets.bool()).sum().item()
                )
                train_total += images.shape[0]

        epoch_loss = running / train_total
        epoch_seg_loss = running_seg / train_total
        epoch_cls_loss = running_cls / train_total
        eps = 1e-7
        train_dice = (2 * train_inter + eps) / (train_pred + train_gt + eps)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        validation = evaluate_split(
            head=head,
            detector=detector,
            fold_dir=fold_dir,
            cfg=cfg,
            device=device,
            batch_size=batch_size,
            role="val",
            workers=workers,
            amp=amp,
            seg_weight=seg_weight,
            cls_weight=cls_weight,
        )
        positive_dice = validation["positive_dice"]
        improved = positive_dice > best_positive_dice
        if improved:
            best_positive_dice = positive_dice
            best_epoch = epoch_index + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch = epoch_index + 1
        history_row = {
            "epoch": epoch,
            "train_total_loss": epoch_loss,
            "train_segmentation_loss": epoch_seg_loss,
            "train_classification_loss": epoch_cls_loss,
            "train_dice": train_dice,
            "train_classification_accuracy": train_acc,
            "validation_total_loss": validation["total_loss"],
            "validation_segmentation_loss": validation["segmentation_loss"],
            "validation_classification_loss": validation["classification_loss"],
            "validation_dice": validation["dice"],
            "validation_positive_dice": positive_dice,
            "validation_classification_accuracy": validation[
                "classification_accuracy"
            ],
            "best_checkpoint": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_duration_seconds": time.perf_counter() - epoch_started,
            "elapsed_training_seconds": time.perf_counter() - training_started,
        }
        history.append(history_row)
        print(
            f" fold {fold_dir.name} | epoch {epoch}/{epochs} | "
            f"loss {epoch_loss:.4f} (seg {epoch_seg_loss:.4f}, cls {epoch_cls_loss:.4f}) "
            f"| dice {train_dice:.4f} | "
            f"val loss {validation['total_loss']:.4f} "
            f"(seg {validation['segmentation_loss']:.4f}, "
            f"cls {validation['classification_loss']:.4f}) | "
            f"val dice {validation['dice']:.4f} | "
            f"val positive dice {positive_dice:.4f} | "
            f"val cls acc {validation['classification_accuracy']:.4f} (diagnostic only)"
        )

        if mlrun is not None:
            mlrun.log_metric(
                f"fold_{fold_idx}_train_loss", epoch_loss, step=epoch_index
            )
            mlrun.log_metric(
                f"fold_{fold_idx}_train_seg_loss",
                epoch_seg_loss,
                step=epoch_index,
            )
            mlrun.log_metric(
                f"fold_{fold_idx}_train_cls_loss",
                epoch_cls_loss,
                step=epoch_index,
            )
            mlrun.log_metric(
                f"fold_{fold_idx}_train_dice", train_dice, step=epoch_index
            )
            mlrun.log_metric(
                f"fold_{fold_idx}_train_acc", train_acc, step=epoch_index
            )
            for name in (
                "total_loss",
                "segmentation_loss",
                "classification_loss",
                "dice",
                "positive_dice",
                "classification_accuracy",
            ):
                mlrun.log_metric(
                    f"fold_{fold_idx}_validation_{name}",
                    validation[name],
                    step=epoch_index,
                )

        if output_dir is not None:
            checkpoint_args = {
                "head": head,
                "optimizer": opt,
                "scaler": scaler,
                "epoch": epoch,
                "training_config": training_config,
                "detector_checkpoint": weights,
                "detector_conf": detector_conf,
                "validation_metrics": validation,
                "seg_weight": seg_weight,
                "cls_weight": cls_weight,
                "image_size": cfg.image_size,
                "batch_size": batch_size,
                "random_seed": random_seed,
            }
            save_training_checkpoint(
                output_dir / f"last_head_fold{fold_idx}.pth", **checkpoint_args
            )
            if improved:
                save_training_checkpoint(
                    output_dir / f"best_head_fold{fold_idx}.pth", **checkpoint_args
                )
            write_history(output_dir / f"history_fold{fold_idx}.csv", history)

        if patience is not None and patience > 0:
            if epochs_without_improvement >= patience:
                print(
                    f" early stopping fold {fold_idx} at epoch {epoch}: "
                    f"validation positive Dice did not improve for {patience} epochs"
                )
                stopped_early = True
                break

    if output_dir is not None:
        summary = {
            "fold": fold_idx,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_positive_dice": best_positive_dice,
            "training_duration_seconds": time.perf_counter() - training_started,
            "stopped_early": stopped_early,
            "detector_checkpoint": str(weights.resolve()),
        }
        (output_dir / f"training_summary_fold{fold_idx}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return head, detector


def accumulate_fold_dice(
    head,
    detector,
    fold_dir: Path,
    cfg: HeadConfig,
    device: str,
    batch_size: int,
    role: str = "test",
    workers: int = 0,
    amp: bool = True,
    seg_weight: float = 1.0,
    cls_weight: float = 0.2,
) -> tuple[int, int, int, float, int, int]:
    """Backward-compatible tuple wrapper around :func:`evaluate_split`."""
    metrics = evaluate_split(
        head=head,
        detector=detector,
        fold_dir=fold_dir,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        role=role,
        workers=workers,
        amp=amp,
        seg_weight=seg_weight,
        cls_weight=cls_weight,
    )
    return (
        metrics["intersection"],
        metrics["pred_area"],
        metrics["gt_area"],
        metrics["total_loss"],
        metrics["correct"],
        metrics["total"],
    )


AGGREGATE_METRICS = {
    "test_dice": ("test_metrics", "dice"),
    "tumour_positive_test_dice": ("test_metrics", "positive_dice"),
    "test_iou_jaccard": ("test_metrics", "iou"),
    "tumour_positive_test_iou_jaccard": ("test_metrics", "positive_iou"),
    "segmentation_precision": ("test_metrics", "segmentation_precision"),
    "segmentation_recall": ("test_metrics", "segmentation_recall"),
    "segmentation_f1": ("test_metrics", "segmentation_f1"),
    "classification_accuracy_diagnostic": (
        "test_metrics",
        "classification_accuracy",
    ),
    "classification_precision_diagnostic": (
        "test_metrics",
        "classification_precision",
    ),
    "classification_recall_diagnostic": (
        "test_metrics",
        "classification_recall",
    ),
    "classification_f1_diagnostic": ("test_metrics", "classification_f1"),
    "classification_specificity_diagnostic": (
        "test_metrics",
        "classification_specificity",
    ),
    "best_epoch": ("best_epoch",),
    "validation_positive_dice": ("validation_positive_dice",),
    "training_duration_seconds": ("training_duration_seconds",),
}


def write_fold_metrics(output_dir: Path, fold: int, report: dict) -> Path:
    """Persist one held-out result without touching any other fold's report."""
    path = output_dir / f"fold_{fold}_metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def aggregate_completed_fold_metrics(
    output_dir: Path, n_folds: int
) -> dict | None:
    """Aggregate only after all independently persisted fold reports exist."""
    paths = [output_dir / f"fold_{fold}_metrics.json" for fold in range(n_folds)]
    if not all(path.exists() for path in paths):
        return None

    folds = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = list(range(n_folds))
    actual = [report.get("fold") for report in folds]
    if actual != expected:
        raise RuntimeError(
            f"cannot aggregate misaligned fold reports: expected {expected}, got {actual}"
        )

    experiment_keys = ("detector_project", "detector_conf", "image_size")
    for key in experiment_keys:
        values = {str(report.get(key)) for report in folds}
        if len(values) != 1:
            raise RuntimeError(
                f"cannot aggregate fold reports with inconsistent {key}: {values}"
            )

    summary = {}
    for metric, keys in AGGREGATE_METRICS.items():
        values = []
        for report in folds:
            value = report
            for key in keys:
                value = value[key]
            values.append(float(value))
        summary[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "values_by_fold": values,
        }

    aggregate = {
        "n_folds": n_folds,
        "standard_deviation": "population (ddof=0)",
        "checkpoint_selection_metric": "validation_positive_dice",
        "classification_metric_note": CLASSIFICATION_METRIC_NOTE,
        "fold_reports": [path.name for path in paths],
        "summary": summary,
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training the relational head across folds, frozen two-stage."
    )

    parser.add_argument(
        "--folds-root", type=Path, default=Path("outputs/ultralytics_folds")
    )
    parser.add_argument(
        "--detector-project",
        type=Path,
        default=Path("runs/segment_cv"),
        help="where train_folds.py wrote each fold's weights",
    )
    parser.add_argument(
        "--detector-conf",
        type=float,
        default=0.25,
        help="confidence threshold for detector ROI boxes",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/relational_head"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest on Windows)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--only-fold",
        type=int,
        default=-1,
        help="train one fold only; -1 trains every fold",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use CUDA automatic mixed precision when running on CUDA",
    )
    parser.add_argument("--seg-weight", type=float, default=1.0)
    parser.add_argument("--cls-weight", type=float, default=0.2)
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help=(
            "stop after this many epochs without validation-positive-Dice "
            "improvement; 0 disables early stopping"
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="save trained head without running the held-out fold evaluation",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="disable MLflow logging"
    )
    parser.add_argument(
        "--mlflow-experiment", default="relational_head", help="MLflow experiment name"
    )
    parser.add_argument(
        "--mlflow-run-name", default="two_stage_cv", help="MLflow run name"
    )
    parser.add_argument(
        "--mlflow-uri",
        default="",
        help="MLflow tracking URI; empty uses local ./mlruns",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.patience < 0:
        raise SystemExit("--patience must be non-negative")
    cfg = HeadConfig(image_size=args.image_size)
    args.output.mkdir(parents=True, exist_ok=True)

    mlrun = MLflowRun(
        enabled=not args.no_mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        tracking_uri=args.mlflow_uri,
    )
    mlrun.log_params(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "workers": args.workers,
            "amp": args.amp,
            "seg_weight": args.seg_weight,
            "cls_weight": args.cls_weight,
            "detector_conf": args.detector_conf,
            "patience": args.patience,
            "skip_eval": args.skip_eval,
            "image_size": args.image_size,
            "n_folds": args.n_folds,
            "embed_dim": cfg.embed_dim,
            "num_heads": cfg.num_heads,
            "reduction": cfg.reduction,
            "dilation": cfg.dilation,
            "use_bilra_in_head": cfg.use_bilra_in_head,
        }
    )

    total_inter = total_pred = total_gt = 0
    total_positive_inter = total_positive_pred = total_positive_gt = 0
    total_correct = total_pixels = 0
    eps = 1e-7
    per_fold = {}

    if args.only_fold < -1 or args.only_fold >= args.n_folds:
        raise SystemExit(f"--only-fold must be between 0 and {args.n_folds - 1}")
    folds = list(range(args.n_folds)) if args.only_fold < 0 else [args.only_fold]
    for fold in folds:
        fold_dir = args.folds_root / f"fold_{fold}"
        weights = detector_checkpoint_for_fold(args.detector_project, fold)
        if not weights.exists():
            raise SystemExit(
                f"detector weights not found: {weights}\n"
                "train this fold's YOLOv8 detector (train_folds.py) first"
            )
        verify_detector_fold_alignment(weights, fold_dir, fold)

        print(f"=== training relational head, fold {fold} ==")
        head, detector = train_one_fold(
            fold_dir=fold_dir,
            weights=weights,
            cfg=cfg,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            mlrun=mlrun,
            fold_idx=fold,
            workers=args.workers,
            amp=args.amp,
            seg_weight=args.seg_weight,
            cls_weight=args.cls_weight,
            detector_conf=args.detector_conf,
            output_dir=args.output,
            patience=args.patience,
        )

        if args.skip_eval:
            continue

        best_checkpoint = args.output / f"best_head_fold{fold}.pth"
        checkpoint = load_head_checkpoint(
            best_checkpoint, head, map_location=args.device
        )
        test_metrics = evaluate_split(
            head=head,
            detector=detector,
            fold_dir=fold_dir,
            cfg=cfg,
            device=args.device,
            batch_size=args.batch_size,
            role="test",
            workers=args.workers,
            amp=args.amp,
            seg_weight=args.seg_weight,
            cls_weight=args.cls_weight,
        )

        per_fold[fold] = {
            "best_epoch": checkpoint.get("epoch"),
            "checkpoint": str(best_checkpoint),
            **test_metrics,
        }
        training_summary_path = args.output / f"training_summary_fold{fold}.json"
        training_summary = (
            json.loads(training_summary_path.read_text(encoding="utf-8"))
            if training_summary_path.exists()
            else {}
        )
        validation_metrics = checkpoint.get("validation_metrics", {})
        fold_report = {
            "fold": fold,
            "fold_data": str(fold_dir.resolve()),
            "detector_project": str(args.detector_project.resolve()),
            "detector_checkpoint": str(weights.resolve()),
            "detector_conf": args.detector_conf,
            "head_checkpoint": str(best_checkpoint.resolve()),
            "image_size": args.image_size,
            "best_epoch": checkpoint.get("epoch"),
            "validation_positive_dice": validation_metrics.get(
                "positive_dice",
                training_summary.get("best_validation_positive_dice"),
            ),
            "training_duration_seconds": training_summary.get(
                "training_duration_seconds", 0.0
            ),
            "classification_metric_note": CLASSIFICATION_METRIC_NOTE,
            "test_metrics": test_metrics,
        }
        fold_metrics_path = write_fold_metrics(args.output, fold, fold_report)
        print(f" fold {fold} metrics written to {fold_metrics_path}")
        print(
            f" fold {fold} BEST-checkpoint test loss "
            f"{test_metrics['total_loss']:.4f} | dice {test_metrics['dice']:.4f} "
            f"| positive dice {test_metrics['positive_dice']:.4f} | "
            f"cls acc {test_metrics['classification_accuracy']:.4f} (diagnostic only)"
        )

        metric_step = checkpoint.get("epoch", args.epochs)
        mlrun.log_metric(
            f"fold_{fold}_test_loss", test_metrics["total_loss"], step=metric_step
        )
        mlrun.log_metric(
            f"fold_{fold}_test_dice", test_metrics["dice"], step=metric_step
        )
        mlrun.log_metric(
            f"fold_{fold}_test_positive_dice",
            test_metrics["positive_dice"],
            step=metric_step,
        )
        mlrun.log_metric(
            f"fold_{fold}_test_acc",
            test_metrics["classification_accuracy"],
            step=metric_step,
        )

        total_inter += test_metrics["intersection"]
        total_pred += test_metrics["pred_area"]
        total_gt += test_metrics["gt_area"]
        total_positive_inter += test_metrics["positive_intersection"]
        total_positive_pred += test_metrics["positive_pred_area"]
        total_positive_gt += test_metrics["positive_gt_area"]
        total_correct += test_metrics["correct"]
        total_pixels += test_metrics["total"]

    if args.skip_eval:
        mlrun.end()
        print("training complete; held-out evaluation skipped")
        return 0

    pooled = (2 * total_inter + eps) / (total_pred + total_gt + eps)
    pooled_positive = (2 * total_positive_inter + eps) / (
        total_positive_pred + total_positive_gt + eps
    )
    pooled_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
    report = {
        "pooled_dice": pooled,
        "pooled_positive_dice": pooled_positive,
        "pooled_acc": pooled_acc,
        "checkpoint_selection_metric": "validation_positive_dice",
        "classification_metric_note": CLASSIFICATION_METRIC_NOTE,
        "per_fold": per_fold,
    }
    pooled_name = (
        f"pooled_metrics_fold{folds[0]}.json"
        if len(folds) == 1
        else "pooled_metrics.json"
    )
    pooled_path = args.output / pooled_name
    pooled_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    aggregate = aggregate_completed_fold_metrics(args.output, args.n_folds)
    mlrun.log_metric("pooled_dice", pooled)
    mlrun.log_metric("pooled_positive_dice", pooled_positive)
    mlrun.log_metric("pooled_acc", pooled_acc)
    mlrun.end()

    print(
        f"\npooled BEST-checkpoint test Dice: {pooled:.4f} | "
        f"positive Dice: {pooled_positive:.4f} | "
        f"classification Acc: {pooled_acc:.4f} (diagnostic only)"
    )
    print(f"metrics written to {pooled_path}")
    if aggregate is None:
        completed = sum(
            (args.output / f"fold_{fold}_metrics.json").exists()
            for fold in range(args.n_folds)
        )
        print(
            f"aggregate pending: {completed}/{args.n_folds} fold reports complete"
        )
    else:
        print(f"five-fold mean/std written to {args.output / 'aggregate_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
