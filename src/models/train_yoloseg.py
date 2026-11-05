"""Train the joint Relational YOLO-SegNet (Track B) across folds.

Unlike the two-stage pipeline (a frozen, pretrained Ultralytics detector
feeding a separate head), yoloseg_relational.YOLOSegNet is one model: its own
backbone, neck, three detection heads, and four relational stages, all trained
together from a single combined loss. There is no frozen component and no
detector-handoff step.

Per batch: the model's own detection heads (out_high/out_mid/out_low) are
supervised by YOLODetectionLoss (box CIoU + classification), and its mask
output is supervised by bce_dice_loss, the paper's segmentation objective.
Evaluation pools predictions across the five test folds into one Dice and one
image-level tumour-presence accuracy, matching the paper's metric.

Known limitation, carried from YOLODetectionLoss's contract: it supervises one
ground-truth box per image. A label file with more than one box has its first
box used for detection supervision; the segmentation mask itself still unions
every box in the file, so multi-tumour images are not silently dropped from
segmentation, only from the box-regression signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.yolo_detection_loss import YOLODetectionLoss  # noqa: E402
from models.yoloseg_relational import YOLOSegNet  # noqa: E402
from utils.losses import bce_dice_loss  # noqa: E402

PRESENCE_MIN_PIXELS = 20   # mask pixels above which an image counts as "tumour"


class MLflowRun:

    def __init__(self, enabled: bool, experiment_name: str, run_name: str,
                tracking_uri: str = ""):
        self.enabled = enabled
        self.mlflow = None
        if not enabled:
            return
        try:
            import mlflow
            self.mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri or "sqlite:///mlflow.db")
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)
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


class FoldDataset(Dataset):

    def __init__(self, fold_dir: Path, role: str, image_size: int = 512):
        self.images_dir = fold_dir / role / "images"
        self.labels_dir = fold_dir / role / "labels"
        self.size = image_size
        self.paths = sorted(self.images_dir.glob("*.png")) + \
            sorted(self.images_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.paths)

    def _read_label(self, label_path: Path):
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        first_box = np.zeros(4, dtype=np.float32)
        has_box = False
        if not label_path.exists():
            return mask, first_box, has_box
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            coords = np.array([float(v) for v in parts[1:]], dtype=np.float64)
            pts = coords.reshape(-1, 2)
            cv2.fillPoly(mask, [(pts * self.size).astype(np.int32)], 1)
            if not has_box:
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                first_box = np.array([x1, y1, x2, y2], dtype=np.float32)
                has_box = True
        return mask, first_box, has_box

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        rgb = np.repeat(img[None], 3, axis=0).astype(np.float32) / 255.0

        mask, gt_box, has_box = self._read_label(
            self.labels_dir / f"{path.stem}.txt")
        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(mask).float(),
            "gt_box": torch.from_numpy(gt_box),
            "gt_label": torch.tensor([1.0 if has_box else 0.0]),
        }


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def image_level_correct(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                        threshold: int = PRESENCE_MIN_PIXELS) -> int:
    """Count images where predicted tumour-presence matches the ground truth."""
    pred_area = pred_mask.flatten(1).sum(dim=1)
    gt_area = gt_mask.flatten(1).sum(dim=1)
    return int(((pred_area > threshold) == (gt_area > 0)).sum().item())


def train_one_fold(fold_dir: Path, model: YOLOSegNet, det_loss_fn: YOLODetectionLoss,
                   device: str, epochs: int, batch_size: int, lr: float,
                   seg_weight: float, det_weight: float,
                   mlrun: MLflowRun | None, fold_idx: int) -> YOLOSegNet:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(FoldDataset(fold_dir, "train", model.image_size),
                        batch_size=batch_size, shuffle=True, collate_fn=collate)

    model.train()
    for epoch in range(epochs):
        running_total = running_seg = running_bbox = running_cls = 0.0
        correct = total = 0
        for batch in loader:
            images = batch["image"].to(device)
            gt_mask = batch["mask"].unsqueeze(1).to(device)
            targets = {"gt_boxes": batch["gt_box"].to(device),
                      "gt_labels": batch["gt_label"].to(device)}

            opt.zero_grad()
            mask_logits, out_high, out_mid, out_low = model(images)
            seg_loss = bce_dice_loss(mask_logits, gt_mask)
            det_joint, det_bbox, det_cls = det_loss_fn(
                (out_high, out_mid, out_low), targets)
            loss = seg_weight * seg_loss + det_weight * det_joint
            loss.backward()
            opt.step()

            running_total += loss.item()
            running_seg += seg_loss.item()
            running_bbox += det_bbox.item()
            running_cls += det_cls.item()
            with torch.no_grad():
                pred = (mask_logits > 0).float()
                correct += image_level_correct(pred, gt_mask)
                total += images.shape[0]

        n = len(loader)
        acc = correct / total if total > 0 else 0.0
        print(f" fold {fold_dir.name} | epoch {epoch + 1}/{epochs} | "
              f"loss {running_total/n:.4f} | seg {running_seg/n:.4f} | "
              f"bbox {running_bbox/n:.4f} | cls {running_cls/n:.4f} | acc {acc:.4f}")
        if mlrun is not None:
            mlrun.log_metric(f"fold_{fold_idx}_train_loss", running_total/n, step=epoch)
            mlrun.log_metric(f"fold_{fold_idx}_train_seg_loss", running_seg/n, step=epoch)
            mlrun.log_metric(f"fold_{fold_idx}_train_bbox_loss", running_bbox/n, step=epoch)
            mlrun.log_metric(f"fold_{fold_idx}_train_cls_loss", running_cls/n, step=epoch)
            mlrun.log_metric(f"fold_{fold_idx}_train_acc", acc, step=epoch)
    return model


@torch.no_grad()
def evaluate_fold(fold_dir: Path, model: YOLOSegNet, device: str, batch_size: int,
                  role: str = "test") -> tuple[int, int, int, int, int]:
    """Return (intersection, pred_area, gt_area, correct, total) for one fold."""
    model.eval()
    loader = DataLoader(FoldDataset(fold_dir, role, model.image_size),
                        batch_size=batch_size, collate_fn=collate, shuffle=False)
    inter = pred_area = gt_area = correct = total = 0
    for batch in loader:
        images = batch["image"].to(device)
        gt_mask = batch["mask"].unsqueeze(1).to(device)
        mask_logits = model(images)[0]
        pred = (mask_logits.sigmoid() > 0.5).float()
        inter += int((pred * gt_mask).sum().item())
        pred_area += int(pred.sum().item())
        gt_area += int(gt_mask.sum().item())
        correct += image_level_correct(pred, gt_mask)
        total += images.shape[0]
    return inter, pred_area, gt_area, correct, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the joint Relational YOLO-SegNet (Track B).")
    p.add_argument("--folds-root", type=Path,
                   default=Path("outputs/ultralytics_folds"))
    p.add_argument("--output", type=Path, default=Path("runs/yoloseg_relational"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=512)

    model_args = p.add_argument_group("model configuration")
    model_args.add_argument("--embed-dim", type=int, default=128)
    model_args.add_argument("--num-heads", type=int, default=8)
    model_args.add_argument("--dilation", type=float, default=1.5)
    model_args.add_argument("--n-regions", type=int, default=4)
    model_args.add_argument("--topk", type=int, default=4)
    model_args.add_argument("--box-conf", type=float, default=0.5,
                            help="confidence used to build ROI masks internally")
    model_args.add_argument("--max-roi-tokens", type=int, default=1024)

    loss_args = p.add_argument_group("loss weighting")
    loss_args.add_argument("--seg-weight", type=float, default=1.0)
    loss_args.add_argument("--det-weight", type=float, default=1.0)

    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default="yoloseg_relational")
    p.add_argument("--mlflow-run-name", default="joint_cv")
    p.add_argument("--mlflow-uri", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mlrun = MLflowRun(not args.no_mlflow, args.mlflow_experiment,
                      args.mlflow_run_name, args.mlflow_uri)
    mlrun.log_params({
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "image_size": args.image_size, "embed_dim": args.embed_dim,
        "num_heads": args.num_heads, "dilation": args.dilation,
        "seg_weight": args.seg_weight, "det_weight": args.det_weight,
        "box_conf": args.box_conf,
    })

    eps = 1e-7
    total_inter = total_pred = total_gt = total_correct = total_images = 0
    per_fold = {}

    for fold in range(args.n_folds):
        fold_dir = args.folds_root / f"fold_{fold}"
        print(f"=== training fold {fold} (joint model) ===")

        model = YOLOSegNet(
            image_size=args.image_size, embed_dim=args.embed_dim,
            num_heads=args.num_heads, dilation=args.dilation,
            n_regions=args.n_regions, topk=args.topk,
            box_conf=args.box_conf).to(args.device)
        det_loss_fn = YOLODetectionLoss(num_classes=1)

        model = train_one_fold(
            fold_dir, model, det_loss_fn, args.device, args.epochs,
            args.batch_size, args.lr, args.seg_weight, args.det_weight,
            mlrun, fold)
        torch.save(model.state_dict(), args.output / f"model_fold{fold}.pth")

        inter, pred, gt, correct, total = evaluate_fold(
            fold_dir, model, args.device, args.batch_size)
        fold_dice = (2 * inter + eps) / (pred + gt + eps)
        fold_acc = correct / total if total > 0 else 0.0
        per_fold[fold] = {"dice": fold_dice, "acc": fold_acc}
        print(f" fold {fold} test dice {fold_dice:.4f} | acc {fold_acc:.4f}")
        mlrun.log_metric(f"fold_{fold}_test_dice", fold_dice, step=args.epochs)
        mlrun.log_metric(f"fold_{fold}_test_acc", fold_acc, step=args.epochs)

        total_inter += inter
        total_pred += pred
        total_gt += gt
        total_correct += correct
        total_images += total

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled_dice = (2 * total_inter + eps) / (total_pred + total_gt + eps)
    pooled_acc = total_correct / total_images if total_images > 0 else 0.0
    report = {"pooled_dice": pooled_dice, "pooled_acc": pooled_acc,
             "per_fold": per_fold}
    (args.output / "pooled_metrics.json").write_text(json.dumps(report, indent=2))
    mlrun.log_metric("pooled_dice", pooled_dice)
    mlrun.log_metric("pooled_acc", pooled_acc)
    mlrun.end()

    print(f"\npooled Dice: {pooled_dice:.4f} | pooled Acc: {pooled_acc:.4f}")
    print(f"metrics written to {args.output / 'pooled_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
