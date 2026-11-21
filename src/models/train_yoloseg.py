"""Train the joint Relational YOLO-SegNet (Track B) across folds.

Unlike the two-stage pipeline (a frozen, pretrained Ultralytics detector
feeding a separate head), yoloseg_relational.YOLOSegNet is one model: its own
backbone, neck, three detection heads, and four relational stages, all trained
together from a single combined loss. There is no frozen component and no
detector-handoff step.

Per batch: the model's own detection heads (out_high/out_mid/out_low) are
supervised by YOLODetectionLoss (box CIoU + per-cell classification), its mask
output by bce_dice_loss (the paper's segmentation objective), and its GAP
classification head (Normal vs Tumour, one logit per image) by binary
cross-entropy against the image's tumour-present label.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.yolo_detection_loss import YOLODetectionLoss
from models.yoloseg_relational import YOLOSegNet
from utils.losses import bce_dice_loss, dice_loss

PRESENCE_MIN_PIXELS = 20  # mask pixels above which an image counts as "tumour"


class MLflowRun:
    """Optional MLflow logging that never stops a training run."""

    def __init__(
        self, enabled: bool, experiment_name: str, run_name: str, tracking_uri: str = ""
    ):
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
    """Serves images, the segmentation mask, and detection targets.

    Reads the same materialised-fold layout as the rest of the pipeline
    (fold_k/role/images, fold_k/role/labels; polygon label lines, one box per
    line, normalised [0, 1] coordinates). Produces:
      image     (3, S, S) grayscale replicated to RGB, matching what YOLO-style
                 backbones expect.
      mask      (S, S) union of every box in the file, for segmentation.
      gt_box    (4,) the FIRST box in the file (or zeros if none), for
                 detection supervision - see the module docstring's limitation.
      gt_label  scalar, 1.0 if the file has any box, 0.0 otherwise.
    """

    def __init__(self, fold_dir: Path, role: str, image_size: int = 512):
        self.images_dir = fold_dir / role / "images"
        self.labels_dir = fold_dir / role / "labels"
        self.size = image_size
        self.paths = sorted(self.images_dir.glob("*.png")) + sorted(
            self.images_dir.glob("*.jpg")
        )

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
        rgb = np.repeat(img[None], 3, axis=0).astype(np.float32) / 255.0

        mask, gt_box, has_box = self._read_label(self.labels_dir / f"{path.stem}.txt")
        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(mask).float(),
            "gt_box": torch.from_numpy(gt_box),
            "gt_label": torch.tensor([1.0 if has_box else 0.0]),
        }


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def image_level_correct(
    pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: int = PRESENCE_MIN_PIXELS
) -> int:
    """Count images where predicted tumour-presence matches the ground truth."""
    pred_area = pred_mask.flatten(1).sum(dim=1)
    gt_area = gt_mask.flatten(1).sum(dim=1)
    return int(((pred_area > threshold) == (gt_area > 0)).sum().item())


def head_correct(cls_logits: torch.Tensor, gt_label: torch.Tensor) -> int:
    """Count images where the classification head's call matches the label."""
    return int(((cls_logits > 0) == (gt_label > 0.5)).sum().item())


def seg_objective(
    mask_logits: torch.Tensor, gt_mask: torch.Tensor, pos_weight: float
) -> torch.Tensor:
    """BCE + Dice. With pos_weight > 1, tumour pixels count more in the BCE term.

    Tumour covers about 1.7% of each image, so plain BCE can be lowered by
    predicting background everywhere; the Dice term gives little gradient on
    positive images once probabilities are near zero. Weighting the positive
    pixels stops that collapse (checked on real fold-0 data).
    """
    if pos_weight == 1.0:
        return bce_dice_loss(mask_logits, gt_mask)
    bce = F.binary_cross_entropy_with_logits(
        mask_logits,
        gt_mask,
        pos_weight=torch.tensor(pos_weight, device=mask_logits.device),
    )
    return 0.5 * bce + 0.5 * dice_loss(mask_logits, gt_mask)


def train_one_fold(
    fold_dir: Path,
    model: YOLOSegNet,
    det_loss_fn: YOLODetectionLoss,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seg_weight: float,
    det_weight: float,
    mlrun: MLflowRun | None,
    fold_idx: int,
    cls_weight: float = 1.0,
    pos_weight: float = 1.0,
) -> YOLOSegNet:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        FoldDataset(fold_dir, "train", model.image_size),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    model.train()
    for epoch in range(epochs):
        running_total = running_seg = running_bbox = running_cls = 0.0
        running_imgcls = 0.0
        correct_head = correct_mask = total = 0
        for batch in loader:
            images = batch["image"].to(device)
            gt_mask = batch["mask"].unsqueeze(1).to(device)
            targets = {
                "gt_boxes": batch["gt_box"].to(device),
                "gt_labels": batch["gt_label"].to(device),
            }

            opt.zero_grad()
            mask_logits, out_high, out_mid, out_low, cls_logits = model(images)
            seg_loss = seg_objective(mask_logits, gt_mask, pos_weight)
            det_joint, det_bbox, det_cls = det_loss_fn(
                (out_high, out_mid, out_low), targets
            )
            img_cls_loss = F.binary_cross_entropy_with_logits(
                cls_logits, targets["gt_labels"]
            )
            loss = (
                seg_weight * seg_loss
                + det_weight * det_joint
                + cls_weight * img_cls_loss
            )
            loss.backward()
            opt.step()

            running_total += loss.item()
            running_seg += seg_loss.item()
            running_bbox += det_bbox.item()
            running_cls += det_cls.item()
            running_imgcls += img_cls_loss.item()
            with torch.no_grad():
                pred = (mask_logits > 0).float()
                correct_mask += image_level_correct(pred, gt_mask)
                correct_head += head_correct(cls_logits, targets["gt_labels"])
                total += images.shape[0]

        n = len(loader)
        acc = correct_head / total if total > 0 else 0.0
        acc_mask = correct_mask / total if total > 0 else 0.0
        print(
            f" fold {fold_dir.name} | epoch {epoch + 1}/{epochs} | "
            f"loss {running_total / n:.4f} | seg {running_seg / n:.4f} | "
            f"bbox {running_bbox / n:.4f} | cls {running_cls / n:.4f} | "
            f"imgcls {running_imgcls / n:.4f} | acc {acc:.4f} | "
            f"acc_mask {acc_mask:.4f}"
        )
        if mlrun is not None:
            mlrun.log_metric(
                f"fold_{fold_idx}_train_loss", running_total / n, step=epoch
            )
            mlrun.log_metric(
                f"fold_{fold_idx}_train_seg_loss", running_seg / n, step=epoch
            )
            mlrun.log_metric(
                f"fold_{fold_idx}_train_bbox_loss", running_bbox / n, step=epoch
            )
            mlrun.log_metric(
                f"fold_{fold_idx}_train_cls_loss", running_cls / n, step=epoch
            )
            mlrun.log_metric(
                f"fold_{fold_idx}_train_imgcls_loss", running_imgcls / n, step=epoch
            )
            mlrun.log_metric(f"fold_{fold_idx}_train_acc", acc, step=epoch)
            mlrun.log_metric(f"fold_{fold_idx}_train_acc_mask", acc_mask, step=epoch)
    return model


@torch.no_grad()
def evaluate_fold(
    fold_dir: Path, model: YOLOSegNet, device: str, batch_size: int, role: str = "test"
) -> tuple[int, int, int, int, int, int]:
    """Return (intersection, pred_area, gt_area, correct_head, correct_mask,
    total) for one fold. correct_head counts images the GAP classification head
    got right (the paper's classification metric); correct_mask counts images
    where a mask of more than PRESENCE_MIN_PIXELS pixels matched the label."""
    model.eval()
    loader = DataLoader(
        FoldDataset(fold_dir, role, model.image_size),
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
    )
    inter = pred_area = gt_area = correct_head = correct_mask = total = 0
    for batch in loader:
        images = batch["image"].to(device)
        gt_mask = batch["mask"].unsqueeze(1).to(device)
        mask_logits, _, _, _, cls_logits = model(images)
        pred = (mask_logits.sigmoid() > 0.5).float()
        inter += int((pred * gt_mask).sum().item())
        pred_area += int(pred.sum().item())
        gt_area += int(gt_mask.sum().item())
        correct_mask += image_level_correct(pred, gt_mask)
        correct_head += head_correct(cls_logits, batch["gt_label"].to(device))
        total += images.shape[0]
    return inter, pred_area, gt_area, correct_head, correct_mask, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the joint Relational YOLO-SegNet (Track B)."
    )
    p.add_argument("--folds-root", type=Path, default=Path("outputs/ultralytics_folds"))
    p.add_argument("--output", type=Path, default=Path("runs/yoloseg_relational"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument(
        "--pos-weight",
        type=float,
        default=55.0,
        help="BCE weight on tumour pixels (about (1-p)/p; tumour is "
        "~1.7%% of pixels). 1.0 turns the weighting off.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=512)

    model_args = p.add_argument_group("model configuration")
    model_args.add_argument("--embed-dim", type=int, default=128)
    model_args.add_argument("--num-heads", type=int, default=8)
    model_args.add_argument("--dilation", type=float, default=1.5)
    model_args.add_argument("--n-regions", type=int, default=4)
    model_args.add_argument("--topk", type=int, default=4)
    model_args.add_argument(
        "--box-conf",
        type=float,
        default=0.5,
        help="confidence used to build ROI masks internally",
    )
    model_args.add_argument("--max-roi-tokens", type=int, default=1024)
    model_args.add_argument(
        "--reduction",
        type=int,
        default=4,
        help="key/value sequence reduction r in the RTrB (Eqs. 8-9); 1 turns it off",
    )
    model_args.add_argument(
        "--ffn-expansion",
        type=int,
        default=4,
        help="Mix-FFN hidden width as a multiple of channels",
    )
    model_args.add_argument(
        "--width-multiple",
        type=float,
        default=1.0,
        help="scales channels in the backbone, neck and "
        "detection heads (YOLOv8: n .25, s .50, m .75, "
        "l 1.0, x 1.25); BiLRA's dimension follows it",
    )
    model_args.add_argument(
        "--depth-multiple",
        type=float,
        default=1.0,
        help="scales C2f repeats (YOLOv8: n/s .33, m .67, l/x 1.0)",
    )

    loss_args = p.add_argument_group("loss weighting")
    loss_args.add_argument("--seg-weight", type=float, default=1.0)
    loss_args.add_argument("--det-weight", type=float, default=1.0)
    loss_args.add_argument(
        "--cls-weight",
        type=float,
        default=1.0,
        help="weight of the image-level classification loss",
    )

    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--mlflow-experiment", default="yoloseg_relational")
    p.add_argument("--mlflow-run-name", default="joint_cv")
    p.add_argument("--mlflow-uri", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mlrun = MLflowRun(
        not args.no_mlflow,
        args.mlflow_experiment,
        args.mlflow_run_name,
        args.mlflow_uri,
    )
    mlrun.log_params(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "pos_weight": args.pos_weight,
            "image_size": args.image_size,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "dilation": args.dilation,
            "seg_weight": args.seg_weight,
            "det_weight": args.det_weight,
            "box_conf": args.box_conf,
            "reduction": args.reduction,
            "ffn_expansion": args.ffn_expansion,
            "cls_weight": args.cls_weight,
            "max_roi_tokens": args.max_roi_tokens,
            "width_multiple": args.width_multiple,
            "depth_multiple": args.depth_multiple,
        }
    )

    eps = 1e-7
    total_inter = total_pred = total_gt = total_images = 0
    total_correct_head = total_correct_mask = 0
    per_fold = {}

    for fold in range(args.n_folds):
        fold_dir = args.folds_root / f"fold_{fold}"
        print(f"=== training fold {fold} (joint model) ===")

        model = YOLOSegNet(
            image_size=args.image_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            dilation=args.dilation,
            n_regions=args.n_regions,
            topk=args.topk,
            box_conf=args.box_conf,
            reduction=args.reduction,
            ffn_expansion=args.ffn_expansion,
            max_roi_tokens=args.max_roi_tokens,
            width_multiple=args.width_multiple,
            depth_multiple=args.depth_multiple,
        ).to(args.device)
        print(
            f" model: width {args.width_multiple}, depth {args.depth_multiple}, "
            f"BiLRA dim {model.backbone_tail_channels}, "
            f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
        )
        det_loss_fn = YOLODetectionLoss(num_classes=1)

        model = train_one_fold(
            fold_dir,
            model,
            det_loss_fn,
            args.device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.seg_weight,
            args.det_weight,
            mlrun,
            fold,
            cls_weight=args.cls_weight,
            pos_weight=args.pos_weight,
        )
        torch.save(model.state_dict(), args.output / f"model_fold{fold}.pth")

        inter, pred, gt, correct_head, correct_mask, total = evaluate_fold(
            fold_dir, model, args.device, args.batch_size
        )
        fold_dice = (2 * inter + eps) / (pred + gt + eps)
        fold_acc = correct_head / total if total > 0 else 0.0
        fold_acc_mask = correct_mask / total if total > 0 else 0.0
        per_fold[fold] = {"dice": fold_dice, "acc": fold_acc, "acc_mask": fold_acc_mask}
        print(
            f" fold {fold} test dice {fold_dice:.4f} | acc {fold_acc:.4f} "
            f"| acc_mask {fold_acc_mask:.4f}"
        )
        mlrun.log_metric(f"fold_{fold}_test_dice", fold_dice, step=args.epochs)
        mlrun.log_metric(f"fold_{fold}_test_acc", fold_acc, step=args.epochs)
        mlrun.log_metric(f"fold_{fold}_test_acc_mask", fold_acc_mask, step=args.epochs)

        total_inter += inter
        total_pred += pred
        total_gt += gt
        total_correct_head += correct_head
        total_correct_mask += correct_mask
        total_images += total

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled_dice = (2 * total_inter + eps) / (total_pred + total_gt + eps)
    pooled_acc = total_correct_head / total_images if total_images > 0 else 0.0
    pooled_acc_mask = total_correct_mask / total_images if total_images > 0 else 0.0
    report = {
        "pooled_dice": pooled_dice,
        "pooled_acc": pooled_acc,
        "pooled_acc_mask": pooled_acc_mask,
        "per_fold": per_fold,
    }
    (args.output / "pooled_metrics.json").write_text(json.dumps(report, indent=2))
    mlrun.log_metric("pooled_dice", pooled_dice)
    mlrun.log_metric("pooled_acc", pooled_acc)
    mlrun.log_metric("pooled_acc_mask", pooled_acc_mask)
    mlrun.end()

    print(
        f"\npooled Dice: {pooled_dice:.4f} | pooled Acc: {pooled_acc:.4f} "
        f"| pooled Acc (mask-derived): {pooled_acc_mask:.4f}"
    )
    print(f"metrics written to {args.output / 'pooled_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
