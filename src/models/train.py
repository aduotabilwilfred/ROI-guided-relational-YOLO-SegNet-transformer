from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.train_handoff import FrozenYOLOFeatures
from utils.losses import bce_dice_loss
from utils.relational_head import HeadConfig, buildHead


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
) -> torch.nn.Module:
    detector = FrozenYOLOFeatures(str(weights), imgsz=cfg.image_size)
    detector.model.to(device)
    head = buildHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)

    train_ds = FoldImageDataset(fold_dir, "train", cfg.image_size)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    head.train()
    for epoch in range(epochs):
        running = 0.0
        train_inter = train_pred = train_gt = 0
        train_correct = train_total = 0
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["mask"].unsqueeze(1).to(device)

            """
            frozen detector runs twice per image
            once on the tensor to hook neck features, once on the file paths so ultralytics predict
            does its own letterbox + NMS for boxes. Both see identical content
            """
            feats = detector.extract_features(images)
            boxes = detector.detect_boxes(batch["path"])
            opt.zero_grad()
            logits = head(feats, boxes)
            loss = bce_dice_loss(logits, targets)
            loss.backward()
            opt.step()
            running += loss.item()

            with torch.no_grad():
                pred = (logits > 0).float()
                train_inter += int((pred * targets).sum().item())
                train_pred += int(pred.sum().item())
                train_gt += int(targets.sum().item())

                pred_per_img = pred.flatten(1).sum(dim=1)
                gt_per_img = targets.flatten(1).sum(dim=1)
                train_correct += int(
                    ((pred_per_img > 20) == (gt_per_img > 0)).sum().item()
                )
                train_total += images.shape[0]

        epoch_loss = running / len(loader)
        eps = 1e-7
        train_dice = (2 * train_inter + eps) / (train_pred + train_gt + eps)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        print(
            f" fold {fold_dir.name} | epoch {epoch + 1}/{epochs} | loss {epoch_loss:.4f} | dice {train_dice:.4f} | acc {train_acc:.4f}"
        )

        if mlrun is not None:
            #  global step across folds keeps curves distinct per fold
            mlrun.log_metric(f"fold_{fold_idx}_train_loss", epoch_loss, step=epoch)
            mlrun.log_metric(f"fold_{fold_idx}_train_dice", train_dice, step=epoch)
            mlrun.log_metric(f"fold_{fold_idx}_train_acc", train_acc, step=epoch)
    return head, detector


@torch.no_grad()
def accumulate_fold_dice(
    head,
    detector,
    fold_dir: Path,
    cfg: HeadConfig,
    device: str,
    batch_size: int,
    presence_min_pixels: int = 20,
    role: str = "test",
) -> tuple[int, int, int, float, int, int]:
    """
    Return (intersection, pred_area, gt_area, test_loss, correct, total) over a fold's test images
    """

    head.eval()
    ds = FoldImageDataset(fold_dir, role, cfg.image_size)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate, shuffle=False)

    inter = pred_area = gt_area = 0
    running_loss = 0.0
    correct = total = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].unsqueeze(1).to(device)

        feats = detector.extract_features(images)
        boxes = detector.detect_boxes(batch["path"])
        logits = head(feats, boxes)
        loss = bce_dice_loss(logits, targets)
        running_loss += loss.item()
        pred = (logits.sigmoid() > 0.5).float()

        inter += int((pred * targets).sum().item())
        pred_area += int(pred.sum().item())
        gt_area += int(targets.sum().item())

        pred_area_per_image = pred.flatten(1).sum(dim=1)
        gt_area_per_image = targets.flatten(1).sum(dim=1)
        pred_has_tumour = pred_area_per_image > presence_min_pixels
        actual_has_tumour = gt_area_per_image > 0
        correct += int((pred_has_tumour == actual_has_tumour).sum().item())
        total += targets.shape[0]

    return inter, pred_area, gt_area, running_loss / len(loader), correct, total


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
    parser.add_argument("--output", type=Path, default=Path("runs/relational_head"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--image-size", type=int, default=512)
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
    total_correct = total_pixels = 0
    eps = 1e-7
    per_fold = {}

    for fold in range(args.n_folds):
        fold_dir = args.folds_root / f"fold_{fold}"
        weights = args.detector_project / f"fold_{fold}/weights/best.pt"
        if not weights.exists():
            raise SystemExit(
                f"detector weights not found: {weights}\ntrain the YOLOv8 detector (train.py) first"
            )

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
        )

        torch.save(head.state_dict(), args.output / f"head_fold{fold}.pth")

        inter, pred, gt, test_loss, test_correct, test_total = accumulate_fold_dice(
            head=head,
            detector=detector,
            fold_dir=fold_dir,
            cfg=cfg,
            device=args.device,
            batch_size=args.batch_size,
        )

        fold_dice = (2 * inter + eps) / (pred + gt + eps)
        test_acc = test_correct / test_total if test_total > 0 else 0.0
        per_fold[fold] = {
            "dice": fold_dice,
            "acc": test_acc,
            "intersection": inter,
            "pred_area": pred,
            "gt_area": gt,
            "test_loss": test_loss,
        }
        print(
            f" fold {fold} test loss {test_loss:.4f} | dice {fold_dice:.4f} | acc {test_acc:.4f}"
        )

        mlrun.log_metric(f"fold_{fold}_test_loss", test_loss, step=args.epochs)
        mlrun.log_metric(f"fold_{fold}_test_dice", fold_dice, step=args.epochs)
        mlrun.log_metric(f"fold_{fold}_test_acc", test_acc, step=args.epochs)

        total_inter += inter
        total_pred += pred
        total_gt += gt
        total_correct += test_correct
        total_pixels += test_total

    pooled = (2 * total_inter + eps) / (total_pred + total_gt + eps)
    pooled_acc = total_correct / total_pixels if total_pixels > 0 else 0.0
    report = {"pooled_dice": pooled, "pooled_acc": pooled_acc, "per_fold": per_fold}
    (args.output / "pooled_metrics.json").write_text(json.dumps(report, indent=2))
    mlrun.log_metric("pooled_dice", pooled)
    mlrun.log_metric("pooled_acc", pooled_acc)
    mlrun.end()

    print(
        f"\npooled Dice over all test folds: {pooled:.4f} | pooled Acc: {pooled_acc:.4f}"
    )
    print(f"metrics written to {args.output / 'pooled_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
