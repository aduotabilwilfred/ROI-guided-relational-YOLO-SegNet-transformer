from __future__ import annotations

import sys
from pathlib import Dict, List, Path, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.inser(0, str(Path(__file__).resolve().parent))

from models.train_handoff import FrozenYOLOFeatures
from utils.losses import bce_dice_loss
from utils.relational_head import HeadConfig, buildHead


class MLflowRun:
    def __init__(
        self, enabled: bool, experiments: str, run_name: str, tracking_uri: str = ""
    ):
        self.enabled = enabled
        self.mlflow = None
        self.experiments = experiments
        self.run_name = run_name

        if not enabled:
            return
        try:
            import mlflow

            self.mlflow = mlflow

            uri = tracking_uri or "sqlite://mlflow.db"
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

    def __getitem__(self, index: int) -> Dict:
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


def collate(batch: List[Dict]) -> Dict:
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

        epoch_loss = running / len(loader)
        print(
            f" fold {fold_dir.name} | epoch {epoch + 1}/{epochs} | loss {epoch_loss:.4f} "
        )

        if mlrun is not None:
            #  global step across folds keeps curves distinct per fold
            mlrun.log_metric(f"fold_{fold_idx}_train_loss", epoch_loss, step=epoch)
    return head, detector


@torch.no_grad()
def accumulate_fold_dice(
    head, detector, fold_dir: Path, cfg: HeadConfig, device: str, batch_size: int
) -> Tuple[int, int, int]:
    """
    Return (intersection, pred_area, gt_area) over a fold's test images
    """

    head.eval()
    ds = FoldImageDataset(fold_dir, "test", cfg.image_size)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate, shuffle=False)

    inter = pred_area = gt_area = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].unsqueeze(1).to(device)

        feats = detector.extract_features(images)
        boxes = detector.detect_boxes(batch["path"])
        logits = head(feats, boxes)
        pred = (logits.sigmoid() > 0.5).float()

        inter += int((pred * targets).sum().item())
        pred_area += int(pred.sum().item())
        gt_area += int(targets.sum().item())

    return inter, pred_area, gt_area
