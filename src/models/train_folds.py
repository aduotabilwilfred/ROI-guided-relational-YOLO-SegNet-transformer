from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR / "data") not in sys.path:
    sys.path.insert(0, str(SRC_DIR / "data"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from make_folds import cv_roles, load_manifest
except ImportError:
    from src.data.make_folds import cv_roles, load_manifest


PAPER_TRAIN_ARGS = {
    "epochs": 50,
    "batch": 16,
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 0.01,
    "cos_lr": True,
    "imgsz": 512,
    "degrees": 10.0,
    "fliplr": 0.5,
    "scale": 0.5,
    "hsv_v": 0.2,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "flipud": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "translate": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "plots": True,
    "val": True,
}


def dice_from_counts(
    intersection: int, pred_area: int, gt_area: int, eps: float = 1e-7
) -> float:
    return (2.0 * intersection + eps) / (pred_area + gt_area + eps)


def load_gt_mask(label_path: Path, size: int) -> np.ndarray:
    """
    Rasterise a label file's polygons into a binary mask at size x size.

    The labels are already in the letterboxed target frame (the fold builder
    wrote images at that size and copied the matching labels), so the
    normalised polygon coordinates map directly onto the size x size canvas.
    """
    import cv2

    mask = np.zeros((size, size), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float64)
        pts = coords.reshape(-1, 2) * size
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def predicted_mask(result, size: int) -> np.ndarray:
    """
    Union of all predicted instance masks as a single binary mask.
    """
    if result.masks is None or len(result.masks.data) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    data = result.masks.data.cpu().numpy()  # (n, h, w) in {0,1}
    union = (data.sum(axis=0) > 0).astype(np.uint8)
    if union.shape != (size, size):
        import cv2

        union = cv2.resize(union, (size, size), interpolation=cv2.INTER_NEAREST)
    return union


def train_one_fold(
    fold_dir: Path, model_arch: str, device: str, project: Path, extra: dict
) -> Path:
    """
    Train one fold and return the path to its best weights."""
    from ultralytics import YOLO

    model = YOLO(model_arch)
    args = dict(PAPER_TRAIN_ARGS)
    args.update(extra)
    model.train(
        data=str(fold_dir / "dataset.yaml"),
        device=device,
        project=str(project),
        name=fold_dir.name,
        exist_ok=True,
        **args,
    )
    return project / fold_dir.name / "weights" / "best.pt"


def evaluate_pooled(
    records: list[dict],
    n_folds: int,
    folds_root: Path,
    weights: dict[int, Path],
    size: int,
    device: str,
) -> tuple[float, dict]:
    """Predict each fold's test images with its model; pool Dice over all."""
    from ultralytics import YOLO

    total_inter = total_pred = total_gt = 0
    per_fold = {}

    for fold in range(n_folds):
        model = YOLO(str(weights[fold]))
        roles = cv_roles(fold, n_folds)
        test_records = [r for r in records if roles[r["fold"]] == "test"]

        test_dir = folds_root / f"fold_{fold}" / "test"
        f_inter = f_pred = f_gt = 0

        for record in test_records:
            stem = record["image_path"].stem
            image_files = list((test_dir / "images").glob(f"{stem}.*"))
            if not image_files:
                raise FileNotFoundError(
                    f"Image for {stem} not found in {test_dir / 'images'}"
                )
            image_file = image_files[0]
            label_file = test_dir / "labels" / f"{stem}.txt"

            predict_kwargs = {"imgsz": size, "verbose": False}
            if device:
                predict_kwargs["device"] = device
            result = model.predict(str(image_file), **predict_kwargs)[0]
            pred = predicted_mask(result, size)
            gt = load_gt_mask(label_file, size)

            f_inter += int(np.logical_and(pred, gt).sum())
            f_pred += int(pred.sum())
            f_gt += int(gt.sum())

        per_fold[fold] = {
            "images": len(test_records),
            "dice": dice_from_counts(f_inter, f_pred, f_gt),
        }
        total_inter += f_inter
        total_pred += f_pred
        total_gt += f_gt

    pooled = dice_from_counts(total_inter, total_pred, total_gt)
    return pooled, {
        "pooled_dice": pooled,
        "per_fold": per_fold,
        "totals": {
            "intersection": total_inter,
            "pred_area": total_pred,
            "gt_area": total_gt,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg across folds and evaluate by pooling."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/folds/cv_manifest.csv")
    )
    parser.add_argument(
        "--folds-root", type=Path, default=Path("outputs/ultralytics_folds")
    )
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="YOLOv8-seg architecture or weights to start from",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/segment_cv"))
    parser.add_argument("--device", default="", help="'' auto, 'cpu', '0', '0,1' etc.")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override paper's 50 (e.g. for a quick test)",
    )
    parser.add_argument(
        "--only-fold",
        type=int,
        default=-1,
        help="train a single fold; skips pooled eval",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training, evaluate existing weights",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, _, n_folds = load_manifest(args.manifest)

    extra = {"imgsz": args.imgsz}
    if args.epochs is not None:
        extra["epochs"] = args.epochs

    folds = range(n_folds) if args.only_fold < 0 else [args.only_fold]
    weights: dict[int, Path] = {}

    for fold in folds:
        fold_dir = args.folds_root / f"fold_{fold}"
        weight_path = args.project / fold_dir.name / "weights" / "best.pt"
        if args.eval_only:
            weights[fold] = weight_path
            continue
        print(f"\n=== training fold {fold} ===")
        weights[fold] = train_one_fold(
            fold_dir, args.model, args.device, args.project, extra
        )
        print(f"fold {fold} weights: {weights[fold]}")

    if args.only_fold >= 0:
        print("\nsingle fold trained; skipping pooled evaluation")
        return 0

    print("\n Pooled evaluation over all test folds")
    pooled, report = evaluate_pooled(
        records, n_folds, args.folds_root, weights, args.imgsz, args.device
    )
    for fold, info in report["per_fold"].items():
        print(f"  fold {fold}: {info['images']} test images, Dice {info['dice']:.4f}")
    print(
        f"\npooled Dice over all {sum(r['images'] for r in report['per_fold'].values())} "
        f"test images: {pooled:.4f}"
    )

    out = args.project / "pooled_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"metrics written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
