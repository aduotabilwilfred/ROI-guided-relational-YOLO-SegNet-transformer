"""Train the frozen YOLOv8-seg detector prerequisite for each CV fold."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data.make_folds import load_manifest


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


def train_one_fold(
    fold_dir: Path,
    model_arch: str,
    device: str,
    project: Path,
    extra: dict,
) -> Path:
    from ultralytics import YOLO

    model = YOLO(model_arch)
    train_args = dict(PAPER_TRAIN_ARGS)
    train_args.update(extra)
    model.train(
        data=str(fold_dir / "dataset.yaml"),
        device=device,
        # Ultralytics resolves relative projects under its configured
        # ``runs_dir``.  Use an absolute path so the detector handoff always
        # lands at runs/segment_cv/fold_N regardless of user settings.
        project=str(project.resolve()),
        name=fold_dir.name,
        exist_ok=True,
        **train_args,
    )
    return project / fold_dir.name / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg across folds.")
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/folds/cv_manifest.csv")
    )
    parser.add_argument(
        "--folds-root", type=Path, default=Path("outputs/ultralytics_folds")
    )
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--project", type=Path, default=Path("runs/segment_cv"))
    parser.add_argument("--device", default="")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--only-fold", type=int, default=-1, help="train one fold; -1 trains all"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _records, _seed, n_folds = load_manifest(args.manifest)
    if args.only_fold < -1 or args.only_fold >= n_folds:
        raise SystemExit(f"--only-fold must be between 0 and {n_folds - 1}")

    folds = range(n_folds) if args.only_fold < 0 else [args.only_fold]
    extra = {
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "workers": args.workers,
        "amp": args.amp,
    }
    for fold in folds:
        fold_dir = args.folds_root / f"fold_{fold}"
        print(f"\n=== training YOLO detector, fold {fold} ===")
        weights = train_one_fold(
            fold_dir, args.model, args.device, args.project, extra
        )
        print(f"fold {fold} weights: {weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
