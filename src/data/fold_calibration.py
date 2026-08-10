from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
from make_folds import cv_roles, load_manifest
from osgdf_preprocessing import optimize_sg_parameters
from prepare_dataset import Letterbox


def _load_resized_gray(image_path: Path, target: int) -> np.ndarray:
    """Read an image as grayscale in [0, 1] and letterbox it to target size."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not read image: {image_path}")
    h, w = image.shape[:2]
    transform = Letterbox.build(h, w, target)
    resized = transform.apply_image(image, pad_value=0)
    return resized.astype(np.float64) / 255.0


def training_images_for_fold(
    records: list[dict], fold: int, n_folds: int
) -> list[Path]:
    """Return the image paths that play the training role in this iteration."""
    roles = cv_roles(fold, n_folds)
    return [r["image_path"] for r in records if roles[r["fold"]] == "train"]


def calibrate_fold(
    records: list[dict],
    fold: int,
    n_folds: int,
    target: int,
    sample_size: int,
    n_agents: int,
    n_iterations: int,
    seed: int,
) -> dict:
    """Calibrate OSGDF for one fold and return its parameter record.

    A random subset of the fold's training images is used for calibration; the
    filter parameters are global, so tuning on a representative sample gives the
    same result as tuning on all of them at a fraction of the cost.
    """
    train_paths = training_images_for_fold(records, fold, n_folds)

    rng = np.random.default_rng(seed + fold)
    if sample_size and sample_size < len(train_paths):
        chosen_idx = rng.choice(len(train_paths), size=sample_size, replace=False)
        calibration_paths = [train_paths[i] for i in sorted(chosen_idx)]
    else:
        calibration_paths = train_paths

    images = [_load_resized_gray(p, target) for p in calibration_paths]

    result = optimize_sg_parameters(
        images=images,
        n_agents=n_agents,
        n_iterations=n_iterations,
        seed=seed + fold,
        verbose=False,
        verify_with_grid=True,
    )

    return {
        "fold": fold,
        "window_size": result["window_size"],
        "poly_order": result["poly_order"],
        "fitness": result["fitness"],
        "n_training_images": len(train_paths),
        "n_calibration_images": len(calibration_paths),
        "target_size": target,
    }


def calibrate_all_folds(
    manifest_path: Path, target: int, sample_size: int, n_agents: int, n_iterations: int
) -> dict:
    records, seed, n_folds = load_manifest(manifest_path)

    fold_params = []
    for fold in range(n_folds):
        params = calibrate_fold(
            records, fold, n_folds, target, sample_size, n_agents, n_iterations, seed
        )
        fold_params.append(params)
        print(
            f"fold {fold}: window {params['window_size']}, "
            f"poly {params['poly_order']}, "
            f"fitness {params['fitness']:.3f}, "
            f"calibrated on {params['n_calibration_images']} of "
            f"{params['n_training_images']} training images"
        )

    return {
        "seed": seed,
        "n_folds": n_folds,
        "target_size": target,
        "folds": fold_params,
    }


def load_fold_params(path: Path) -> dict[int, dict]:
    """Read fold_params.json into a {fold_index: params} mapping."""
    data = json.loads(Path(path).read_text())
    return {p["fold"]: p for p in data["folds"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate OSGDF parameters per cross-validation fold."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/folds/cv_manifest.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/folds/fold_params.json")
    )
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=40,
        help="training images sampled per fold for calibration",
    )
    parser.add_argument("--agents", type=int, default=25, help="TSO population size")
    parser.add_argument("--iters", type=int, default=60, help="TSO iterations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(
            f"manifest not found: {args.manifest}\nrun make_folds.py first"
        )

    result = calibrate_all_folds(
        args.manifest, args.target_size, args.sample_size, args.agents, args.iters
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nfold parameters written to {args.output}")
    print("next: build FoldDataset with this manifest and these parameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
