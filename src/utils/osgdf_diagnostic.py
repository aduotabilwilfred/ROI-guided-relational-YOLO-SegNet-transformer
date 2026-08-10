from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from fold_calibration import load_fold_params
from make_folds import cv_roles, load_manifest
from osgdf_preprocessing import apply_sg_filter, estimate_noise_sigma, snr_blind
from prepare_dataset import Letterbox


def load_resized(image_path: Path, target: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not read image: {image_path}")
    h, w = image.shape[:2]
    transform = Letterbox.build(h, w, target)
    return transform.apply_image(image, pad_value=0).astype(np.float64) / 255.0


def pick_instances(
    records: list[dict], fold: int, n_folds: int, n: int, prefer_tumour: bool
) -> list[dict]:
    """Choose n training images from this fold, preferring tumour-bearing ones."""
    roles = cv_roles(fold, n_folds)
    train = [r for r in records if roles[r["fold"]] == "train"]
    if prefer_tumour:
        tumour = [r for r in train if r["has_tumour"]]
        background = [r for r in train if not r["has_tumour"]]
        ordered = tumour + background
    else:
        ordered = train
    return ordered[:n]


def save_panel(
    original: np.ndarray, filtered: np.ndarray, title: str, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = original - filtered
    limit = float(np.abs(residual).max()) or 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original (resized)")
    axes[1].imshow(filtered, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(title)
    im = axes[2].imshow(residual, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title(f"removed (residual, max {limit:.3f})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise the calibrated OSGDF filter on real images."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/folds/cv_manifest.csv")
    )
    parser.add_argument(
        "--fold-params", type=Path, default=Path("outputs/folds/fold_params.json")
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/osgdf_diagnostic")
    )
    parser.add_argument(
        "--all-backgrounds",
        action="store_true",
        help="do not prefer tumour images; sample as-is",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, _, n_folds = load_manifest(args.manifest)
    params = load_fold_params(args.fold_params)[args.fold]
    window, poly = int(params["window_size"]), int(params["poly_order"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances = pick_instances(
        records,
        args.fold,
        n_folds,
        args.n_images,
        prefer_tumour=not args.all_backgrounds,
    )

    print(f"fold {args.fold}: OSGDF window {window}, poly {poly}")
    print(
        f"showing {len(instances)} images "
        f"(target {args.target_size}x{args.target_size})\n"
    )
    print(
        f"{'image':<28}  {'SNR before':>10}  {'SNR after':>9}  "
        f"{'gain':>6}  {'noise before':>12}  {'noise after':>11}"
    )

    gains = []
    for record in instances:
        stem = record["image_path"].stem
        original = load_resized(record["image_path"], args.target_size)
        filtered = apply_sg_filter(original, window, poly)

        snr_before = snr_blind(original)
        snr_after = snr_blind(filtered)
        noise_before = estimate_noise_sigma(original)
        noise_after = estimate_noise_sigma(filtered)
        gains.append(snr_after - snr_before)

        tag = "tumour" if record["has_tumour"] else "background"
        title = f"OSGDF w={window} p={poly}"
        save_panel(
            original, filtered, title, args.output_dir / f"fold{args.fold}_{stem}.png"
        )

        label = f"{stem[:22]} ({tag})"
        print(
            f"{label:<28}  {snr_before:>10.2f}  {snr_after:>9.2f}  "
            f"{snr_after - snr_before:>+6.2f}  {noise_before:>12.4f}  "
            f"{noise_after:>11.4f}"
        )

    print(f"\nmean SNR gain over {len(gains)} images: {np.mean(gains):+.2f} dB")
    print(f"panels written to {args.output_dir}")
    if abs(np.mean(gains)) < 0.5:
        print(
            "\nnote: SNR barely changes, which is consistent with the images "
            "already being low-noise after downscaling. Check the residual "
            "panels: near-blank residual means the filter is a light touch by "
            "necessity, not a failure of the search."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
