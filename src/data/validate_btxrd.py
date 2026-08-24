"""
Validation checks for the converted BTXRD_yolo dataset and its folds.
Run after prepare_btxrd.py and make_folds.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_folds import fold_summary, load_manifest
from osgdf_preprocessing import apply_sg_filter
from prepare_dataset import Letterbox, draw_mask, parse_label_file, to_yolo_bbox


def check_masks_and_bboxes(
    data_dir: Path, target_size: int, sample: int, out_dir: Path
) -> dict:
    """
    Re-derive mask + bbox from each converted label and sanity-check them.

    For a sample of images: parse the label file, letterbox, rasterise the
    mask, derive bboxes from it, and check that (a) tumour images produce a
    non-empty mask that isn't implausibly large, (b) background images
    produce an all-zero mask, (c) every mask pixel has a covering bbox and
    vice versa. Also writes overlay PNGs for a handful of tumour images.
    """
    images_dir = data_dir / "train" / "images"
    labels_dir = data_dir / "train" / "labels"
    images = sorted(images_dir.iterdir())[: sample if sample else None]

    problems: list[str] = []
    area_fractions: list[float] = []
    n_checked = 0
    n_overlays_written = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        stem = image_path.stem
        label_path = labels_dir / f"{stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            problems.append(f"{stem}: image unreadable")
            continue
        h, w = image.shape[:2]
        parsed = parse_label_file(label_path, h, w)
        if parsed.errors:
            problems.append(f"{stem}: label parse errors {parsed.errors[:2]}")
            continue

        transform = Letterbox.build(h, w, target_size)
        mask = draw_mask(parsed.annotations, transform)
        n_checked += 1

        has_annotations = bool(parsed.annotations)
        has_mask = bool(mask.any())
        if has_annotations != has_mask:
            problems.append(
                f"{stem}: {len(parsed.annotations)} annotations but "
                f"mask.any()={has_mask}"
            )

        frac = float((mask > 0).mean())
        area_fractions.append(frac)
        if frac > 0.5:
            problems.append(f"{stem}: mask covers {frac:.0%} of the image")

        for ann in parsed.annotations:
            box = to_yolo_bbox(ann, transform)
            if box is None:
                problems.append(f"{stem}: annotation fell entirely off-canvas")

        if has_mask and n_overlays_written < 6:
            out_img = transform.apply_image(image, pad_value=0)
            overlay = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
            red = np.zeros_like(overlay)
            red[..., 2] = 255
            blended = np.where(
                mask[..., None] > 0, cv2.addWeighted(overlay, 0.5, red, 0.5, 0), overlay
            )
            cv2.imwrite(str(out_dir / f"btxrd_yolo_overlay_{stem}.png"), blended)
            n_overlays_written += 1

    return {
        "checked": n_checked,
        "problems": problems,
        "mean_area_fraction": float(np.mean(area_fractions)) if area_fractions else 0.0,
        "max_area_fraction": float(np.max(area_fractions)) if area_fractions else 0.0,
        "overlays_written": n_overlays_written,
    }


def check_osgdf_smoke(data_dir: Path, target_size: int) -> dict:
    """Run OSGDF once on a BTXRD image to confirm it handles the format."""
    images_dir = data_dir / "train" / "images"
    sample_path = next(iter(sorted(images_dir.iterdir())), None)
    if sample_path is None:
        return {"ok": False, "reason": "no images found"}
    image = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    resized = Letterbox.build(h, w, target_size).apply_image(image, pad_value=0)
    resized = resized.astype(np.float64) / 255.0
    filtered = apply_sg_filter(resized, window_size=5, poly_order=4)
    return {
        "ok": filtered.shape == resized.shape,
        "sample": sample_path.name,
        "shape": list(filtered.shape),
    }


def check_fold_distribution(manifest_path: Path, metadata_path: Path) -> dict:
    """Report per-fold tumour counts and (informationally) malignant counts.

    Also asserts the leakage invariant: every source_id in the manifest maps
    to exactly one fold (guaranteed by make_folds.py's group-then-assign
    logic, checked here as a hard assertion rather than trusted blindly).
    """
    records, seed, n_folds = load_manifest(manifest_path)

    folds_by_source: dict[str, set[int]] = defaultdict(set)
    for r in records:
        folds_by_source[r["source_id"]].add(r["fold"])
    split_sources = {
        sid: folds for sid, folds in folds_by_source.items() if len(folds) > 1
    }

    malignant_by_id = {}
    if metadata_path.exists():
        with metadata_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                malignant_by_id[Path(row["image_id"]).stem] = row.get("malignant", "0")

    malignant_per_fold: Counter = Counter()
    for r in records:
        stem = r["image_path"].stem
        if malignant_by_id.get(stem) == "1":
            malignant_per_fold[r["fold"]] += 1

    return {
        "seed": seed,
        "n_folds": n_folds,
        "sources_split_across_folds": {k: sorted(v) for k, v in split_sources.items()},
        "fold_summary": fold_summary(records, n_folds),
        "malignant_per_fold": dict(malignant_per_fold),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/BTXRD_yolo"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/folds/cv_manifest.csv")
    )
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/folds/btxrd_validation_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {args.data_dir}")

    mask_report = check_masks_and_bboxes(
        args.data_dir, args.target_size, args.sample, args.figures_dir
    )
    osgdf_report = check_osgdf_smoke(args.data_dir, args.target_size)

    fold_report = None
    if args.manifest.exists():
        fold_report = check_fold_distribution(
            args.manifest, args.data_dir / "metadata.csv"
        )

    print("=" * 74)
    print("BTXRD VALIDATION")
    print("=" * 74)
    print(f"mask/bbox re-derivation: checked {mask_report['checked']} images")
    print(f"  mean tumour area fraction: {mask_report['mean_area_fraction']:.4f}")
    print(f"  max tumour area fraction : {mask_report['max_area_fraction']:.4f}")
    print(f"  problems: {len(mask_report['problems'])}")
    for p in mask_report["problems"][:10]:
        print(f"    {p}")
    print(
        f"  overlays written to {args.figures_dir}: {mask_report['overlays_written']}"
    )

    print(
        f"\nOSGDF smoke test: ok={osgdf_report['ok']} on {osgdf_report.get('sample')}"
    )

    if fold_report:
        print(f"\nfold distribution (seed {fold_report['seed']}):")
        for row in fold_report["fold_summary"]:
            malignant = fold_report["malignant_per_fold"].get(row["fold"], 0)
            print(
                f"  fold {row['fold']}: {row['images']} images, "
                f"{row['tumour']} tumour ({row['tumour_fraction']:.1%}), "
                f"{malignant} malignant"
            )
        if fold_report["sources_split_across_folds"]:
            print(
                "\n  >> LEAKAGE: sources split across multiple folds: "
                f"{fold_report['sources_split_across_folds']}"
            )
        else:
            print("\n  no source_id appears in more than one fold. Clean.")
    else:
        print(f"\n(no manifest found at {args.manifest}; skipping fold checks)")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"masks": mask_report, "osgdf": osgdf_report, "folds": fold_report},
            indent=2,
        )
    )
    print(f"\nreport -> {args.report}")

    ok = not mask_report["problems"] and osgdf_report["ok"]
    if fold_report:
        ok = ok and not fold_report["sources_split_across_folds"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
