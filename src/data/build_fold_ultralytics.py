from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold_calibration import load_fold_params
from make_folds import cv_roles, load_manifest
from osgdf_preprocessing import apply_sg_filter
from prepare_dataset import Letterbox


def denoise_to_disk(
    image_path: Path, out_path: Path, window: int, poly: int, target: int
) -> Letterbox:
    """Letterbox to target size, denoise with the fold's filter, write as PNG.

    PNG is lossless, so it does not add compression artefacts on top of the
    denoised result. Ordering (resize then denoise) matches the paper's
    preprocessing block operating on resized images.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"could not read image: {image_path}")
    h, w = image.shape[:2]
    transform = Letterbox.build(h, w, target)
    resized = transform.apply_image(image, pad_value=0)
    resized = resized.astype(np.float64) / 255.0

    filtered = apply_sg_filter(resized, window, poly)
    out = np.rint(np.clip(filtered, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(out_path), out):
        raise OSError(f"could not write image: {out_path}")
    return transform


def transform_label_text(text: str, transform: Letterbox) -> str:
    """Map YOLO segmentation polygons into the letterboxed image frame.

    Source label coordinates are normalised against the original image.  The
    shared ``Letterbox`` first maps them to pixels using the exact scale and
    integer padding applied to the image, after which this function normalises
    them against the final output canvas.
    """
    out_h, out_w = transform.output_size
    transformed_lines = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2:
            raise ValueError(
                f"line {line_number}: expected class plus at least three x,y pairs"
            )

        try:
            values = np.asarray(
                [float(value) for value in parts[1:]], dtype=np.float64
            )
        except ValueError as exc:
            raise ValueError(
                f"line {line_number}: polygon coordinates must be numeric"
            ) from exc
        if not np.isfinite(values).all():
            raise ValueError(f"line {line_number}: polygon coordinates must be finite")

        source_points = values.reshape(-1, 2)
        output_pixels = transform.apply_points(
            [(float(x), float(y)) for x, y in source_points]
        )
        output_points = np.asarray(output_pixels, dtype=np.float64)
        output_points[:, 0] = np.clip(output_points[:, 0] / out_w, 0.0, 1.0)
        output_points[:, 1] = np.clip(output_points[:, 1] / out_h, 0.0, 1.0)
        coordinates = " ".join(f"{value:.16g}" for value in output_points.ravel())
        transformed_lines.append(f"{parts[0]} {coordinates}")

    return "\n".join(transformed_lines) + ("\n" if transformed_lines else "")


def _normalised_transform_is_identity(transform: Letterbox) -> bool:
    """True when letterboxing leaves normalised coordinates unchanged."""
    out_h, out_w = transform.output_size
    corners = np.asarray(transform.apply_points([(0.0, 0.0), (1.0, 1.0)]))
    corners[:, 0] /= out_w
    corners[:, 1] /= out_h
    return bool(np.allclose(corners, ((0.0, 0.0), (1.0, 1.0))))


def place_label(
    label_path: Path, dest: Path, transform: Letterbox, use_symlink: bool
) -> None:
    """Write a label transformed into the output image coordinate frame.

    A symlink is only possible when normalised coordinates are unchanged, such
    as for a square source image.  Non-square images must receive a materialised
    transformed label even when ``--link`` was requested.

    A background image has no source label; an empty label file is written so
    Ultralytics registers it as a negative sample, not a missing one.
    """
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if not label_path.exists():
        dest.write_text("")
        return
    if use_symlink and _normalised_transform_is_identity(transform):
        try:
            dest.symlink_to(label_path.resolve())
            # Verify the link actually resolves; on NTFS it may not.
            if dest.read_text() is not None:
                return
        except (OSError, NotImplementedError):
            pass
    dest.write_text(transform_label_text(label_path.read_text(), transform))


def write_dataset_yaml(fold_dir: Path, class_names: list[str], has_test: bool) -> Path:
    """Write the dataset.yaml Ultralytics reads for this fold.

    `path` is the fold root; train/val/test are relative to it. Only train and
    val are required by Ultralytics; test is included for final evaluation.
    """
    lines = [
        f"path: {fold_dir.resolve()}",
        "train: train/images",
        "val: val/images",
    ]
    if has_test:
        lines.append("test: test/images")
    lines.append("names:")
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    yaml_path = fold_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def build_fold(
    records: list[dict],
    fold: int,
    n_folds: int,
    params: dict,
    out_root: Path,
    class_names: list[str],
    target: int,
    image_format: str,
    use_symlink: bool,
) -> dict:
    """Materialise one cross-validation iteration into an Ultralytics dataset."""
    roles = cv_roles(fold, n_folds)
    window, poly = int(params["window_size"]), int(params["poly_order"])
    fold_dir = out_root / f"fold_{fold}"

    counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        role = roles[record["fold"]]
        images_dir = fold_dir / role / "images"
        labels_dir = fold_dir / role / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        stem = record["image_path"].stem
        transform = denoise_to_disk(
            record["image_path"],
            images_dir / f"{stem}.{image_format}",
            window,
            poly,
            target,
        )
        place_label(
            record["label_path"],
            labels_dir / f"{stem}.txt",
            transform,
            use_symlink,
        )
        counts[role] += 1

    yaml_path = write_dataset_yaml(fold_dir, class_names, has_test=counts["test"] > 0)
    return {
        "fold": fold,
        "dir": str(fold_dir),
        "yaml": str(yaml_path),
        "window": window,
        "poly": poly,
        **counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialise per-fold denoised datasets for Ultralytics YOLOv8-seg."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/folds/cv_manifest.csv")
    )
    parser.add_argument(
        "--fold-params", type=Path, default=Path("outputs/folds/fold_params.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/ultralytics_folds")
    )
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument(
        "--class-names", nargs="+", default=["cancer"], help="class names in id order"
    )
    parser.add_argument(
        "--only-fold",
        type=int,
        default=-1,
        help="build a single fold instead of all (-1 = all)",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="symlink labels only when letterboxing leaves their normalised "
        "coordinates unchanged; transformed labels are always written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not args.fold_params.exists():
        raise SystemExit(
            f"fold params not found: {args.fold_params}\nrun fold_calibration.py first"
        )

    records, _, n_folds = load_manifest(args.manifest)
    fold_params = load_fold_params(args.fold_params)

    folds = range(n_folds) if args.only_fold < 0 else [args.only_fold]
    for fold in folds:
        summary = build_fold(
            records,
            fold,
            n_folds,
            fold_params[fold],
            args.output_dir,
            args.class_names,
            args.target_size,
            args.image_format,
            args.link,
        )
        print(
            f"fold {summary['fold']}: window {summary['window']}, "
            f"poly {summary['poly']} | "
            f"train {summary['train']}, val {summary['val']}, "
            f"test {summary['test']} -> {summary['yaml']}"
        )

    print(f"\ndatasets written under {args.output_dir}")
    print("train one fold with, e.g.:")
    print(
        f"  yolo segment train data={args.output_dir}/fold_0/dataset.yaml "
        f"model=yolov8n-seg.pt imgsz={args.target_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
