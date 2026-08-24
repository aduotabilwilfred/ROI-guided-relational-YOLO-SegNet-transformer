from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2

CLASS_ID = 0
CLASS_NAME = "cancer"


def read_dataset_csv(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def polygon_lines(
    json_path: Path, fallback_h: int, fallback_w: int
) -> tuple[list[str], int]:
    """
    Return YOLO polygon lines (one per traced tumour) and the shape count.

    Uses the image dimensions recorded in the annotation itself; falls back to
    the actual image's dimensions if the annotation is missing them or they
    disagree with a 0-pixel read.
    """
    data = json.loads(json_path.read_text())
    h = data.get("imageHeight") or fallback_h
    w = data.get("imageWidth") or fallback_w

    lines = []
    n_polygons = 0
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        n_polygons += 1
        coords = []
        for px, py in points:
            nx = min(1.0, max(0.0, px / w))
            ny = min(1.0, max(0.0, py / h))
            coords.extend((nx, ny))
        lines.append(f"{CLASS_ID} " + " ".join(f"{c:.6f}" for c in coords))
    return lines, n_polygons


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    data_dir: Path,
    output_dir: Path,
    split_name: str,
) -> dict:
    images_root = data_dir / "images"
    annotations_root = data_dir / "Annotations"
    csv_path = data_dir / "dataset.csv"

    rows = read_dataset_csv(csv_path)

    out_images = output_dir / split_name / "images"
    out_labels = output_dir / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_rows": len(rows),
        "written": 0,
        "with_tumour": 0,
        "background": 0,
        "n_annotations": 0,
        "duplicates_dropped": [],
        "missing_image": [],
        "missing_annotation_but_tumour_flagged": [],
        "unreadable_image": [],
    }
    metadata_rows = []
    seen_hashes: dict[str, str] = {}  # content hash -> image_id kept

    for row in sorted(rows, key=lambda r: r["image_id"]):
        image_id = row["image_id"]
        stem = Path(image_id).stem
        src_image = images_root / image_id
        if not src_image.exists():
            stats["missing_image"].append(image_id)
            continue

        digest = file_hash(src_image)
        if digest in seen_hashes:
            stats["duplicates_dropped"].append((image_id, seen_hashes[digest]))
            continue
        seen_hashes[digest] = image_id

        image = cv2.imread(str(src_image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            stats["unreadable_image"].append(image_id)
            continue
        fallback_h, fallback_w = image.shape[:2]

        json_path = annotations_root / f"{stem}.json"
        has_tumour_flag = row.get("tumor") == "1"
        if json_path.exists():
            lines, n_polygons = polygon_lines(json_path, fallback_h, fallback_w)
        else:
            if has_tumour_flag:
                stats["missing_annotation_but_tumour_flagged"].append(image_id)
            lines, n_polygons = [], 0

        shutil.copy2(src_image, out_images / image_id)
        (out_labels / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

        stats["written"] += 1
        stats["n_annotations"] += n_polygons
        stats["with_tumour" if n_polygons else "background"] += 1

        metadata_rows.append(
            {
                "image_id": image_id,
                "tumor": int(n_polygons > 0),
                "n_instances": n_polygons,
                "malignant": row.get("malignant", ""),
                "benign": row.get("benign", ""),
                "center": row.get("center", ""),
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
            }
        )

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "tumor",
                "n_instances",
                "malignant",
                "benign",
                "center",
                "age",
                "gender",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(f"names:\n  0: {CLASS_NAME}\n")

    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(stats, indent=2))

    return stats


def print_report(stats: dict) -> None:
    print("=" * 74)
    print("BTXRD -> YOLO-label conversion")
    print("=" * 74)
    print(f"  rows in dataset.csv        : {stats['total_rows']}")
    print(f"  written                    : {stats['written']}")
    print(f"  with tumour                : {stats['with_tumour']}")
    print(f"  background                 : {stats['background']}")
    print(f"  polygon annotations        : {stats['n_annotations']}")
    print(f"  exact-duplicate images     : {len(stats['duplicates_dropped'])} dropped")
    if stats["duplicates_dropped"]:
        for dup, kept in stats["duplicates_dropped"][:5]:
            print(f"      {dup} == {kept} (kept {kept})")
    if stats["missing_image"]:
        print(f"  rows with no image file    : {len(stats['missing_image'])}")
    if stats["unreadable_image"]:
        print(f"  images that failed to read : {len(stats['unreadable_image'])}")
    if stats["missing_annotation_but_tumour_flagged"]:
        print(
            "  >> WARNING: tumor=1 in dataset.csv but no Annotations/*.json found for "
            f"{len(stats['missing_annotation_but_tumour_flagged'])} images:"
        )
        print(f"      {stats['missing_annotation_but_tumour_flagged'][:5]}")
    print("=" * 74)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/BTXRD"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/BTXRD_yolo"))
    parser.add_argument("--split-name", default="train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {args.data_dir}")

    stats = convert(args.data_dir, args.output_dir, args.split_name)
    print_report(stats)
    print(f"\nwritten to {args.output_dir}")
    print(
        "\nnext: python src/data/prepare_dataset.py --data-dir "
        f"{args.output_dir} --inspect"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
