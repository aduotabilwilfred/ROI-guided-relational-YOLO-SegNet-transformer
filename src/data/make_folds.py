from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_dataset import (
    find_splits,
    list_images,
    source_image_id,
)

N_FOLDS = 5


def has_tumour(label_path: Path) -> bool:
    """True if the label file exists and contains at least one annotation.

    Presence is decided by non-empty content, not by parsing coordinates: an
    empty label file is a genuine background image, a missing file is treated
    the same way. This avoids reading the image and keeps folding fast.
    """
    if not label_path.exists():
        return False
    for line in label_path.read_text().splitlines():
        if line.strip():
            return True
    return False


def collect_images(data_dir: Path) -> list[dict]:
    """Gather every image across the detected splits with its label and flag.

    Returns one record per image: its path, the matching label path, the source
    id it belongs to, and whether it contains a tumour. The original split the
    image came from is kept only for reporting; fold assignment ignores it.
    """
    splits = find_splits(data_dir)
    if not splits:
        raise SystemExit(f"no split with both images/ and labels/ under {data_dir}")

    records = []
    for split in splits:
        labels_dir = data_dir / split / "labels"
        for image_path in list_images(data_dir / split):
            label_path = labels_dir / f"{image_path.stem}.txt"
            records.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "source_id": source_image_id(image_path.stem),
                    "has_tumour": has_tumour(label_path),
                    "origin_split": split,
                }
            )
    return records


def group_by_source(records: list[dict]) -> dict[str, list[dict]]:
    """Group image records by source id so copies stay together."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["source_id"]].append(record)
    return groups


def assign_folds(records: list[dict], n_folds: int, seed: int) -> None:
    """Assign each record a fold index, stratified by tumour presence.

    Works at the source-group level: each source is placed in one fold and all
    its images inherit that fold. Sources are stratified into two streams,
    tumour-bearing and background, and each stream is dealt round-robin across
    folds after a seeded shuffle. Round-robin on shuffled, sorted input gives
    near-equal fold sizes and near-equal tumour ratios without depending on the
    dataset being divisible by n_folds.
    """
    import random

    rng = random.Random(seed)
    groups = group_by_source(records)

    # A source counts as tumour-bearing if any of its images has a tumour.
    tumour_sources, background_sources = [], []
    for source_id, items in groups.items():
        stream = (
            tumour_sources
            if any(r["has_tumour"] for r in items)
            else background_sources
        )
        stream.append(source_id)

    for stream in (tumour_sources, background_sources):
        stream.sort()  # deterministic starting order
        rng.shuffle(stream)  # then seeded shuffle
        for position, source_id in enumerate(stream):
            fold = position % n_folds
            for record in groups[source_id]:
                record["fold"] = fold


def cv_roles(fold: int, n_folds: int = N_FOLDS) -> dict[int, str]:
    """Map every fold index to its role for one CV iteration.

    For iteration `fold`, that subset is the test set, the next one (mod
    n_folds) is validation, and the remaining three are training. Rotating
    `fold` from 0 to n_folds-1 makes each subset the test set exactly once.
    """
    test = fold
    val = (fold + 1) % n_folds
    roles = {}
    for i in range(n_folds):
        if i == test:
            roles[i] = "test"
        elif i == val:
            roles[i] = "val"
        else:
            roles[i] = "train"
    return roles


def write_manifest(records: list[dict], path: Path, seed: int, n_folds: int) -> None:
    """Write the fold assignment as CSV, newest columns last.

    One row per image. The fold column is the assignment; train/val/test roles
    are not stored because they depend on which iteration is being run and are
    derived from the fold via cv_roles() at read time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# seed", seed, "n_folds", n_folds])
        writer.writerow(
            [
                "fold",
                "has_tumour",
                "source_id",
                "origin_split",
                "image_path",
                "label_path",
            ]
        )
        for record in sorted(records, key=lambda r: (r["fold"], r["image_path"])):
            writer.writerow(
                [
                    record["fold"],
                    int(record["has_tumour"]),
                    record["source_id"],
                    record["origin_split"],
                    str(record["image_path"]),
                    str(record["label_path"]),
                ]
            )


def load_manifest(path: Path) -> tuple[list[dict], int, int]:
    """Read a manifest back. Returns (records, seed, n_folds).

    Downstream training code uses this plus cv_roles(fold) to get the
    train/val/test image lists for each of the five iterations.
    """
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        seed = int(header[1])
        n_folds = int(header[3])
        next(reader)  # column names
        records = []
        for row in reader:
            records.append(
                {
                    "fold": int(row[0]),
                    "has_tumour": bool(int(row[1])),
                    "source_id": row[2],
                    "origin_split": row[3],
                    "image_path": Path(row[4]),
                    "label_path": Path(row[5]),
                }
            )
    return records, seed, n_folds


def fold_summary(records: list[dict], n_folds: int) -> list[dict]:
    """Per-fold image count, tumour count, and tumour fraction."""
    summary = []
    for fold in range(n_folds):
        in_fold = [r for r in records if r["fold"] == fold]
        tumour = sum(1 for r in in_fold if r["has_tumour"])
        total = len(in_fold)
        summary.append(
            {
                "fold": fold,
                "images": total,
                "tumour": tumour,
                "background": total - tumour,
                "tumour_fraction": tumour / total if total else 0.0,
            }
        )
    return summary


def print_report(records: list[dict], n_folds: int) -> None:
    total = len(records)
    tumour = sum(1 for r in records if r["has_tumour"])
    sources = len({r["source_id"] for r in records})

    print(f"pooled {total} images from {sources} source images")
    print(
        f"tumour {tumour}, background {total - tumour} "
        f"({tumour / total:.1%} tumour overall)"
    )
    print()
    print(
        f"{'fold':>4}  {'images':>7}  {'tumour':>7}  "
        f"{'background':>11}  {'tumour %':>9}"
    )
    for row in fold_summary(records, n_folds):
        print(
            f"{row['fold']:>4}  {row['images']:>7}  {row['tumour']:>7}  "
            f"{row['background']:>11}  {row['tumour_fraction']:>8.1%}"
        )

    print()
    print("cross-validation roles per iteration (train uses 3 folds, val 1, test 1):")
    for fold in range(n_folds):
        roles = cv_roles(fold, n_folds)
        role_str = "  ".join(f"fold{i}={roles[i]}" for i in range(n_folds))
        print(f"  iteration {fold}: {role_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stratified five-fold cross-validation manifest."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="dataset root containing the split folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/folds/cv_manifest.csv"),
        help="where to write the manifest CSV",
    )
    parser.add_argument("--folds", type=int, default=N_FOLDS, help="number of folds")
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for the stratified shuffle"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {args.data_dir}")

    records = collect_images(args.data_dir)
    assign_folds(records, args.folds, args.seed)

    print_report(records, args.folds)

    write_manifest(records, args.output, args.seed, args.folds)
    print(f"\nmanifest written to {args.output}")

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_folds": args.folds,
                "total_images": len(records),
                "folds": fold_summary(records, args.folds),
            },
            indent=2,
        )
    )
    print(f"summary written to {summary_path}")

    print(
        "\nnext: for each fold, convert that fold's labels to masks and "
        "calibrate OSGDF on its training images only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
