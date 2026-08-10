from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_CANDIDATES = ("train", "val", "valid", "validation", "test")

# A coordinate above this must be in pixels; between 1.0 and this it is
# treated as annotation overflow and clipped.
PIXEL_COORD_THRESHOLD = 1.5

# Roboflow renames every exported file to "<original_stem>.rf.<md5>". When it
# generates augmented copies, each copy keeps the original stem and gets a new
# hash, so the stem identifies the underlying photograph.
ROBOFLOW_HASH = re.compile(r"\.rf\.[0-9a-fA-F]{6,}$")


def source_image_id(stem: str) -> str:
    """Strip Roboflow's .rf.<hash> suffix to recover the source image identity.

    'IMG_7519_JPG.rf.2de4c67d...' and 'IMG_7519_JPG.rf.878f1f9f...' are two
    augmented copies of one X-ray and both return 'IMG_7519_JPG'.
    """
    return ROBOFLOW_HASH.sub("", stem)


# label parsing


@dataclass
class Annotation:
    """One parsed annotation, in normalised [0, 1] coordinates."""

    class_id: int
    points: list[tuple[float, float]]
    kind: str  # 'polygon' | 'obb' | 'box'


@dataclass
class ParseResult:
    annotations: list[Annotation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    was_empty: bool = False  # genuine background: file present, no lines
    converted_from_pixels: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors


def load_class_names(data_dir: Path) -> list[str] | None:
    """Read the `names:` list from a Roboflow/Ultralytics data.yaml, if present.

    Parsed with a small hand-rolled reader so PyYAML is not a dependency.
    Handles both the inline form (names: ['a', 'b']) and the block form.
    """
    for candidate in ("data.yaml", "data.yml", "dataset.yaml"):
        path = data_dir / candidate
        if not path.exists():
            continue
        try:
            text = path.read_text()
        except OSError:
            continue

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("names:"):
                continue
            value = stripped[len("names:") :].strip()
            # Inline list form:  names: ['a', 'b']
            if value.startswith("[") and value.endswith("]"):
                items = [v.strip().strip("'\"") for v in value[1:-1].split(",")]
                return [v for v in items if v]
            # Inline dict form:  names: {0: a, 1: b}
            if value.startswith("{") and value.endswith("}"):
                pairs = {}
                for chunk in value[1:-1].split(","):
                    if ":" not in chunk:
                        continue
                    k, v = chunk.split(":", 1)
                    pairs[k.strip().strip("'\"")] = v.strip().strip("'\"")
                return _names_from_pairs(pairs)

        # Block forms under a bare `names:` line. Two variants coexist in the
        # wild: Ultralytics/Roboflow dict form
        #     names:
        #       0: cancer
        #       1: benign
        # and the older list form
        #     names:
        #       - cancer
        #       - benign
        block_pairs: dict[str, str] = {}
        block_list: list[str] = []
        collecting = False
        for line in text.splitlines():
            if line.strip().startswith("names:"):
                collecting = True
                continue
            if collecting:
                stripped = line.strip()
                if not stripped:
                    continue
                indented = line.startswith((" ", "\t"))
                if stripped.startswith("- "):
                    block_list.append(stripped[2:].strip().strip("'\""))
                elif indented and ":" in stripped:
                    k, v = stripped.split(":", 1)
                    block_pairs[k.strip().strip("'\"")] = v.strip().strip("'\"")
                elif not indented:
                    break  # dedented to a new top-level key; names block ended
        if block_pairs:
            return _names_from_pairs(block_pairs)
        if block_list:
            return block_list
    return None


def _names_from_pairs(pairs: dict[str, str]) -> list[str]:
    """Turn a {index: name} mapping into an ordered list, indexed by class id.

    Roboflow writes `names: {0: cancer}`. The list this returns is positional,
    so class_map later inverts it back to name->id correctly even if ids are
    sparse or out of order.
    """
    numbered = {}
    for key, name in pairs.items():
        try:
            numbered[int(key)] = name
        except ValueError:
            continue
    if not numbered:
        return [v for v in pairs.values() if v]
    size = max(numbered) + 1
    return [numbered.get(i, str(i)) for i in range(size)]


def _normalise_coords(
    coords: Sequence[float], height: int, width: int
) -> tuple[list[float], bool]:
    """Convert a flat coord list to normalised [0, 1], detecting pixel input."""
    from_pixels = max(coords) > PIXEL_COORD_THRESHOLD
    if from_pixels:
        coords = [c / width if i % 2 == 0 else c / height for i, c in enumerate(coords)]
    return [min(1.0, max(0.0, c)) for c in coords], from_pixels


def parse_label_file(
    path: Path, height: int, width: int, class_map: dict[str, int] | None = None
) -> ParseResult:
    """Parse one YOLO-style TXT label file.

    Accepts, per line:
      * `class cx cy w h`                    - axis-aligned detection box
      * `class x1 y1 x2 y2 x3 y3 ...`        - polygon or oriented box
    where `class` is an integer id or a name present in data.yaml.

    Unlike the original, a line that cannot be parsed produces an entry in
    `errors` rather than being skipped silently.
    """
    result = ParseResult()

    try:
        raw_lines = path.read_text().splitlines()
    except OSError as exc:
        result.errors.append(f"unreadable: {exc}")
        return result

    lines = [ln.strip() for ln in raw_lines if ln.strip()]
    if not lines:
        result.was_empty = True
        return result

    for line_no, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 5:
            result.errors.append(
                f"line {line_no}: only {len(parts)} fields, need at least 5"
            )
            continue

        # -- class token: integer id, or a name resolved via data.yaml --
        token = parts[0]
        try:
            class_id = int(token)
        except ValueError:
            if class_map and token in class_map:
                class_id = class_map[token]
            else:
                known = sorted(class_map) if class_map else []
                result.errors.append(
                    f"line {line_no}: class '{token}' is not an integer and "
                    f"is not in data.yaml names {known}"
                )
                continue

        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError as exc:
            result.errors.append(f"line {line_no}: non-numeric coordinate ({exc})")
            continue

        if len(coords) == 4:
            # Axis-aligned YOLO detection box -> expand to 4 corners so the
            # rest of the pipeline has a single representation.
            coords, from_pixels = _normalise_coords(coords, height, width)
            cx, cy, bw, bh = coords
            x0, x1 = cx - bw / 2, cx + bw / 2
            y0, y1 = cy - bh / 2, cy + bh / 2
            points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            result.annotations.append(Annotation(class_id, points, "box"))
            result.converted_from_pixels |= from_pixels
            continue

        if len(coords) % 2 != 0:
            result.errors.append(
                f"line {line_no}: {len(coords)} coordinates is odd; "
                "polygon coordinates must come in x,y pairs"
            )
            continue

        if len(coords) < 6:
            result.errors.append(
                f"line {line_no}: {len(coords) // 2} points is too few for a "
                "polygon (need 3+)"
            )
            continue

        coords, from_pixels = _normalise_coords(coords, height, width)
        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        kind = "obb" if len(points) == 4 else "polygon"
        result.annotations.append(Annotation(class_id, points, kind))
        result.converted_from_pixels |= from_pixels

    return result


# geometry: letterbox transform applied to points, not to masks
@dataclass
class Letterbox:
    """Scale-and-pad transform mapping an original image into a square canvas."""

    original_h: int
    original_w: int
    target: int
    scale: float
    new_h: int
    new_w: int
    pad_top: int
    pad_left: int

    @classmethod
    def build(cls, height: int, width: int, target: int) -> Letterbox:
        if target <= 0:  # resizing disabled: identity transform
            return cls(height, width, 0, 1.0, height, width, 0, 0)
        scale = target / max(height, width)
        new_h, new_w = round(height * scale), round(width * scale)
        new_h, new_w = min(new_h, target), min(new_w, target)
        return cls(
            height,
            width,
            target,
            scale,
            new_h,
            new_w,
            (target - new_h) // 2,
            (target - new_w) // 2,
        )

    @property
    def output_size(self) -> tuple[int, int]:
        if self.target <= 0:
            return self.original_h, self.original_w
        return self.target, self.target

    def apply_image(self, image: np.ndarray, pad_value: int = 0) -> np.ndarray:
        if self.target <= 0:
            return image
        # INTER_AREA is the correct anti-aliased choice for downscaling;
        # Lanczos aliases and rings when shrinking.
        interp = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(image, (self.new_w, self.new_h), interpolation=interp)
        return cv2.copyMakeBorder(
            resized,
            self.pad_top,
            self.target - self.new_h - self.pad_top,
            self.pad_left,
            self.target - self.new_w - self.pad_left,
            cv2.BORDER_CONSTANT,
            value=pad_value,
        )

    def apply_points(
        self, points: Sequence[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Map normalised points in the original frame to pixels in the output.

        Doing the transform on coordinates means the mask is filled once, at
        the final resolution, and never resampled.
        """
        out = []
        for nx, ny in points:
            px = nx * self.original_w
            py = ny * self.original_h
            if self.target > 0:
                px = px * self.scale + self.pad_left
                py = py * self.scale + self.pad_top
            out.append((px, py))
        return out


def draw_mask(annotations: Sequence[Annotation], transform: Letterbox) -> np.ndarray:
    """Fill all annotation polygons into a single binary mask (0 / 255)."""
    out_h, out_w = transform.output_size
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    for ann in annotations:
        pts = transform.apply_points(ann.points)
        poly = np.array([[round(x), round(y)] for x, y in pts], dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
    return mask


def to_yolo_bbox(
    annotation: Annotation, transform: Letterbox
) -> tuple[int, float, float, float, float] | None:
    """Axis-aligned YOLO box (normalised) for one annotation in output space."""
    out_h, out_w = transform.output_size
    pts = transform.apply_points(annotation.points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x0, x1 = max(0.0, min(xs)), min(float(out_w), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(out_h), max(ys))
    if x1 <= x0 or y1 <= y0:
        return None  # annotation fell entirely outside the canvas

    return (
        annotation.class_id,
        ((x0 + x1) / 2) / out_w,
        ((y0 + y1) / 2) / out_h,
        (x1 - x0) / out_w,
        (y1 - y0) / out_h,
    )


# dataset discovery


def find_splits(data_dir: Path) -> list[str]:
    """Detect split subdirectories that contain both images/ and labels/."""
    splits = []
    for name in SPLIT_CANDIDATES:
        d = data_dir / name
        if d.is_dir() and (d / "images").is_dir() and (d / "labels").is_dir():
            splits.append(name)
    return splits


def list_images(split_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in (split_dir / "images").iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# inspection (writes nothing)


def audit_source_overlap(data_dir: Path, splits: Sequence[str]) -> dict:
    """Detect augmented duplicates and cross-split leakage.

    Roboflow can be configured to augment *before* splitting. When that
    happens, augmented copies of the same photograph land in both train and
    val, so the validation set is scoring images the model has effectively
    already seen. Validation metrics then measure memorisation and look
    excellent right up until the model meets a genuinely new patient.

    This is invisible to any per-file check, because every filename is
    distinct. Only the stem before '.rf.' reveals it.
    """
    per_split: dict[str, Counter] = {}
    for split in splits:
        counter: Counter = Counter()
        for image_path in list_images(data_dir / split):
            counter[source_image_id(image_path.stem)] += 1
        per_split[split] = counter

    overlaps = {}
    split_list = list(splits)
    for i, a in enumerate(split_list):
        for b in split_list[i + 1 :]:
            shared = sorted(set(per_split[a]) & set(per_split[b]))
            if shared:
                overlaps[f"{a}|{b}"] = shared

    return {
        "per_split": {
            s: {
                "files": sum(c.values()),
                "unique_sources": len(c),
                "multiplier": (sum(c.values()) / len(c)) if c else 0.0,
                "max_copies": max(c.values()) if c else 0,
            }
            for s, c in per_split.items()
        },
        "overlaps": overlaps,
    }


def print_source_audit(audit: dict) -> bool:
    """Print the audit. Returns True if leakage was found."""
    print("\n" + "=" * 74)
    print("SOURCE IMAGE AUDIT (augmented duplicates and cross-split leakage)")
    print("=" * 74)

    for split, info in audit["per_split"].items():
        print(
            f"  {split:6s}: {info['files']:5d} files from "
            f"{info['unique_sources']:5d} unique source images "
            f"(x{info['multiplier']:.2f} average, max x{info['max_copies']})"
        )

    augmented = any(i["multiplier"] > 1.05 for i in audit["per_split"].values())
    if augmented:
        print("\n  Augmented copies detected. Two consequences:")
        print("   - Your effective dataset size is the unique-source count,")
        print("     not the file count. Quote both in the write-up.")
        print("   - Any cross-validation must split on source id, never on")
        print("     filename, or folds will share the same X-rays.")

    if not audit["overlaps"]:
        print("\n  No source image appears in more than one split. Clean.")
        return False

    print("\n  >> LEAKAGE: the same source images appear in multiple splits.")
    for pair, shared in audit["overlaps"].items():
        a, b = pair.split("|")
        print(f"     {a} <-> {b}: {len(shared)} shared source images")
        print(f"       e.g. {', '.join(shared[:5])}")
    print("\n     Validation scores computed on this split measure")
    print("     memorisation, not generalisation. Re-export from Roboflow")
    print("     with the split applied BEFORE augmentation, or re-split")
    print("     yourself grouping by source id.")
    return True


def inspect(data_dir: Path, class_names: list[str] | None, sample: int = 0) -> dict:
    """Report what the labels actually contain, without converting anything."""
    class_map = {n: i for i, n in enumerate(class_names)} if class_names else None
    splits = find_splits(data_dir)
    if not splits:
        raise SystemExit(
            f"ERROR: no split with both images/ and labels/ under {data_dir}\n"
            f"       looked for: {', '.join(SPLIT_CANDIDATES)}"
        )

    print("=" * 74)
    print("DATASET INSPECTION (no files written)")
    print("=" * 74)
    print(f"\nRoot        : {data_dir}")
    print(f"Splits      : {', '.join(splits)}")
    print(f"Class names : {class_names if class_names else 'no data.yaml found'}")

    report: dict = {"root": str(data_dir), "splits": {}, "class_names": class_names}

    for split in splits:
        split_dir = data_dir / split
        images = list_images(split_dir)
        if sample:
            images = images[:sample]

        point_counts: Counter = Counter()
        kinds: Counter = Counter()
        classes: Counter = Counter()
        sizes: Counter = Counter()
        missing_label, empty_label, failed = [], [], []
        pixel_coord_files = []
        n_annotations = 0

        for image_path in images:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed.append((image_path.name, "image unreadable"))
                continue
            h, w = img.shape[:2]
            sizes[f"{w}x{h}"] += 1

            label_path = split_dir / "labels" / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_label.append(image_path.name)
                continue

            parsed = parse_label_file(label_path, h, w, class_map)
            if parsed.errors:
                failed.append((image_path.name, parsed.errors[0]))
                continue
            if parsed.was_empty:
                empty_label.append(image_path.name)
                continue
            if parsed.converted_from_pixels:
                pixel_coord_files.append(image_path.name)

            for ann in parsed.annotations:
                point_counts[len(ann.points)] += 1
                kinds[ann.kind] += 1
                classes[ann.class_id] += 1
                n_annotations += 1

        print(f"\n{'-' * 74}\n{split.upper()}  -  {len(images)} images\n{'-' * 74}")
        print(f"  annotations parsed   : {n_annotations}")
        print(f"  empty label files    : {len(empty_label)}  (genuine background)")
        print(f"  missing label files  : {len(missing_label)}")
        print(f"  FAILED to parse      : {len(failed)}")
        if failed:
            for name, reason in failed[:5]:
                print(f"      {name}: {reason}")
            if len(failed) > 5:
                print(f"      ... and {len(failed) - 5} more")
        if missing_label:
            print(f"      e.g. {', '.join(missing_label[:5])}")
        if pixel_coord_files:
            print(
                f"  pixel-coordinate files: {len(pixel_coord_files)} "
                f"(will be normalised)"
            )

        if sizes:
            common = ", ".join(f"{k} x{v}" for k, v in sizes.most_common(4))
            print(
                f"  image sizes          : {common}"
                f"{' (mixed)' if len(sizes) > 1 else ''}"
            )

        if classes:
            label_of = lambda c: (
                class_names[c] if class_names and 0 <= c < len(class_names) else str(c)
            )
            print(
                "  class distribution   : "
                + ", ".join(f"{label_of(c)}={n}" for c, n in sorted(classes.items()))
            )

        if point_counts:
            print(
                "  points per annotation: "
                + ", ".join(f"{p}pts x{n}" for p, n in sorted(point_counts.items()))
            )

        # The claim that actually matters for the segmentation branch.
        box_like = kinds["obb"] + kinds["box"]
        if n_annotations:
            share = box_like / n_annotations
            print()
            if share >= 0.5:
                print(
                    f"  >> WARNING: {box_like}/{n_annotations} "
                    f"({share:.0%}) of annotations are 4-point boxes, not"
                )
                print("     traced outlines. Masks built from these are filled")
                print("     rectangles. A Dice score against them measures")
                print("     agreement with rectangles, not tumour delineation,")
                print("     and is not comparable to the ~97% Dice quoted for")
                print("     outline-annotated datasets.")
            elif box_like:
                print(
                    f"  >> NOTE: {box_like}/{n_annotations} ({share:.0%}) of "
                    f"annotations are 4-point boxes; the rest are outlines."
                )
                print("     Consider excluding the box ones from segmentation")
                print("     training, or report them separately.")
            else:
                print(
                    f"  >> Good: all {n_annotations} annotations are traced "
                    f"outlines (>4 points)."
                )

        report["splits"][split] = {
            "n_images": len(images),
            "n_annotations": n_annotations,
            "empty_labels": len(empty_label),
            "missing_labels": missing_label,
            "failed": [{"file": f, "reason": r} for f, r in failed],
            "kinds": dict(kinds),
            "points_per_annotation": {str(k): v for k, v in point_counts.items()},
            "class_distribution": {str(k): v for k, v in classes.items()},
            "image_sizes": dict(sizes),
            "pixel_coordinate_files": len(pixel_coord_files),
        }

    audit = audit_source_overlap(data_dir, splits)
    leaked = print_source_audit(audit)
    report["source_audit"] = audit

    total_failed = sum(len(s["failed"]) for s in report["splits"].values())
    total_missing = sum(len(s["missing_labels"]) for s in report["splits"].values())
    print("\n" + "=" * 74)
    if total_failed or total_missing:
        print(
            f"Resolve {total_failed} parse failures and {total_missing} missing "
            f"labels before converting."
        )
    elif leaked:
        print("Labels parse cleanly, but fix the split leakage before you")
        print("trust any validation number this dataset produces.")
    else:
        print("No parse failures, missing labels, or split leakage. Safe to convert.")
    print("=" * 74)
    return report


# conversion


def convert_split(
    split: str,
    data_dir: Path,
    output_dir: Path,
    target_size: int,
    class_map: dict[str, int] | None,
    mask_suffix: str,
    image_format: str,
    strict: bool,
    limit: int = 0,
) -> dict:
    """Convert one split; returns a stats dict."""
    split_dir = data_dir / split
    out_images = output_dir / split / "images"
    out_masks = output_dir / split / "masks"
    out_bboxes = output_dir / split / "bboxes"
    for d in (out_images, out_masks, out_bboxes):
        d.mkdir(parents=True, exist_ok=True)

    images = list_images(split_dir)
    if limit:
        images = images[:limit]

    stats = {
        "n_images": len(images),
        "written": 0,
        "background": 0,
        "with_tumour": 0,
        "failed": [],
        "missing_labels": [],
        "dropped_annotations": 0,
        "n_annotations": 0,
    }

    for i, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            stats["failed"].append(
                {"file": image_path.name, "reason": "image unreadable"}
            )
            continue

        h, w = image.shape[:2]
        label_path = split_dir / "labels" / f"{image_path.stem}.txt"

        if label_path.exists():
            parsed = parse_label_file(label_path, h, w, class_map)
        else:
            parsed = ParseResult(was_empty=True)
            stats["missing_labels"].append(image_path.name)

        if parsed.errors:
            stats["failed"].append(
                {"file": image_path.name, "reason": "; ".join(parsed.errors[:3])}
            )
            if strict:
                raise SystemExit(
                    f"\nERROR: {label_path} failed to parse:\n  "
                    + "\n  ".join(parsed.errors)
                    + "\n\nRe-run with --no-strict to skip bad files, or "
                    "--inspect to see every problem at once."
                )
            continue  # never emit a blank mask for a file we could not read

        transform = Letterbox.build(h, w, target_size)

        # image
        out_image = transform.apply_image(image, pad_value=0)
        cv2.imwrite(str(out_images / f"{image_path.stem}.{image_format}"), out_image)

        # mask: filled once at final resolution, never resampled
        mask = draw_mask(parsed.annotations, transform)
        cv2.imwrite(str(out_masks / f"{image_path.stem}{mask_suffix}.png"), mask)

        # bboxes: always written, empty file for background
        lines = []
        for ann in parsed.annotations:
            box = to_yolo_bbox(ann, transform)
            if box is None:
                stats["dropped_annotations"] += 1
                continue
            cid, cx, cy, bw, bh = box
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_bboxes / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

        stats["n_annotations"] += len(parsed.annotations)
        stats["with_tumour" if parsed.annotations else "background"] += 1
        stats["written"] += 1

        if i % 100 == 0 or i == len(images):
            sys.stdout.write(f"\r  {split}: {i}/{len(images)}   ")
            sys.stdout.flush()

    if images:
        sys.stdout.write("\n")
    return stats


def verify_split(split: str, output_dir: Path, mask_suffix: str) -> dict:
    """Check the converted split is internally consistent."""
    out_images = output_dir / split / "images"
    out_masks = output_dir / split / "masks"
    out_bboxes = output_dir / split / "bboxes"

    problems: list[str] = []
    n_checked = 0
    mask_pixels = []

    for image_path in sorted(out_images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = image_path.stem
        n_checked += 1

        mask_path = out_masks / f"{stem}{mask_suffix}.png"
        bbox_path = out_bboxes / f"{stem}.txt"

        if not mask_path.exists():
            problems.append(f"{stem}: mask missing")
            continue
        if not bbox_path.exists():
            problems.append(f"{stem}: bbox file missing")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            problems.append(f"{stem}: unreadable output")
            continue
        if image.shape != mask.shape:
            problems.append(f"{stem}: image {image.shape} != mask {mask.shape}")

        unique = np.unique(mask)
        if not set(unique.tolist()).issubset({0, 255}):
            problems.append(f"{stem}: mask not binary ({len(unique)} values)")
        mask_pixels.append(int((mask > 0).sum()))

        n_boxes = len([ln for ln in bbox_path.read_text().splitlines() if ln.strip()])
        has_mask = mask.any()
        if has_mask and n_boxes == 0:
            problems.append(f"{stem}: mask has content but no bbox")
        if n_boxes and not has_mask:
            problems.append(f"{stem}: bbox present but mask empty")

        for line_no, line in enumerate(bbox_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            vals = line.split()
            if len(vals) != 5:
                problems.append(f"{stem}: bbox line {line_no} has {len(vals)} fields")
                continue
            if not all(0.0 <= float(v) <= 1.0 for v in vals[1:]):
                problems.append(f"{stem}: bbox line {line_no} outside [0,1]")

    tumour_pixels = [p for p in mask_pixels if p > 0]
    return {
        "checked": n_checked,
        "problems": problems,
        "masks_with_content": len(tumour_pixels),
        "mean_tumour_pixels": float(np.mean(tumour_pixels)) if tumour_pixels else 0.0,
    }


# cli


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert TXT polygon/OBB labels to segmentation masks and "
        "YOLO bboxes, with letterbox resizing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="dataset root containing train/ and val/",
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/prepared"))
    p.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="letterbox to NxN; 0 keeps original size",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="report label contents and exit without writing",
    )
    p.add_argument(
        "--mask-suffix",
        default="",
        help="appended to mask filenames; '' keeps stems identical "
        "to images, which makes Phase 3 pairing trivial. Use "
        "'_mask' to match the earlier script's naming.",
    )
    p.add_argument(
        "--image-format",
        choices=["png", "jpg"],
        default="png",
        help="png avoids adding JPEG artefacts before denoising",
    )
    p.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="skip unparseable label files instead of stopping",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="process only the first N images per split"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"ERROR: --data-dir does not exist: {args.data_dir}")

    class_names = load_class_names(args.data_dir)
    class_map = {n: i for i, n in enumerate(class_names)} if class_names else None

    if args.inspect:
        report = inspect(args.data_dir, class_names, sample=args.limit)
        out = args.data_dir / "inspection_report.json"
        try:
            out.write_text(json.dumps(report, indent=2))
            print(f"\nFull report -> {out}")
        except OSError as exc:
            print(f"(could not write report: {exc})")
        return 0

    splits = find_splits(args.data_dir)
    if not splits:
        raise SystemExit(
            f"ERROR: no split with both images/ and labels/ under {args.data_dir}"
        )

    print("=" * 74)
    print("PHASE 1 - LABELS -> MASKS + BBOXES")
    print("=" * 74)
    print(f"  input      : {args.data_dir}")
    print(f"  output     : {args.output_dir}")
    print(f"  splits     : {', '.join(splits)}")
    print(
        f"  target size: "
        f"{f'{args.target_size}x{args.target_size} letterboxed' if args.target_size else 'unchanged'}"
    )
    print(f"  classes    : {class_names if class_names else 'none (integer ids only)'}")
    print()

    all_stats, all_verify = {}, {}
    for split in splits:
        all_stats[split] = convert_split(
            split,
            args.data_dir,
            args.output_dir,
            args.target_size,
            class_map,
            args.mask_suffix,
            args.image_format,
            args.strict,
            args.limit,
        )
        all_verify[split] = verify_split(split, args.output_dir, args.mask_suffix)

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    total_failed = 0
    for split in splits:
        s, v = all_stats[split], all_verify[split]
        total_failed += len(s["failed"])
        print(f"\n{split.upper()}")
        print(f"  written            : {s['written']}/{s['n_images']}")
        print(f"  with tumour        : {s['with_tumour']}")
        print(f"  background (empty) : {s['background']}")
        print(f"  annotations        : {s['n_annotations']}")
        if s["missing_labels"]:
            print(
                f"  missing label files: {len(s['missing_labels'])} "
                f"(e.g. {', '.join(s['missing_labels'][:3])})"
            )
        if s["dropped_annotations"]:
            print(f"  dropped (off-canvas): {s['dropped_annotations']}")
        if s["failed"]:
            print(f"  FAILED             : {len(s['failed'])}")
            for f in s["failed"][:5]:
                print(f"      {f['file']}: {f['reason']}")
        print(
            f"  verification       : {len(v['problems'])} problems in "
            f"{v['checked']} checked"
        )
        for problem in v["problems"][:5]:
            print(f"      {problem}")
        if v["masks_with_content"]:
            print(f"  mean tumour area   : {v['mean_tumour_pixels']:.0f} px")

    audit = audit_source_overlap(args.data_dir, splits)
    print_source_audit(audit)

    report_path = args.output_dir / "preparation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "config": {
                    "data_dir": str(args.data_dir),
                    "target_size": args.target_size,
                    "class_names": class_names,
                    "mask_suffix": args.mask_suffix,
                },
                "conversion": all_stats,
                "verification": all_verify,
                "source_audit": audit,
            },
            indent=2,
        )
    )
    print(f"\nReport -> {report_path}")

    print("\n" + "=" * 74)
    if total_failed:
        print(f"{total_failed} files failed. Run --inspect to see every problem.")
    else:
        print("NEXT: OSGDF. Calibrate on TRAIN ONLY, then reuse those parameters")
        print("on val - tuning filter parameters on validation data leaks")
        print("information from your held-out set into preprocessing.\n")
        train = "train" if "train" in splits else splits[0]
        others = [s for s in splits if s != train]
        print("  python run_osgdf.py \\")
        print(f"      --input-dir  {args.output_dir}/{train}/images \\")
        print(f"      --output-dir {args.output_dir}/{train}/osgdf_images \\")
        print(f"      --report-dir outputs/results/{train}")
        for split in others:
            print("\n  # then reuse the window/order it printed:")
            print(
                "  python run_osgdf.py --skip-optimization --window <W> --poly <P> \\"
            )
            print(f"      --input-dir  {args.output_dir}/{split}/images \\")
            print(f"      --output-dir {args.output_dir}/{split}/osgdf_images \\")
            print(f"      --report-dir outputs/results/{split}")
    print("=" * 74)
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
