from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data.build_fold_ultralytics import (
    denoise_to_disk,
    place_label,
    transform_label_text,
)
from src.data.prepare_dataset import Letterbox


def parse_label(text: str) -> list[tuple[str, np.ndarray]]:
    parsed = []
    for line in text.splitlines():
        parts = line.split()
        parsed.append((parts[0], np.asarray(parts[1:], dtype=float).reshape(-1, 2)))
    return parsed


class FoldUltralyticsLetterboxTests(unittest.TestCase):
    def _write_image_and_transform(
        self, directory: Path, height: int, width: int
    ) -> tuple[Path, Letterbox]:
        source = directory / f"source_{width}x{height}.png"
        output = directory / f"output_{width}x{height}.png"
        image = np.full((height, width), 127, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(source), image))

        transform = denoise_to_disk(source, output, window=5, poly=2, target=512)
        written = cv2.imread(str(output), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(written)
        self.assertEqual(written.shape, (512, 512))
        return output, transform

    def test_landscape_image_and_polygon_share_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, transform = self._write_image_and_transform(Path(tmp), 100, 200)

        self.assertAlmostEqual(transform.scale, 2.56)
        self.assertEqual((transform.new_h, transform.new_w), (256, 512))
        self.assertEqual((transform.pad_top, transform.pad_left), (128, 0))

        result = parse_label(
            transform_label_text("0 0.1 0.2 0.4 0.2 0.4 0.6 0.1 0.6\n", transform)
        )
        expected = np.asarray(((0.1, 0.35), (0.4, 0.35), (0.4, 0.55), (0.1, 0.55)))
        self.assertEqual(result[0][0], "0")
        np.testing.assert_allclose(result[0][1], expected, atol=1e-12)

    def test_portrait_image_and_polygon_share_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, transform = self._write_image_and_transform(Path(tmp), 200, 100)

        self.assertAlmostEqual(transform.scale, 2.56)
        self.assertEqual((transform.new_h, transform.new_w), (512, 256))
        self.assertEqual((transform.pad_top, transform.pad_left), (0, 128))

        result = parse_label(
            transform_label_text("3 0.2 0.1 0.8 0.1 0.8 0.9 0.2 0.9", transform)
        )
        expected = np.asarray(((0.35, 0.1), (0.65, 0.1), (0.65, 0.9), (0.35, 0.9)))
        self.assertEqual(result[0][0], "3")
        np.testing.assert_allclose(result[0][1], expected, atol=1e-12)

    def test_square_image_preserves_normalised_polygon(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, transform = self._write_image_and_transform(Path(tmp), 120, 120)

        self.assertEqual((transform.new_h, transform.new_w), (512, 512))
        self.assertEqual((transform.pad_top, transform.pad_left), (0, 0))
        source = "0 0.05 0.1 0.95 0.1 0.95 0.9 0.05 0.9\n"
        actual = parse_label(transform_label_text(source, transform))[0][1]
        expected = parse_label(source)[0][1]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_multiple_polygons_and_class_ids_are_preserved(self):
        transform = Letterbox.build(height=100, width=200, target=512)
        source = (
            "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
            "5 0.6 0.6 0.9 0.6 0.9 0.8 0.6 0.8\n"
        )
        result = parse_label(transform_label_text(source, transform))
        self.assertEqual([class_id for class_id, _ in result], ["0", "5"])
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(
            result[0][1], ((0.1, 0.3), (0.2, 0.3), (0.2, 0.35), (0.1, 0.35))
        )
        np.testing.assert_allclose(
            result[1][1], ((0.6, 0.55), (0.9, 0.55), (0.9, 0.65), (0.6, 0.65))
        )

    def test_empty_label_remains_empty(self):
        transform = Letterbox.build(height=100, width=200, target=512)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "empty.txt"
            output = root / "output.txt"
            source.write_text("")
            place_label(source, output, transform, use_symlink=False)
            self.assertEqual(output.read_text(), "")

    def test_border_points_are_clamped_to_normalised_bounds(self):
        transform = Letterbox.build(height=200, width=100, target=512)
        source = "0 -0.01 0 1.01 0 1.01 1 -0.01 1\n"
        points = parse_label(transform_label_text(source, transform))[0][1]
        self.assertTrue(np.all(points >= 0.0))
        self.assertTrue(np.all(points <= 1.0))
        np.testing.assert_allclose(
            points, ((0.245, 0.0), (0.755, 0.0), (0.755, 1.0), (0.245, 1.0))
        )


if __name__ == "__main__":
    unittest.main()
