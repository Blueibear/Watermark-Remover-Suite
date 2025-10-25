import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core import utils


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        base = np.full((50, 80, 3), 120, dtype=np.uint8)
        cv2.putText(base, "WM", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        self.image = base

    def test_detect_watermark_mask_threshold(self) -> None:
        mask = utils.detect_watermark_mask(
            self.image, method="threshold", threshold=200, dilate_iterations=1
        )
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, self.image.shape[:2])
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_detect_watermark_mask_adaptive(self) -> None:
        mask = utils.detect_watermark_mask(self.image, method="adaptive", adaptive_block_size=11)
        self.assertEqual(mask.dtype, np.uint8)

    def test_detect_watermark_mask_laplacian(self) -> None:
        mask = utils.detect_watermark_mask(self.image, method="laplacian", threshold=5)
        self.assertEqual(mask.dtype, np.uint8)

    def test_detect_watermark_mask_invalid_method(self) -> None:
        with self.assertRaises(ValueError):
            utils.detect_watermark_mask(self.image, method="unknown")

    def test_prepare_and_resolve_mask(self) -> None:
        raw_mask = utils.detect_watermark_mask(self.image)
        prepared = utils.prepare_mask(raw_mask, self.image.shape[:2])
        self.assertTrue(np.array_equal(prepared, raw_mask))

        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = Path(tmp_dir) / "mask.png"
            cv2.imwrite(str(mask_path), raw_mask)
            resolved_from_path = utils.resolve_mask(mask_path, self.image)
            self.assertTrue(np.array_equal(resolved_from_path, raw_mask))

    def test_load_and_save_image_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image.png"
            utils.save_image(path, self.image)
            loaded = utils.load_image(path)
            self.assertEqual(loaded.shape, self.image.shape)


if __name__ == "__main__":
    unittest.main()
