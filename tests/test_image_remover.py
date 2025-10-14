import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.image_remover import ImageWatermarkRemover
from core import utils
from .helpers import create_synthetic_sample


class TestImageWatermarkRemover(unittest.TestCase):
    def test_inpainting_recovers_gradient_background(self) -> None:
        base, watermarked = create_synthetic_sample()
        mask = utils.detect_watermark_mask(
            watermarked,
            method="threshold",
            threshold=230,
            dilate_iterations=2,
            blur_kernel=5,
        )

        remover = ImageWatermarkRemover(inpaint_radius=3, method="telea")
        result, mask_used = remover.remove_watermark(watermarked, mask=mask)

        self.assertEqual(mask.shape, mask_used.shape)
        restored_region = result[mask_used > 0]
        baseline_region = base[mask_used > 0]
        mean_diff = np.mean(
            np.abs(restored_region.astype(np.int16) - baseline_region.astype(np.int16))
        )
        self.assertLess(
            mean_diff,
            35,
            "Inpainting did not sufficiently restore the masked region.",
        )

    def test_process_file_outputs_expected_artifacts(self) -> None:
        _, watermarked = create_synthetic_sample()
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "watermarked.png"
            output_path = Path(tmp_dir) / "restored.png"
            cv2.imwrite(str(input_path), watermarked)

            remover = ImageWatermarkRemover()
            processed, mask_path = remover.process_file(
                input_path,
                output_path,
                auto_mask_kwargs={"threshold": 230, "dilate_iterations": 2, "blur_kernel": 5},
            )

            self.assertTrue(processed.exists(), "Output image was not written.")
            self.assertTrue(mask_path.exists(), "Mask image was not written.")
            restored = utils.load_image(processed)
            self.assertEqual(restored.shape, watermarked.shape)


if __name__ == "__main__":
    unittest.main()
