"""Smoke tests for video processing pipeline - quick validation of critical paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from watermark_remover.core.pipeline import process_video


class VideoSmokeTests(unittest.TestCase):
    """Smoke tests for video processing - fast, end-to-end validation."""

    @classmethod
    def setUpClass(cls):
        """Create a minimal test video (runs once for all tests)."""
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Create minimal 4-frame test video (160x120 for speed)
        cls.video_path = cls.temp_path / "smoke_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(cls.video_path), fourcc, 10.0, (160, 120))

        for i in range(4):
            # Simple gradient with watermark
            frame = np.ones((120, 160, 3), dtype=np.uint8) * (60 + i * 10)
            # Bottom-right watermark
            frame[90:, 120:] = 255
            writer.write(frame)

        writer.release()

        # Create simple mask
        cls.mask_path = cls.temp_path / "mask.png"
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[90:, 120:] = 255
        cv2.imwrite(str(cls.mask_path), mask)

    @classmethod
    def tearDownClass(cls):
        """Clean up test video."""
        cls.temp_dir.cleanup()

    def test_smoke_video_telea_no_temporal(self):
        """Smoke test: Telea inpainting without temporal guidance."""
        output_path = self.temp_path / "smoke_telea.mp4"

        process_video(
            input_path=self.video_path,
            output_path=output_path,
            mask_mode="manual",
            mask_path=self.mask_path,
            dilate=2,
            method="telea",
            window=4,
            overlap=1,
            temporal_guidance=False,
            qc_expr=None,
            retry_dilate=0,
            seed=42,
        )

        # Verify output exists and has frames
        self.assertTrue(output_path.exists())
        cap = cv2.VideoCapture(str(output_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(frame_count, 4)

    def test_smoke_video_telea_with_temporal(self):
        """Smoke test: Telea inpainting with temporal guidance."""
        output_path = self.temp_path / "smoke_temporal.mp4"

        process_video(
            input_path=self.video_path,
            output_path=output_path,
            mask_mode="manual",
            mask_path=self.mask_path,
            dilate=2,
            method="telea",
            window=4,
            overlap=1,
            temporal_guidance=True,  # Enable flow-based blending
            qc_expr=None,
            retry_dilate=0,
            seed=42,
        )

        # Verify output
        self.assertTrue(output_path.exists())
        cap = cv2.VideoCapture(str(output_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(frame_count, 4)

    def test_smoke_video_auto_mask(self):
        """Smoke test: Auto-masking mode."""
        output_path = self.temp_path / "smoke_auto.mp4"

        process_video(
            input_path=self.video_path,
            output_path=output_path,
            mask_mode="auto",  # Automatic watermark detection
            mask_path=None,
            dilate=3,
            method="telea",
            window=4,
            overlap=1,
            temporal_guidance=False,
            qc_expr=None,
            retry_dilate=0,
            seed=42,
        )

        # Verify output
        self.assertTrue(output_path.exists())

    def test_smoke_video_noop(self):
        """Smoke test: No-op mode (copy frames)."""
        output_path = self.temp_path / "smoke_noop.mp4"

        process_video(
            input_path=self.video_path,
            output_path=output_path,
            mask_mode="auto",
            mask_path=None,
            dilate=0,
            method="noop",  # Should just copy frames
            window=4,
            overlap=0,
            temporal_guidance=False,
            qc_expr=None,
            retry_dilate=0,
            seed=42,
        )

        # Verify output
        self.assertTrue(output_path.exists())

    def test_smoke_video_chunked_processing(self):
        """Smoke test: Chunked processing with overlap."""
        output_path = self.temp_path / "smoke_chunked.mp4"

        process_video(
            input_path=self.video_path,
            output_path=output_path,
            mask_mode="manual",
            mask_path=self.mask_path,
            dilate=2,
            method="telea",
            window=2,  # Process in 2-frame chunks
            overlap=1,  # 1-frame overlap for blending
            temporal_guidance=True,
            qc_expr=None,
            retry_dilate=0,
            seed=42,
        )

        # Verify output
        self.assertTrue(output_path.exists())
        cap = cv2.VideoCapture(str(output_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.assertEqual(frame_count, 4)


class VideoStressTests(unittest.TestCase):
    """Stress tests for edge cases and error handling."""

    def setUp(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_single_frame_video(self):
        """Test processing a single-frame video."""
        video_path = self.temp_path / "single_frame.mp4"
        output_path = self.temp_path / "output.mp4"

        # Create 1-frame video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))
        frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        writer.write(frame)
        writer.release()

        # Process
        process_video(
            input_path=video_path,
            output_path=output_path,
            mask_mode="auto",
            mask_path=None,
            dilate=3,
            method="telea",
            window=10,  # Window larger than video
            overlap=2,
            temporal_guidance=False,
            qc_expr=None,
            retry_dilate=0,
            seed=42,
        )

        # Verify output
        self.assertTrue(output_path.exists())

    def test_missing_video_file(self):
        """Test error handling for missing input video."""
        missing_path = self.temp_path / "nonexistent.mp4"
        output_path = self.temp_path / "output.mp4"

        with self.assertRaises(FileNotFoundError):
            process_video(
                input_path=missing_path,
                output_path=output_path,
                mask_mode="auto",
                mask_path=None,
                dilate=3,
                method="telea",
                window=10,
                overlap=2,
                temporal_guidance=False,
                qc_expr=None,
                retry_dilate=0,
                seed=42,
            )

    def test_invalid_window_size(self):
        """Test handling of invalid window/overlap parameters."""
        # Create minimal video
        video_path = self.temp_path / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))
        for _ in range(2):
            frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        output_path = self.temp_path / "output.mp4"

        # Window <= overlap should be handled gracefully
        # (implementation may clamp or raise error)
        try:
            process_video(
                input_path=video_path,
                output_path=output_path,
                mask_mode="auto",
                mask_path=None,
                dilate=3,
                method="telea",
                window=5,
                overlap=10,  # Overlap > window
                temporal_guidance=False,
                qc_expr=None,
                retry_dilate=0,
                seed=42,
            )
        except (ValueError, AssertionError):
            # Expected error for invalid params
            pass


if __name__ == "__main__":
    # Run smoke tests with verbose output
    unittest.main(verbosity=2)
