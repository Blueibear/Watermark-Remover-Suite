"""Unit tests for watermark_remover MVP package (pipeline, flow, temporal)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from watermark_remover.core.pipeline import process_image, process_video
from watermark_remover.core.flow import FlowEstimator
from watermark_remover.core.temporal import make_chunks, blend_overlap


class TestPipelineImage(unittest.TestCase):
    """Test process_image function with various modes and methods."""

    def setUp(self):
        """Create synthetic test images."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create test image (640x480, RGB)
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.input_path = self.temp_path / "test_input.png"
        cv2.imwrite(str(self.input_path), self.test_image)

        # Create test mask (bottom-right watermark region)
        self.test_mask = np.zeros((480, 640), dtype=np.uint8)
        self.test_mask[400:, 500:] = 255
        self.mask_path = self.temp_path / "test_mask.png"
        cv2.imwrite(str(self.mask_path), self.test_mask)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_process_image_auto_mask_telea(self):
        """Test image processing with automatic masking and Telea inpainting."""
        output_path = self.temp_path / "output_auto.png"

        process_image(
            input_path=self.input_path,
            output_path=output_path,
            mask_mode="auto",
            mask_path=None,
            dilate=5,
            method="telea",
            seed=1234,
        )

        # Verify output exists and has correct dimensions
        self.assertTrue(output_path.exists())
        result = cv2.imread(str(output_path))
        self.assertEqual(result.shape, self.test_image.shape)

    def test_process_image_manual_mask_telea(self):
        """Test image processing with manual mask and Telea inpainting."""
        output_path = self.temp_path / "output_manual.png"

        process_image(
            input_path=self.input_path,
            output_path=output_path,
            mask_mode="manual",
            mask_path=self.mask_path,
            dilate=5,
            method="telea",
            seed=1234,
        )

        # Verify output exists and has correct dimensions
        self.assertTrue(output_path.exists())
        result = cv2.imread(str(output_path))
        self.assertEqual(result.shape, self.test_image.shape)

    def test_process_image_noop(self):
        """Test noop method (copy input to output)."""
        output_path = self.temp_path / "output_noop.png"

        process_image(
            input_path=self.input_path,
            output_path=output_path,
            mask_mode="auto",
            mask_path=None,
            dilate=0,
            method="noop",
            seed=1234,
        )

        # Verify output exists
        self.assertTrue(output_path.exists())
        result = cv2.imread(str(output_path))
        # noop should produce identical output
        np.testing.assert_array_equal(result, self.test_image)

    def test_process_image_missing_input(self):
        """Test error handling for missing input file."""
        missing_path = self.temp_path / "nonexistent.png"
        output_path = self.temp_path / "output.png"

        with self.assertRaises(FileNotFoundError):
            process_image(
                input_path=missing_path,
                output_path=output_path,
                mask_mode="auto",
                mask_path=None,
                dilate=5,
                method="telea",
                seed=1234,
            )

    def test_process_image_missing_mask(self):
        """Test error handling for missing manual mask."""
        missing_mask = self.temp_path / "nonexistent_mask.png"
        output_path = self.temp_path / "output.png"

        with self.assertRaises(FileNotFoundError):
            process_image(
                input_path=self.input_path,
                output_path=output_path,
                mask_mode="manual",
                mask_path=missing_mask,
                dilate=5,
                method="telea",
                seed=1234,
            )


class TestPipelineVideo(unittest.TestCase):
    """Test process_video function."""

    def setUp(self):
        """Create synthetic test video."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create test video (8 frames, 320x240, 24fps)
        self.input_path = self.temp_path / "test_video.mp4"
        self.output_path = self.temp_path / "output_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.input_path), fourcc, 24.0, (320, 240))

        for i in range(8):
            # Create frame with gradual motion
            frame = np.ones((240, 320, 3), dtype=np.uint8) * (30 * i)
            # Add watermark in bottom-right
            frame[180:, 240:] = 255
            writer.write(frame)

        writer.release()

        # Create mask
        self.test_mask = np.zeros((240, 320), dtype=np.uint8)
        self.test_mask[180:, 240:] = 255
        self.mask_path = self.temp_path / "mask.png"
        cv2.imwrite(str(self.mask_path), self.test_mask)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_process_video_basic(self):
        """Test basic video processing with sequential mode."""
        process_video(
            input_path=self.input_path,
            output_path=self.output_path,
            mask_mode="manual",
            mask_path=self.mask_path,
            dilate=3,
            method="telea",
            window=8,
            overlap=2,
            temporal_guidance=False,
            qc_expr=None,
            retry_dilate=0,
            seed=1234,
        )

        # Verify output exists
        self.assertTrue(self.output_path.exists())

        # Verify output video has correct frame count
        cap = cv2.VideoCapture(str(self.output_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.assertEqual(frame_count, 8)

    def test_process_video_with_temporal_guidance(self):
        """Test video processing with temporal guidance enabled."""
        process_video(
            input_path=self.input_path,
            output_path=self.output_path,
            mask_mode="manual",
            mask_path=self.mask_path,
            dilate=3,
            method="telea",
            window=4,
            overlap=1,
            temporal_guidance=True,
            qc_expr=None,
            retry_dilate=0,
            seed=1234,
        )

        self.assertTrue(self.output_path.exists())


class TestFlowEstimator(unittest.TestCase):
    """Test optical flow estimation with FlowEstimator."""

    def test_flow_estimator_opencv_fallback(self):
        """Test FlowEstimator with OpenCV backend (always available)."""
        # Create two synthetic frames with simple motion
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add a moving rectangle
        cv2.rectangle(frame1, (20, 20), (40, 40), (255, 255, 255), -1)
        cv2.rectangle(frame2, (25, 20), (45, 40), (255, 255, 255), -1)

        estimator = FlowEstimator(raft_path=None)  # Force OpenCV fallback
        flow = estimator.flow(frame1, frame2)

        # Verify flow output shape
        self.assertEqual(flow.shape, (100, 100, 2))
        self.assertEqual(flow.dtype, np.float32)

    def test_flow_warp(self):
        """Test image warping using computed flow."""
        # Create frame with distinct pattern
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (20, 20), (40, 40), (255, 0, 0), -1)

        # Create shifted frame
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (25, 20), (45, 40), (255, 0, 0), -1)

        estimator = FlowEstimator(raft_path=None)
        flow = estimator.flow(frame1, frame2)
        warped = estimator.warp(frame1, flow)

        # Verify warped output shape
        self.assertEqual(warped.shape, frame1.shape)


class TestTemporal(unittest.TestCase):
    """Test temporal utilities (chunking and blending)."""

    def test_make_chunks_exact_division(self):
        """Test chunking with exact division (no remainder)."""
        chunks = make_chunks(total=100, window=10, overlap=2)

        # Verify chunk structure
        self.assertEqual(len(chunks), 13)  # (100 - 10) / (10 - 2) + 1

        # First chunk
        self.assertEqual(chunks[0], (0, 10))

        # Last chunk should end at total
        self.assertEqual(chunks[-1][1], 100)

    def test_make_chunks_with_remainder(self):
        """Test chunking with remainder."""
        chunks = make_chunks(total=15, window=10, overlap=3)

        # Verify last chunk extends to total
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[-1][1], 15)

    def test_make_chunks_single_chunk(self):
        """Test chunking when window >= total."""
        chunks = make_chunks(total=10, window=20, overlap=5)

        # Should produce single chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], (0, 10))

    def test_blend_overlap(self):
        """Test flow-warped blending at chunk boundaries."""
        # Create two overlapping chunks (frames)
        chunk_a = [
            np.ones((100, 100, 3), dtype=np.uint8) * 50,
            np.ones((100, 100, 3), dtype=np.uint8) * 60,
        ]
        chunk_b = [
            np.ones((100, 100, 3), dtype=np.uint8) * 70,
            np.ones((100, 100, 3), dtype=np.uint8) * 80,
        ]

        estimator = FlowEstimator(raft_path=None)
        blended = blend_overlap(chunk_a, chunk_b, overlap=1, estimator=estimator)

        # Verify blended output
        self.assertEqual(len(blended), 3)  # len(chunk_a) + len(chunk_b) - overlap
        self.assertEqual(blended[0].shape, (100, 100, 3))

        # First frame should be from chunk_a
        np.testing.assert_array_equal(blended[0], chunk_a[0])

        # Last frame should be from chunk_b
        np.testing.assert_array_equal(blended[-1], chunk_b[-1])


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def test_end_to_end_image_pipeline(self):
        """Test complete image processing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test image with watermark
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add visible watermark
            img[400:450, 500:600] = [255, 255, 255]

            input_path = temp_path / "input.png"
            output_path = temp_path / "output.png"
            cv2.imwrite(str(input_path), img)

            # Process with auto-masking
            process_image(
                input_path=input_path,
                output_path=output_path,
                mask_mode="auto",
                mask_path=None,
                dilate=5,
                method="telea",
                seed=1234,
            )

            # Verify output
            result = cv2.imread(str(output_path))
            self.assertEqual(result.shape, img.shape)

            # Watermark region should be modified
            self.assertFalse(np.array_equal(result[400:450, 500:600], img[400:450, 500:600]))


if __name__ == "__main__":
    unittest.main()
