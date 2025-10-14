import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from moviepy import AudioArrayClip, ImageSequenceClip, VideoFileClip

from core import utils
from core.batch_manager import BatchItem, BatchWatermarkProcessor
from core.image_remover import ImageWatermarkRemover
from core.video_remover import VideoWatermarkRemover
from .helpers import create_synthetic_sample


def _build_test_clip(tmp_dir: Path, fps: int = 5) -> tuple[Path, list[np.ndarray], list[np.ndarray]]:
    base_frames = []
    watermarked_frames_rgb = []
    for _ in range(8):
        base, watermarked = create_synthetic_sample()
        base_frames.append(base)
        watermarked_frames_rgb.append(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))

    clip = ImageSequenceClip(watermarked_frames_rgb, fps=fps)
    duration = clip.duration
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_clip = AudioArrayClip(audio.reshape(-1, 1), fps=sample_rate)
    clip = clip.with_audio(audio_clip)

    input_path = tmp_dir / "watermarked.mp4"
    clip.write_videofile(
        str(input_path),
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        logger=None,
    )
    clip.close()
    return input_path, base_frames, watermarked_frames_rgb


class TestVideoWatermarkRemover(unittest.TestCase):
    def test_video_processing_preserves_audio(self) -> None:
        image_remover = ImageWatermarkRemover()
        remover = VideoWatermarkRemover(image_remover=image_remover, reuse_mask=True)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            input_path, base_frames, watermarked_frames = _build_test_clip(tmp_dir)
            output_path = tmp_dir / "restored.mp4"

            remover.process_file(
                input_path,
                output_path,
                auto_mask_kwargs={"threshold": 230, "dilate_iterations": 2, "blur_kernel": 5},
            )

            self.assertTrue(output_path.exists(), "Processed video was not created.")

            processed_clip = VideoFileClip(str(output_path))
            try:
                self.assertIsNotNone(processed_clip.audio, "Audio track was not preserved.")
                self.assertAlmostEqual(processed_clip.audio.fps, 44100, delta=1)

                restored_frame_rgb = processed_clip.get_frame(0)
                restored_frame_bgr = cv2.cvtColor(restored_frame_rgb, cv2.COLOR_RGB2BGR)
                original_frame_bgr = cv2.cvtColor(watermarked_frames[0], cv2.COLOR_RGB2BGR)
                base_frame = base_frames[0]
                mask = utils.detect_watermark_mask(
                    original_frame_bgr,
                    method="threshold",
                    threshold=230,
                    dilate_iterations=2,
                    blur_kernel=5,
                )
                restored_region = restored_frame_bgr[mask > 0]
                baseline_region = base_frame[mask > 0]
                mean_diff = np.mean(
                    np.abs(restored_region.astype(np.int16) - baseline_region.astype(np.int16))
                )
                self.assertLess(
                    mean_diff,
                    45,
                    "Watermark removal did not sufficiently restore masked region in video.",
                )
            finally:
                processed_clip.close()

    def test_batch_processor_handles_image_and_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)

            # Prepare synthetic assets.
            _, watermarked_image = create_synthetic_sample()
            image_input = tmp_dir / "watermarked.png"
            cv2.imwrite(str(image_input), watermarked_image)

            video_input, _, _ = _build_test_clip(tmp_dir)

            image_output = tmp_dir / "restored.png"
            video_output = tmp_dir / "restored.mp4"

            processor = BatchWatermarkProcessor()
            jobs = [
                BatchItem(
                    media_type="image",
                    input_path=image_input,
                    output_path=image_output,
                    auto_mask_kwargs={"threshold": 230},
                ),
                BatchItem(
                    media_type="video",
                    input_path=video_input,
                    output_path=video_output,
                    auto_mask_kwargs={"threshold": 230},
                ),
            ]

            results = processor.process(jobs)

            self.assertEqual(len(results), 2)
            for result in results:
                self.assertTrue(result.success, f"Batch item failed: {result.error}")
                self.assertTrue(result.output_path and result.output_path.exists())
                if result.media_type == "image":
                    self.assertTrue(result.mask_path and result.mask_path.exists())


if __name__ == "__main__":
    unittest.main()
