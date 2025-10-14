import tempfile
import unittest
from pathlib import Path

from backend.generate_samples import generate_images, generate_videos


class TestSampleGeneration(unittest.TestCase):
    def test_generate_images_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            output_dir = Path(tmp_dir_name) / "images"
            paths = generate_images(output_dir, count=2)
            self.assertEqual(len(paths), 2)
            for watermarked, mask in paths:
                self.assertTrue(watermarked.exists())
                self.assertTrue(mask.exists())

    def test_generate_videos_creates_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            output_dir = Path(tmp_dir_name) / "videos"
            videos = generate_videos(output_dir, count=1)
            self.assertEqual(len(videos), 1)
            for video in videos:
                self.assertTrue(video.exists())
                self.assertGreater(video.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
