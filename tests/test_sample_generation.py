import argparse
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import watermark_remover.backend.run_benchmarks as run_benchmarks
from watermark_remover.backend.generate_samples import generate_images, generate_videos


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

    def test_main_runs_without_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            images_dir = Path(tmp_dir_name) / "img"
            videos_dir = Path(tmp_dir_name) / "vid"
            test_args = [
                "generate_samples",
                "--images-dir",
                str(images_dir),
                "--videos-dir",
                str(videos_dir),
                "--image-count",
                "1",
                "--video-count",
                "1",
                "--no-videos",
                "--log-level",
                "DEBUG",
            ]
            with mock.patch.object(sys, "argv", test_args):
                runpy.run_module("backend.generate_samples", run_name="__main__")
            self.assertTrue(any(images_dir.iterdir()))

    def test_run_benchmarks_skips_missing_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            out_dir = Path(tmp_dir_name) / "bench"
            bench_args = argparse.Namespace(
                images=Path(tmp_dir_name) / "missing_images",
                videos=Path(tmp_dir_name) / "missing_videos",
                output=out_dir,
                config=None,
                log_level="INFO",
                videos_enabled=False,
            )
            result = run_benchmarks.main(bench_args)
            self.assertEqual(result["images"], [])
            self.assertEqual(result["videos"], [])

    def test_run_benchmarks_on_sample_inputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sample_images = root / "sample_inputs" / "images"
        if not sample_images.exists():
            self.skipTest("Sample images not available.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "bench"
            bench_args = argparse.Namespace(
                images=sample_images,
                videos=root / "sample_inputs" / "videos",
                output=output_dir,
                config=root / "config" / "config.yaml",
                log_level="INFO",
                videos_enabled=False,
            )
            result = run_benchmarks.main(bench_args)
            self.assertGreater(len(result["images"]), 0)
            bench_summary = bench_args.output / "summary.json"
            self.assertTrue(bench_summary.exists())

    def test_run_benchmarks_with_videos(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sample_videos = root / "sample_inputs" / "videos"
        if not sample_videos.exists():
            self.skipTest("Sample videos not available.")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "bench"
            bench_args = argparse.Namespace(
                images=root / "sample_inputs" / "images",
                videos=sample_videos,
                output=output_dir,
                config=root / "config" / "config.yaml",
                log_level="INFO",
                videos_enabled=True,
            )
            result = run_benchmarks.main(bench_args)
            self.assertGreater(len(result["videos"]), 0)


if __name__ == "__main__":
    unittest.main()
