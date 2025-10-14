import logging
import tempfile
import unittest
from pathlib import Path

import cv2
import yaml

from cli import main as cli_main
from config import DEFAULT_CONFIG_PATH
from core import utils
from .helpers import create_synthetic_sample, create_test_video_clip


class TestCLI(unittest.TestCase):
    def run_cli(self, args: list[str]) -> int:
        exit_code = cli_main.main(args)
        logging.shutdown()
        logging.getLogger().handlers.clear()
        return exit_code

    def test_image_command_processes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            _, watermarked = create_synthetic_sample()
            input_path = tmp_dir / "watermarked.png"
            output_path = tmp_dir / "restored.png"
            cv2.imwrite(str(input_path), watermarked)

            log_file = tmp_dir / "cli.log"
            exit_code = self.run_cli(
                [
                    "--config",
                    str(DEFAULT_CONFIG_PATH),
                    "--log-level",
                    "DEBUG",
                    "--log-file",
                    str(log_file),
                    "image",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists(), "Output image not created.")
            mask_path = output_path.with_suffix(".mask.png")
            self.assertTrue(mask_path.exists(), "Mask image not created.")
            self.assertTrue(log_file.exists(), "Log file override was not respected.")
            restored = utils.load_image(output_path)
            self.assertEqual(restored.shape, watermarked.shape)

    def test_video_command_processes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            input_path, _, _ = create_test_video_clip(tmp_dir)
            output_path = tmp_dir / "restored.mp4"

            exit_code = self.run_cli(
                [
                    "--config",
                    str(DEFAULT_CONFIG_PATH),
                    "video",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--bitrate",
                    "500k",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists(), "Output video not created.")

    def test_batch_command_runs_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)

            # Prepare image asset.
            _, watermarked = create_synthetic_sample()
            image_input = tmp_dir / "watermarked.png"
            image_output = tmp_dir / "restored.png"
            cv2.imwrite(str(image_input), watermarked)

            # Prepare video asset.
            video_input, _, _ = create_test_video_clip(tmp_dir)
            video_output = tmp_dir / "restored.mp4"

            manifest = [
                {
                    "type": "image",
                    "input": str(image_input),
                    "output": str(image_output),
                    "auto_mask": {"threshold": 240},
                },
                {
                    "type": "video",
                    "input": str(video_input),
                    "output": str(video_output),
                },
            ]

            manifest_path = tmp_dir / "manifest.yaml"
            manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

            exit_code = self.run_cli(
                [
                    "--config",
                    str(DEFAULT_CONFIG_PATH),
                    "batch",
                    "--manifest",
                    str(manifest_path),
                    "--max-workers",
                    "1",
                    "--halt-on-error",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(image_output.exists(), "Image output missing after batch run.")
            self.assertTrue(video_output.exists(), "Video output missing after batch run.")
            mask_output = image_output.with_suffix(".mask.png")
            self.assertTrue(mask_output.exists(), "Image mask missing after batch run.")


if __name__ == "__main__":
    unittest.main()
