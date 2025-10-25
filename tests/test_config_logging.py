import logging
import os
import tempfile
import unittest
from pathlib import Path

from config import load_config
from core.batch_manager import BatchWatermarkProcessor
from core.image_remover import ImageWatermarkRemover
from core.logger import setup_logging
from core.video_remover import VideoWatermarkRemover


class TestConfigurationAndLogging(unittest.TestCase):
    def test_load_config_returns_expected_sections(self) -> None:
        config = load_config()
        self.assertIn("image_processing", config)
        self.assertIn("video_processing", config)
        self.assertIn("logging", config)

    def test_removers_from_config_use_defaults(self) -> None:
        config = load_config()
        image_remover = ImageWatermarkRemover.from_config(config)
        video_remover = VideoWatermarkRemover.from_config(config, image_remover=image_remover)

        image_settings = config["image_processing"]
        video_settings = config["video_processing"]

        self.assertEqual(image_remover.inpaint_radius, image_settings["inpaint_radius"])
        self.assertEqual(image_remover.method, image_settings["inpaint_method"])
        self.assertEqual(image_remover.auto_mask_defaults, image_settings["detection"])

        self.assertEqual(video_remover.codec, video_settings["codec"])
        self.assertEqual(video_remover.audio_codec, video_settings["audio_codec"])
        self.assertEqual(video_remover.auto_mask_defaults, video_settings["auto_mask"])

    def test_batch_processor_uses_config_defaults(self) -> None:
        config = load_config()
        processor = BatchWatermarkProcessor(config=config)

        batch_settings = config["batch"]
        self.assertEqual(processor.max_workers, batch_settings["max_workers"])
        self.assertEqual(processor.halt_on_error, batch_settings["halt_on_error"])

        # Ensure the processor is reusing configured removers.
        self.assertEqual(
            processor.image_remover.auto_mask_defaults,
            config["image_processing"]["detection"],
        )
        self.assertEqual(
            processor.video_remover.auto_mask_defaults,
            config["video_processing"]["auto_mask"],
        )

    def test_setup_logging_creates_file_handler(self) -> None:
        config = load_config()
        logging_settings = config["logging"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            expected_log_dir = Path(tmp_dir) / "WatermarkRemoverSuite" / "logs"
            expected_log_path = expected_log_dir / "suite.log"
            overrides = logging_settings.copy()
            overrides["file"] = dict(overrides["file"])
            overrides["file"]["filename"] = "%APPDATA%/WatermarkRemoverSuite/logs/suite.log"
            overrides["file"]["enabled"] = True
            overrides["console"] = {"enabled": False}

            original_appdata = os.environ.get("APPDATA")
            os.environ["APPDATA"] = tmp_dir
            try:
                setup_logging(overrides, force=True)
                logger = logging.getLogger("watermark.tests")
                logger.info("log-line")
                logging.shutdown()

                self.assertTrue(expected_log_path.exists(), "Log file was not created.")
                self.assertGreater(expected_log_path.stat().st_size, 0, "Log file is empty.")

                # Reset logging to avoid dangling handlers once the temp directory is removed.
                setup_logging(
                    {"level": "WARNING", "console": {"enabled": False}, "file": {"enabled": False}},
                    force=True,
                )
            finally:
                if original_appdata is not None:
                    os.environ["APPDATA"] = original_appdata
                else:
                    os.environ.pop("APPDATA", None)


if __name__ == "__main__":
    unittest.main()
