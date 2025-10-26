import json
import os
import tempfile
import unittest
from pathlib import Path

from watermark_remover.backend import publish_release


class TestPublishRelease(unittest.TestCase):
    def test_publish_creates_confirmation_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            notes = tmp_dir / "notes.md"
            notes.write_text("Release notes placeholder", encoding="utf-8")
            output = tmp_dir / "release.json"
            log_file = tmp_dir / "upload.log"

            args = publish_release.parse_args(
                [
                    "--repo",
                    "ExampleRepo",
                    "--tag",
                    "v0.0.1",
                    "--title",
                    "Mock Release",
                    "--notes",
                    str(notes),
                    "--artifacts",
                    "artifact1.zip",
                    "artifact2.exe",
                    "--output",
                    str(output),
                    "--log",
                    str(log_file),
                ]
            )
            os.environ["GITHUB_TOKEN"] = "dummy-token"
            publish_release.main(args)

            self.assertTrue(output.exists())
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["tag"], "v0.0.1")
            self.assertIn("artifact1.zip", data["artifacts"])
            self.assertTrue(log_file.exists())
            log_content = log_file.read_text(encoding="utf-8")
            self.assertIn("Mock release", log_content)
            self.assertIn("***", log_content)


if __name__ == "__main__":
    unittest.main()
