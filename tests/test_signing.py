import tempfile
import unittest
from pathlib import Path

import pytest
from watermark_remover.backend import sign_release


class TestSigningPlaceholder(unittest.TestCase):
    @pytest.mark.windows_only
    @pytest.mark.integration
    def test_sign_release_copies_file_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            input_exe = tmp_dir / "input.exe"
            output_exe = tmp_dir / "output.exe"
            log_file = tmp_dir / "signature.log"

            # Simulate an EXE binary with placeholder text
            input_exe.write_text("binary-placeholder", encoding="utf-8")

            args = sign_release.parse_args(
                [
                    "--input",
                    str(input_exe),
                    "--output",
                    str(output_exe),
                    "--log",
                    str(log_file),
                ]
            )

            sign_release.main(args)

            # Assertions
            self.assertTrue(output_exe.exists(), "Output EXE not created")
            self.assertEqual(output_exe.read_text(encoding="utf-8"), "binary-placeholder")
            self.assertTrue(log_file.exists(), "Signature log not created")
            log_content = log_file.read_text(encoding="utf-8")
            self.assertIn("Mock signed", log_content)


if __name__ == "__main__":
    unittest.main()
