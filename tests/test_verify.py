import tempfile
import unittest
from pathlib import Path

from backend import verify_release


class TestVerifyRelease(unittest.TestCase):
    def test_verification_success_with_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            artifact = tmp_dir / "artifact.exe"
            artifact.write_text("binary", encoding="utf-8")
            checksum_file = tmp_dir / "SHA256SUMS.txt"
            checksum = verify_release._hash_file(artifact)
            checksum_file.write_text(f"{checksum}  {artifact.name}\n", encoding="utf-8")
            log_file = tmp_dir / "verify.log"

            args = verify_release.parse_args(
                [
                    "--artifacts",
                    str(artifact),
                    "--checksums",
                    str(checksum_file),
                    "--log",
                    str(log_file),
                ]
            )
            code, messages = verify_release.main(args)
            self.assertEqual(code, 0)
            self.assertTrue(any("OK" in msg for msg in messages))
            self.assertTrue(log_file.exists())

    def test_verification_detects_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            missing = tmp_dir / "missing.exe"
            log_file = tmp_dir / "verify.log"

            args = verify_release.parse_args(
                [
                    "--artifacts",
                    str(missing),
                    "--log",
                    str(log_file),
                ]
            )
            code, messages = verify_release.main(args)
            self.assertEqual(code, 1)
            self.assertTrue(any("Missing artifact" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()
