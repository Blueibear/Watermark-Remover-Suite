from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ✅ Real signtool signing setup
SIGNTOOL = r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"
DEFAULT_PFX_PATH = "certs/watermark_dev.pfx"
DEFAULT_PFX_PASSWORD = "testpassword"
SHA_SUMS_FILE = Path("installers/build/SHA256SUMS.txt")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real signing for Watermark Remover Suite.")
    parser.add_argument("--input", type=Path, required=True, help="Path to unsigned executable.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write signed executable.")
    parser.add_argument("--log", type=Path, default=Path("signature_verification.log"), help="File to append signing log entries.")
    return parser.parse_args(argv)

def sha256(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()

def sign_with_signtool(target: Path, pfx: str, pwd: str):
    print(f"🔏 Signing {target.name}...")
    result = subprocess.run([
        SIGNTOOL,
        "sign",
        "/f", pfx,
        "/p", pwd,
        "/fd", "SHA256",
        "/tr", "http://timestamp.digicert.com",
        "/td", "SHA256",
        str(target)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Signing failed for {target.name}:\n{result.stderr}")
        sys.exit(1)
    print(f"✅ Signed {target.name}")

def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    # Copy input to output first
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input, args.output)

    # Use env vars or fallback defaults
    cert_path = os.getenv("CODESIGN_PFX_PATH", DEFAULT_PFX_PATH)
    cert_password = os.getenv("CODESIGN_PFX_PASSWORD", DEFAULT_PFX_PASSWORD)

    # Sign the output file
    sign_with_signtool(args.output, cert_path, cert_password)

    # Calculate and write hash
    hash_val = sha256(args.output)
    SHA_SUMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SHA_SUMS_FILE.open("a", encoding="utf-8") as sums:
        sums.write(f"{hash_val}  {args.output.name}\n")

    # Write log
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(f"Signed {args.input} -> {args.output}\n")
        handle.write(f"CODESIGN_PFX_PATH={cert_path}\n")
        handle.write(f"CODESIGN_PFX_PASSWORD={'***' if cert_password != 'unset' else 'unset'}\n")
        handle.write(f"SHA256: {hash_val}\n")

if __name__ == "__main__":
    main()