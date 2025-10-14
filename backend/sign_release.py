"""Placeholder signing workflow for the Watermark Remover Suite.

This does not perform real signing; it simulates the workflow by copying the
input executable and noting the action in the signature log.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock signing for Watermark Remover Suite.")
    parser.add_argument("--input", type=Path, required=True, help="Path to unsigned executable.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the 'signed' executable.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("signature_verification.log"),
        help="File to append signing log entries.",
    )
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    cert_path = os.getenv("CODESIGN_PFX_PATH", "unset")
    cert_password = os.getenv("CODESIGN_PFX_PASSWORD", "unset")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input, args.output)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(f"Mock signed {args.input} -> {args.output}\n")
        handle.write(f"CODESIGN_PFX_PATH={cert_path}\n")
        handle.write(f"CODESIGN_PFX_PASSWORD={'***' if cert_password != 'unset' else 'unset'}\n")


if __name__ == "__main__":
    main()
