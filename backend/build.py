"""Build helpers for the Watermark Remover Suite."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
SPEC_PATH = PROJECT_ROOT / "backend" / "pyinstaller.spec"


def run_pyinstaller(extra_args: Sequence[str] | None = None) -> None:
    args = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "pyinstaller.exe"),
        str(SPEC_PATH),
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
    ]
    if extra_args:
        args.extend(extra_args)
    subprocess.check_call(args)
