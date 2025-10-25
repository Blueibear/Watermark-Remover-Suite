"""Utility script to prepare model cache directories for optional backends."""

from __future__ import annotations

import argparse
from pathlib import Path

CACHE_DIR = Path.home() / ".wmr" / "models"

MODELS = {
    "raft": {
        "url": "<add RAFT weights URL or instructions>",
        "sha256": "<fill>",
    },
    "lama": {
        "url": "<add LaMa ONNX URL or instructions>",
        "sha256": "<fill>",
    },
}


def ensure_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def print_instructions() -> None:
    print(f"Model cache directory: {CACHE_DIR}")
    for name, meta in MODELS.items():
        print(f"- {name}: download from {meta['url']} and place weights here; set sha256 later.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare model cache for the Watermark Remover Suite."
    )
    parser.add_argument(
        "--all", action="store_true", help="Reserved flag for future automatic downloads."
    )
    parser.parse_args(argv)
    ensure_cache()
    print_instructions()


if __name__ == "__main__":
    main()
