"""Mock GitHub release publishing for Phase 13."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock GitHub release publishing.")
    parser.add_argument("--repo", default="Watermark-Remover-Suite", help="Repository name.")
    parser.add_argument("--tag", default="v1.0.0", help="Release tag.")
    parser.add_argument("--title", default="Initial Stable Release", help="Release title.")
    parser.add_argument(
        "--notes",
        type=Path,
        default=Path("verification_reports/final_summary.md"),
        help="Path to release notes file.",
    )
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=[
            "dist/WatermarkRemoverSuite_signed.exe",
            "installers/build/WatermarkRemoverSuite_Setup.exe",
        ],
        help="Artifact paths to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("release_confirmation.json"),
        help="Path to write confirmation JSON.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("github_upload_log.txt"),
        help="Path to append upload log.",
    )
    return parser.parse_args(argv)


def _read_notes(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    github_token = os.getenv("GITHUB_TOKEN", "unset")
    artifacts = [str(Path(p)) for p in args.artifacts]

    args.output.write_text(
        json.dumps(
            {
                "repository": args.repo,
                "tag": args.tag,
                "title": args.title,
                "notes": _read_notes(Path(args.notes)),
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        handle.write(f"Mock release to {args.repo} {args.tag} ({args.title})\n")
        handle.write(f"Notes source: {args.notes}\n")
        handle.write(f"Artifacts: {', '.join(artifacts)}\n")
        handle.write(f"GITHUB_TOKEN={'***' if github_token != 'unset' else 'unset'}\n")


if __name__ == "__main__":
    main()
