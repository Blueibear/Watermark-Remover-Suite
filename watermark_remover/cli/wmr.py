"""Minimal CLI wiring for the Watermark Remover Suite MVP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from watermark_remover.core.pipeline import process_image, process_video


def _add_common_image_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mask", choices=["auto", "manual"], default="auto")
    parser.add_argument("--dilate", type=int, default=5)
    parser.add_argument("--method", choices=["telea", "lama", "sd", "noop"], default="telea")
    parser.add_argument("--seed", type=int, default=1234)


def _add_common_video_args(parser: argparse.ArgumentParser) -> None:
    _add_common_image_args(parser)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--overlap", type=int, default=12)
    parser.add_argument("--temporal-guidance", default="none")
    parser.add_argument("--wm-estimation", default="none")
    parser.add_argument("--seam-blend", default="none")
    parser.add_argument(
        "--qc", default="warped_ssim>=0.92", help="QC gate, e.g. warped_ssim>=0.92 or none"
    )
    parser.add_argument("--retry", type=int, default=1, help="Retries per failing frame")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wmr", description="Watermark Remover Suite (MVP stub)")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    image_parser = subparsers.add_parser("image", help="Process a single image")
    _add_common_image_args(image_parser)

    video_parser = subparsers.add_parser("video", help="Process a video (sequential)")
    _add_common_video_args(video_parser)

    return parser


def _run_image(args: argparse.Namespace) -> int:
    process_image(
        args.input,
        args.out,
        method=args.method,
        mask_mode=args.mask,
        dilate=args.dilate,
        seed=args.seed,
    )
    return 0


def _run_video(args: argparse.Namespace) -> int:
    process_video(
        args.input,
        args.out,
        method=args.method,
        mask_mode=args.mask,
        dilate=args.dilate,
        seed=args.seed,
        window=args.window,
        overlap=args.overlap,
        qc=args.qc,
        retry=args.retry,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "image":
        return _run_image(args)
    if args.cmd == "video":
        return _run_video(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
