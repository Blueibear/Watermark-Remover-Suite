"""Command-line interface for the Watermark Remover Suite."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from watermark_remover.config import DEFAULT_CONFIG_PATH, load_config
from watermark_remover.core.logger import setup_logging

from watermark_remover.core import (
    BatchItem,
    BatchResult,
    BatchWatermarkProcessor,
    ImageWatermarkRemover,
    VideoWatermarkRemover,
)

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError as exc:  # pragma: no cover - argparse failure path
        raise argparse.ArgumentTypeError(f"Expected integer, received '{value}'") from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return ivalue


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watermark-remover",
        description="Watermark Remover Suite command-line interface.",
        epilog="Full documentation: docs/user_guide.md",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to configuration YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--log-level",
        help="Override logging level (e.g. INFO, DEBUG).",
    )
    parser.add_argument(
        "--log-file",
        help="Override log file path.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser(
        "image", help="Process a single image file."
    )
    image_parser.add_argument("-i", "--input", required=True, help="Path to the input image.")
    image_parser.add_argument("-o", "--output", required=True, help="Path for the restored image.")
    image_parser.add_argument("-m", "--mask", help="Optional binary mask image path.")
    image_parser.add_argument("--inpaint-radius", type=_positive_int, help="Override inpainting radius.")
    image_parser.add_argument(
        "--inpaint-method",
        choices=["telea", "ns"],
        help="Override inpainting method.",
    )
    image_parser.add_argument("--auto-threshold", type=int, help="Override mask detection threshold.")
    image_parser.add_argument("--auto-dilate", type=int, help="Override mask dilation iterations.")
    image_parser.add_argument("--auto-blur", type=int, help="Override mask blur kernel size.")

    video_parser = subparsers.add_parser(
        "video", help="Process a single video file."
    )
    video_parser.add_argument("-i", "--input", required=True, help="Path to the input video.")
    video_parser.add_argument("-o", "--output", required=True, help="Path for the restored video.")
    video_parser.add_argument("-m", "--mask", help="Optional binary mask image path.")
    video_parser.add_argument("--reuse-mask", action=argparse.BooleanOptionalAction, default=None)
    video_parser.add_argument("--preserve-audio", action=argparse.BooleanOptionalAction, default=None)
    video_parser.add_argument("--codec", help="Override video codec (default from config).")
    video_parser.add_argument("--audio-codec", help="Override audio codec (default from config).")
    video_parser.add_argument("--bitrate", help="Override output bitrate, e.g. 4M.")
    video_parser.add_argument("--auto-threshold", type=int, help="Override mask detection threshold.")
    video_parser.add_argument("--auto-dilate", type=int, help="Override mask dilation iterations.")
    video_parser.add_argument("--auto-blur", type=int, help="Override mask blur kernel size.")

    batch_parser = subparsers.add_parser(
        "batch", help="Process a batch manifest describing multiple jobs."
    )
    batch_parser.add_argument(
        "-m",
        "--manifest",
        required=True,
        help="Path to a YAML or JSON manifest describing batch jobs.",
    )
    batch_parser.add_argument(
        "--max-workers",
        type=_positive_int,
        help="Override maximum concurrent workers in batch processor.",
    )
    batch_parser.add_argument(
        "--halt-on-error",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop processing remaining items after the first failure.",
    )

    return parser


def _apply_logging_overrides(overrides: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.log_level:
        overrides.setdefault("logging", {})["level"] = args.log_level.upper()
    if args.log_file:
        logging_overrides = overrides.setdefault("logging", {})
        file_overrides = logging_overrides.setdefault("file", {})
        file_overrides["enabled"] = True
        file_overrides["filename"] = args.log_file


def _apply_batch_overrides(overrides: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.command != "batch":
        return
    batch_overrides = overrides.setdefault("batch", {})
    if args.max_workers is not None:
        batch_overrides["max_workers"] = args.max_workers
    if args.halt_on_error is not None:
        batch_overrides["halt_on_error"] = args.halt_on_error


def _configure_logging(config: Dict[str, Any]) -> None:
    logging_settings = config.get("logging", {})
    setup_logging(logging_settings, force=True)


def _build_image_remover(config: Dict[str, Any], args: argparse.Namespace) -> ImageWatermarkRemover:
    settings = dict(config.get("image_processing", {}))
    detection = dict(settings.get("detection", {}))

    radius = getattr(args, "inpaint_radius", None)
    method = getattr(args, "inpaint_method", None)
    auto_threshold = getattr(args, "auto_threshold", None)
    auto_dilate = getattr(args, "auto_dilate", None)
    auto_blur = getattr(args, "auto_blur", None)

    if radius is not None:
        settings["inpaint_radius"] = radius
    if method is not None:
        settings["inpaint_method"] = method
    if auto_threshold is not None:
        detection["threshold"] = auto_threshold
    if auto_dilate is not None:
        detection["dilate_iterations"] = auto_dilate
    if auto_blur is not None:
        detection["blur_kernel"] = auto_blur

    remover = ImageWatermarkRemover(
        inpaint_radius=int(settings.get("inpaint_radius", 3)),
        method=settings.get("inpaint_method", "telea"),
        auto_mask_defaults=detection,
    )
    return remover


def _build_video_remover(
    config: Dict[str, Any],
    args: argparse.Namespace,
    image_remover: Optional[ImageWatermarkRemover] = None,
) -> VideoWatermarkRemover:
    settings = dict(config.get("video_processing", {}))
    auto_mask = dict(settings.get("auto_mask", {}))

    if args.reuse_mask is not None:
        settings["reuse_mask"] = args.reuse_mask
    if args.preserve_audio is not None:
        settings["preserve_audio"] = args.preserve_audio
    if args.codec is not None:
        settings["codec"] = args.codec
    if args.audio_codec is not None:
        settings["audio_codec"] = args.audio_codec
    if args.bitrate is not None:
        settings["bitrate"] = args.bitrate
    if args.auto_threshold is not None:
        auto_mask["threshold"] = args.auto_threshold
    if args.auto_dilate is not None:
        auto_mask["dilate_iterations"] = args.auto_dilate
    if args.auto_blur is not None:
        auto_mask["blur_kernel"] = args.auto_blur

    remover = VideoWatermarkRemover(
        image_remover=image_remover or _build_image_remover(config, args),
        reuse_mask=bool(settings.get("reuse_mask", True)),
        preserve_audio=bool(settings.get("preserve_audio", True)),
        codec=settings.get("codec", "libx264"),
        audio_codec=settings.get("audio_codec", "aac"),
        bitrate=settings.get("bitrate"),
        auto_mask_defaults=auto_mask,
    )
    return remover


def _load_manifest(path: Path) -> Iterable[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("Batch manifest must be a list of job entries.")
    return data


def _prepare_batch_items(entries: Iterable[Dict[str, Any]]) -> List[BatchItem]:
    items: List[BatchItem] = []
    for entry in entries:
        if "type" not in entry:
            raise ValueError("Batch entry missing 'type' field.")
        input_path = entry.get("input")
        output_path = entry.get("output")
        if not input_path or not output_path:
            raise ValueError("Batch entry must include 'input' and 'output' fields.")
        auto_mask = entry.get("auto_mask")
        items.append(
            BatchItem(
                media_type=entry["type"],
                input_path=input_path,
                output_path=output_path,
                mask_path=entry.get("mask"),
                auto_mask_kwargs=auto_mask,
            )
        )
    return items


def _run_image(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    remover = _build_image_remover(config, args)
    output, mask_path = remover.process_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        mask_path=Path(args.mask) if args.mask else None,
        auto_mask_kwargs=None,  # Already handled via remover defaults.
    )
    logger.info("Image processed successfully: %s (mask saved to %s)", output, mask_path)
    return 0


def _run_video(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    image_remover = _build_image_remover(config, args)
    remover = _build_video_remover(config, args, image_remover=image_remover)
    remover.process_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        mask_path=Path(args.mask) if args.mask else None,
    )
    logger.info("Video processed successfully: %s", args.output)
    return 0


def _summarize_batch(results: List[BatchResult]) -> int:
    success = sum(1 for r in results if r.success)
    failures = [r for r in results if not r.success]
    logger.info("Batch complete. Successes: %s | Failures: %s", success, len(failures))
    for result in failures:
        logger.error(
            "Failed %s job for %s: %s",
            result.media_type,
            result.input_path,
            result.error,
        )
    return 0 if not failures else 1


def _run_batch(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    manifest_path = Path(args.manifest)
    entries = _load_manifest(manifest_path)
    items = _prepare_batch_items(entries)
    logger.info("Processing %s batch item(s) defined in %s", len(items), manifest_path)
    processor = BatchWatermarkProcessor(config=config)
    results = processor.process(items)
    return _summarize_batch(results)


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    _apply_logging_overrides(overrides, args)
    _apply_batch_overrides(overrides, args)

    config = load_config(args.config, overrides=overrides or None)
    _configure_logging(config)

    try:
        if args.command == "image":
            return _run_image(args, config)
        if args.command == "video":
            return _run_video(args, config)
        if args.command == "batch":
            return _run_batch(args, config)
        parser.error(f"Unknown command: {args.command}")  # pragma: no cover
    except Exception as exc:  # pragma: no cover - command failure path
        logger.exception("Command failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
