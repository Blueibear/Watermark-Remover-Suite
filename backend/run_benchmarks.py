"""Run lightweight benchmarks against sample media assets."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from watermark_remover.config import load_config
from watermark_remover.core.logger import setup_logging

from watermark_remover.core import ImageWatermarkRemover, VideoWatermarkRemover

logger = logging.getLogger(__name__)


def _benchmark_images(remover: ImageWatermarkRemover, images_dir: Path, output_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for watermarked_path in sorted(images_dir.glob("*_watermarked.png")):
        stem = watermarked_path.stem.replace("_watermarked", "")
        mask_path = images_dir / f"{stem}_mask.png"
        output_path = output_dir / f"{stem}_restored.png"

        start = time.perf_counter()
        remover.process_file(watermarked_path, output_path, mask_path=mask_path if mask_path.exists() else None)
        elapsed = time.perf_counter() - start
        logger.info("Image %s processed in %.3fs", watermarked_path.name, elapsed)
        results.append(
            {
                "type": "image",
                "input": str(watermarked_path),
                "mask": str(mask_path) if mask_path.exists() else None,
                "output": str(output_path),
                "time_seconds": round(elapsed, 4),
            }
        )
    return results


def _benchmark_videos(remover: VideoWatermarkRemover, videos_dir: Path, output_dir: Path) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for video_path in sorted(videos_dir.glob("*.mp4")):
        start = time.perf_counter()
        output_path = output_dir / video_path.name.replace("_watermarked", "_restored")
        remover.process_file(video_path, output_path)
        elapsed = time.perf_counter() - start
        logger.info("Video %s processed in %.3fs", video_path.name, elapsed)
        results.append(
            {
                "type": "video",
                "input": str(video_path),
                "output": str(output_path),
                "time_seconds": round(elapsed, 4),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sample benchmarks.")
    parser.add_argument("--images", type=Path, default=Path("sample_inputs/images"), help="Directory containing sample images.")
    parser.add_argument("--videos", type=Path, default=Path("sample_inputs/videos"), help="Directory containing sample videos.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"), help="Directory to write results into.")
    parser.add_argument("--config", type=Path, default=None, help="Optional config path for remover settings.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--videos-enabled", action=argparse.BooleanOptionalAction, default=True, help="Include video benchmarks.")
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> Dict[str, List[Dict[str, object]]]:
    if args is None:
        args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s | %(message)s")
    setup_logging({"level": args.log_level.upper(), "console": {"enabled": True}}, force=True)

    results_dir = args.output
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.images
    videos_dir = args.videos

    remover_config = load_config(args.config) if args.config else load_config()
    image_remover = ImageWatermarkRemover.from_config(remover_config)
    video_remover = VideoWatermarkRemover.from_config(remover_config, image_remover=image_remover)

    benchmark_data: Dict[str, List[Dict[str, object]]] = {"images": [], "videos": []}
    if images_dir.exists():
        benchmark_data["images"] = _benchmark_images(image_remover, images_dir, results_dir / "images")
    else:
        logger.warning("Images directory %s not found.", images_dir)

    if args.videos_enabled and videos_dir.exists():
        benchmark_data["videos"] = _benchmark_videos(video_remover, videos_dir, results_dir / "videos")
    elif args.videos_enabled:
        logger.warning("Videos directory %s not found; skipping video benchmarks.", videos_dir)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    logger.info("Benchmark summary written to %s", summary_path)
    return benchmark_data


if __name__ == "__main__":
    main()
