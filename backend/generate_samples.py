"""Generate synthetic sample media for the Watermark Remover Suite."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from moviepy import ImageSequenceClip

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_COUNT = 3
DEFAULT_VIDEO_COUNT = 2
IMAGE_SIZE = (256, 160)  # width, height
VIDEO_FPS = 8
VIDEO_FRAMES = 24


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _base_gradient(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(50, 200, width, dtype=np.float32)
    y_noise = rng.normal(0, 5, size=(height, 1)).astype(np.float32)
    gradient = np.clip(x + y_noise, 0, 255).astype(np.uint8)
    base = np.dstack([gradient] * 3)
    return base


def _apply_watermark(image: np.ndarray, text: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    watermarked = image.copy()
    font_scale = rng.uniform(1.2, 2.0)
    thickness = rng.integers(2, 5)
    position = (
        rng.integers(10, image.shape[1] // 3),
        rng.integers(image.shape[0] // 2, image.shape[0] - 10),
    )
    color = tuple(int(c) for c in rng.integers(200, 255, size=3))
    cv2.putText(
        watermarked,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        int(thickness),
        cv2.LINE_AA,
    )
    return watermarked


def _save_image_pair(
    output_dir: Path, index: int, base: np.ndarray, watermarked: np.ndarray
) -> None:
    base_path = output_dir / f"sample_{index:02d}_base.png"
    watermarked_path = output_dir / f"sample_{index:02d}_watermarked.png"
    mask_path = output_dir / f"sample_{index:02d}_mask.png"

    cv2.imwrite(str(base_path), base)
    cv2.imwrite(str(watermarked_path), watermarked)

    gray = cv2.cvtColor(watermarked, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(mask_path), mask)

    logger.info("Generated sample image %s", watermarked_path)


def generate_images(output_dir: Path, count: int = DEFAULT_IMAGE_COUNT) -> List[Tuple[Path, Path]]:
    _ensure_dir(output_dir)
    outputs = []
    for idx in range(count):
        base = _base_gradient(*IMAGE_SIZE, seed=idx)
        watermarked = _apply_watermark(base, text=f"WATERMARK {idx+1}", seed=idx + 100)
        _save_image_pair(output_dir, idx + 1, base, watermarked)
        outputs.append(
            (
                output_dir / f"sample_{idx+1:02d}_watermarked.png",
                output_dir / f"sample_{idx+1:02d}_mask.png",
            )
        )
    return outputs


def _generate_video_frames(seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    width, height = IMAGE_SIZE
    base = _base_gradient(width, height, seed=seed)
    for frame_idx in range(VIDEO_FRAMES):
        shifted = np.roll(base, shift=frame_idx * 2, axis=1)
        watermark = _apply_watermark(shifted, text="VIDEO WM", seed=seed + frame_idx)
        if frame_idx % 2 == 0:
            noise = rng.normal(0, 8, size=watermark.shape).astype(np.int16)
            watermark = np.clip(watermark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        yield cv2.cvtColor(watermark, cv2.COLOR_BGR2RGB)


def generate_videos(output_dir: Path, count: int = DEFAULT_VIDEO_COUNT) -> List[Path]:
    _ensure_dir(output_dir)
    outputs: List[Path] = []
    for idx in range(count):
        video_path = output_dir / f"sample_{idx+1:02d}_watermarked.mp4"
        frames = list(_generate_video_frames(seed=idx + 200))
        clip = ImageSequenceClip(frames, fps=VIDEO_FPS)
        clip.write_videofile(str(video_path), codec="libx264", audio=False, logger=None)
        clip.close()
        logger.info("Generated sample video %s", video_path)
        outputs.append(video_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample media assets.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("sample_inputs/images"),
        help="Directory for sample images.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("sample_inputs/videos"),
        help="Directory for sample videos.",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=DEFAULT_IMAGE_COUNT,
        help="Number of synthetic images to generate.",
    )
    parser.add_argument(
        "--video-count",
        type=int,
        default=DEFAULT_VIDEO_COUNT,
        help="Number of synthetic videos to generate.",
    )
    parser.add_argument(
        "--videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable video generation.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging verbosity level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
    )

    logger.info("Generating sample images into %s", args.images_dir)
    generate_images(args.images_dir, count=args.image_count)

    if args.videos:
        logger.info("Generating sample videos into %s", args.videos_dir)
        generate_videos(args.videos_dir, count=args.video_count)
    else:
        logger.info("Skipping video generation per configuration.")


if __name__ == "__main__":
    main()
