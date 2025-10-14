from __future__ import annotations

from math import pi
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from moviepy import AudioArrayClip, ImageSequenceClip


def create_synthetic_sample(
    width: int = 160,
    height: int = 100,
    *,
    text: str = "WM",
    thickness: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a gradient background and an overlaid watermark."""
    gradient = np.tile(np.linspace(60, 200, width, dtype=np.uint8), (height, 1))
    base = np.dstack([gradient] * 3)
    watermarked = base.copy()
    cv2.putText(
        watermarked,
        text,
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return base, watermarked


def create_test_video_clip(
    tmp_dir: Path,
    *,
    fps: int = 5,
    frame_count: int = 8,
    include_audio: bool = True,
) -> Tuple[Path, List[np.ndarray], List[np.ndarray]]:
    """Generate a synthetic video clip with optional audio for testing."""
    base_frames: List[np.ndarray] = []
    watermarked_frames_rgb: List[np.ndarray] = []
    for _ in range(frame_count):
        base, watermarked = create_synthetic_sample()
        base_frames.append(base)
        watermarked_frames_rgb.append(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))

    clip = ImageSequenceClip(watermarked_frames_rgb, fps=fps)

    if include_audio:
        duration = clip.duration
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
        audio_wave = (0.2 * np.sin(2 * pi * 440 * t)).astype(np.float32)
        audio_clip = AudioArrayClip(audio_wave.reshape(-1, 1), fps=sample_rate)
        clip = clip.with_audio(audio_clip)

    input_path = tmp_dir / "watermarked.mp4"
    clip.write_videofile(
        str(input_path),
        codec="libx264",
        audio_codec="aac" if include_audio else None,
        fps=fps,
        logger=None,
    )
    clip.close()
    return input_path, base_frames, watermarked_frames_rgb
