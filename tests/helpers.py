from __future__ import annotations

import cv2
import numpy as np


def create_synthetic_sample(
    width: int = 160,
    height: int = 100,
    *,
    text: str = "WM",
    thickness: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
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
