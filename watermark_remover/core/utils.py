"""Utility helpers for image watermark removal workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Common file extensions we explicitly allow when validating paths.
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def load_image(path: PathLike) -> np.ndarray:
    """Load an image as a BGR numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If OpenCV fails to decode the image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.warning("Attempting to load image with uncommon extension: %s", path.suffix)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    logger.debug("Loaded image %s with shape %s", path, image.shape)
    return image


def save_image(path: PathLike, image: np.ndarray) -> None:
    """Persist an image to disk, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image is None or image.size == 0:
        raise ValueError("Cannot save empty image.")
    success = cv2.imwrite(str(path), image)
    if not success:
        raise IOError(f"Failed to save image at {path}")
    logger.debug("Saved image to %s", path)


def detect_watermark_mask(
    image: np.ndarray,
    method: str = "threshold",
    *,
    threshold: int = 240,
    adaptive_block_size: int = 35,
    adaptive_c: int = 10,
    dilate_iterations: int = 1,
    blur_kernel: int = 3,
    invert: bool = False,
) -> np.ndarray:
    """Generate a candidate watermark mask using simple heuristics.

    Args:
        image: Input BGR image.
        method: One of {"threshold", "adaptive", "laplacian"}.
        threshold: Brightness threshold for the `threshold` or `laplacian` strategies.
        adaptive_block_size: Window size for adaptive thresholding (must be odd).
        adaptive_c: Constant subtracted in adaptive thresholding.
        dilate_iterations: Morphological dilation passes to grow the mask.
        blur_kernel: Size of the Gaussian blur kernel (must be odd when > 0).
        invert: Flip mask polarity when True.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot detect watermark mask from empty image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if method == "threshold":
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif method == "adaptive":
        block_size = adaptive_block_size + (1 - adaptive_block_size % 2)  # enforce odd
        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            adaptive_c,
        )
    elif method == "laplacian":
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        mask = np.uint8((np.abs(laplacian) > threshold) * 255)
    else:
        raise ValueError(f"Unsupported watermark detection method: {method}")

    if invert:
        mask = cv2.bitwise_not(mask)

    if blur_kernel and blur_kernel > 1:
        kernel_size = blur_kernel + (1 - blur_kernel % 2)
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigmaX=0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if dilate_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

    logger.debug(
        "Generated watermark mask via %s strategy (threshold=%s, invert=%s).",
        method,
        threshold,
        invert,
    )
    return mask


def prepare_mask(mask: np.ndarray, reference_shape: Tuple[int, int]) -> np.ndarray:
    """Ensure a mask matches the expected shape and dtype."""
    if mask is None or mask.size == 0:
        raise ValueError("Provided watermark mask is empty.")
    processed = mask
    if processed.ndim == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if processed.shape[:2] != reference_shape:
        raise ValueError(
            f"Mask shape {processed.shape[:2]} does not match image shape {reference_shape}."
        )
    if processed.dtype != np.uint8:
        processed = np.clip(processed, 0, 255).astype(np.uint8)
    _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY)
    return processed


def resolve_mask(
    mask: Optional[Union[np.ndarray, PathLike]],
    image: np.ndarray,
    detector_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Resolve a mask from an array, path, or fall back to auto-detection."""
    if mask is None:
        detector_kwargs = detector_kwargs or {}
        logger.info("No mask supplied; attempting automatic watermark detection.")
        return detect_watermark_mask(image, **detector_kwargs)

    if isinstance(mask, (str, Path)):
        mask_path = Path(mask)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask path not found: {mask_path}")
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise ValueError(f"Unable to decode mask image: {mask_path}")
    else:
        raw_mask = mask

    return prepare_mask(raw_mask, image.shape[:2])
