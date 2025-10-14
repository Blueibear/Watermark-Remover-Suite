"""Image watermark removal engine built on OpenCV inpainting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import cv2
import numpy as np

from . import utils

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

INPAINT_METHODS = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


class ImageWatermarkRemover:
    """High-level helper for removing watermarks from still images."""

    def __init__(
        self,
        inpaint_radius: int = 3,
        method: str = "telea",
        *,
        auto_mask_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        if inpaint_radius <= 0:
            raise ValueError("inpaint_radius must be a positive integer.")
        if method not in INPAINT_METHODS:
            raise ValueError(f"Unsupported inpainting method: {method}")
        self.inpaint_radius = inpaint_radius
        self.method = method
        self._cv_method = INPAINT_METHODS[method]
        self.auto_mask_defaults = dict(auto_mask_defaults or {})
        logger.debug(
            "Initialized ImageWatermarkRemover (radius=%s, method=%s)",
            self.inpaint_radius,
            self.method,
        )

    def remove_watermark(
        self,
        image: np.ndarray,
        *,
        mask: Optional[Union[np.ndarray, PathLike]] = None,
        auto_mask_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove a watermark from an image array.

        Args:
            image: Input BGR image.
            mask: Optional binary mask or mask path.
            auto_mask_kwargs: Parameters for automatic mask detection when mask is None.

        Returns:
            Tuple of (result_image, mask_used).
        """
        if image is None or image.size == 0:
            raise ValueError("Cannot remove watermark from an empty image.")

        effective_mask_kwargs: Dict[str, Any] = dict(self.auto_mask_defaults)
        if auto_mask_kwargs:
            effective_mask_kwargs.update(auto_mask_kwargs)

        prepared_mask = utils.resolve_mask(mask, image, detector_kwargs=effective_mask_kwargs)
        result = cv2.inpaint(image, prepared_mask, self.inpaint_radius, self._cv_method)
        logger.debug(
            "Inpainted image with radius=%s (method=%s).",
            self.inpaint_radius,
            self.method,
        )
        return result, prepared_mask

    def process_file(
        self,
        input_path: PathLike,
        output_path: PathLike,
        *,
        mask_path: Optional[PathLike] = None,
        auto_mask_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        """Remove watermark from a file and store the result."""
        image = utils.load_image(input_path)
        logger.info("Processing image %s", input_path)
        result, mask_used = self.remove_watermark(
            image, mask=mask_path, auto_mask_kwargs=auto_mask_kwargs
        )
        utils.save_image(output_path, result)
        mask_output_path = Path(output_path).with_suffix(".mask.png")
        utils.save_image(mask_output_path, mask_used)
        return Path(output_path), mask_output_path

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ImageWatermarkRemover":
        settings = dict(config.get("image_processing", {}))
        detection_defaults = dict(settings.get("detection", {}))
        return cls(
            inpaint_radius=int(settings.get("inpaint_radius", 3)),
            method=settings.get("inpaint_method", "telea"),
            auto_mask_defaults=detection_defaults,
        )


__all__ = ["ImageWatermarkRemover"]
