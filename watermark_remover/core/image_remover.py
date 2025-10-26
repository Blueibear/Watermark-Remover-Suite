"""Thin image wrapper for MVP pipeline - safe to import in tests."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

# Lazy import to avoid heavy work at import time
if TYPE_CHECKING:
    from .pipeline import process_image

@dataclass
class ImageWatermarkRemover:
    """Thin wrapper around MVP process_image for test compatibility."""

    method: str = "telea"
    mask_mode: str = "auto"
    dilate: int = 5
    seed: int = 1234
    inpaint_radius: int = 3  # for compatibility; not directly used in MVP
    auto_mask_defaults: Optional[dict[str, Any]] = None

    @classmethod
    def from_config(cls, config: dict, **kwargs: Any) -> "ImageWatermarkRemover":
        """Create from config dict (for UI compatibility)."""
        # Extract relevant fields from config, using defaults
        image_cfg = config.get("image", {})
        return cls(
            method=image_cfg.get("method", "telea"),
            mask_mode=image_cfg.get("mask_mode", "auto"),
            dilate=image_cfg.get("dilate", 5),
            seed=image_cfg.get("seed", 1234),
            inpaint_radius=image_cfg.get("inpaint_radius", 3),
            **kwargs
        )

    def run(self, input_path: str | Path, output_path: str | Path, **kwargs: Any) -> None:
        """Process image with optional overrides."""
        from .pipeline import process_image  # Import here to avoid import-time deps

        args: dict[str, Any] = dict(kwargs)
        args.setdefault("method", self.method)
        args.setdefault("mask_mode", self.mask_mode)
        args.setdefault("dilate", self.dilate)
        args.setdefault("seed", self.seed)
        process_image(Path(input_path), Path(output_path), **args)

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        mask_path: Optional[str | Path] = None,
        **kwargs: Any
    ) -> Tuple[Path, Path]:
        """Process image file (UI-compatible signature)."""
        self.run(input_path, output_path, **kwargs)
        # Return output path and a dummy mask path for UI compatibility
        return Path(output_path), Path(output_path).with_suffix(".mask.png")
