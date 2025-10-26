"""Thin image wrapper for MVP pipeline - safe to import in tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


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
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> "ImageWatermarkRemover":
        """Create from config dict (for UI compatibility)."""
        image_cfg = dict(config.get("image_processing", {}))
        detection_cfg = dict(image_cfg.get("detection", {}))
        dilate_default = detection_cfg.get("dilate_iterations", image_cfg.get("dilate", 5))
        if dilate_default is not None:
            try:
                dilate_default = int(dilate_default)
            except (TypeError, ValueError):
                dilate_default = 5
        return cls(
            method=image_cfg.get("inpaint_method", "telea"),
            mask_mode=image_cfg.get("mask_mode", "auto"),
            dilate=dilate_default if isinstance(dilate_default, int) else 5,
            seed=image_cfg.get("seed", 1234),
            inpaint_radius=image_cfg.get("inpaint_radius", 3),
            auto_mask_defaults=detection_cfg or None,
            **kwargs,
        )

    def run(self, input_path: str | Path, output_path: str | Path, **kwargs: Any) -> None:
        """Process image with optional overrides."""
        from .pipeline import process_image  # Import here to avoid import-time deps

        args: dict[str, Any] = dict(kwargs)
        # Compatibility: discard unsupported keys from higher-level callers
        auto_mask_kwargs = args.pop("auto_mask_kwargs", None)
        args.pop("auto_mask_defaults", None)
        if auto_mask_kwargs:
            # Allow simple overrides when provided
            dilate_override = auto_mask_kwargs.get("dilate_iterations")
            if dilate_override is not None:
                args.setdefault("dilate", dilate_override)
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
