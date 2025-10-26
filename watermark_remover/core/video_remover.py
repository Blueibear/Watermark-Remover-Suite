"""Thin video wrapper for MVP pipeline - safe to import in tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from .image_remover import ImageWatermarkRemover

@dataclass
class VideoWatermarkRemover:
    """Thin wrapper around MVP process_video for test compatibility."""

    image_remover: Optional["ImageWatermarkRemover"] = None
    method: str = "telea"
    mask_mode: str = "auto"
    dilate: int = 5
    seed: int = 1234
    window: int = 48
    overlap: int = 12
    qc: str = "warped_ssim>=0.92"
    retry: int = 1
    bitrate: Optional[str] = None  # for compatibility; not used in MVP
    extra: Optional[dict[str, Any]] = None
    reuse_mask: bool = True
    preserve_audio: bool = True
    codec: str = "libx264"
    audio_codec: str = "aac"
    auto_mask_defaults: Optional[dict[str, Any]] = None

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], **kwargs: Any
    ) -> "VideoWatermarkRemover":
        """Create from config dict (for UI compatibility)."""
        # Extract relevant fields from config, using defaults
        video_cfg = dict(config.get("video_processing", {}))
        auto_mask_defaults = dict(video_cfg.get("auto_mask", {}))
        dilate_default = auto_mask_defaults.get("dilate_iterations", video_cfg.get("dilate", 5))
        if dilate_default is not None:
            try:
                dilate_default = int(dilate_default)
            except (TypeError, ValueError):
                dilate_default = 5
        if "image_remover" not in kwargs:
            from .image_remover import ImageWatermarkRemover

            kwargs["image_remover"] = ImageWatermarkRemover.from_config(config)
        return cls(
            method=video_cfg.get("method", "telea"),
            mask_mode=video_cfg.get("mask_mode", "auto"),
            dilate=dilate_default if isinstance(dilate_default, int) else 5,
            seed=video_cfg.get("seed", 1234),
            window=video_cfg.get("window", 48),
            overlap=video_cfg.get("overlap", 12),
            qc=video_cfg.get("qc", "warped_ssim>=0.92"),
            retry=video_cfg.get("retry", 1),
            reuse_mask=bool(video_cfg.get("reuse_mask", True)),
            preserve_audio=bool(video_cfg.get("preserve_audio", True)),
            codec=video_cfg.get("codec", "libx264"),
            audio_codec=video_cfg.get("audio_codec", "aac"),
            bitrate=video_cfg.get("bitrate"),
            auto_mask_defaults=auto_mask_defaults or None,
            **kwargs
        )

    def run(self, input_path: str | Path, output_path: str | Path, **kwargs: Any) -> None:
        """Process video with optional overrides."""
        from .pipeline import process_video  # Import here to avoid import-time deps

        args: dict[str, Any] = dict(kwargs)
        auto_mask_kwargs = args.pop("auto_mask_kwargs", None)
        args.pop("auto_mask_defaults", None)
        args.pop("codec", None)
        args.pop("audio_codec", None)
        args.pop("bitrate", None)
        args.pop("preserve_audio", None)
        args.pop("reuse_mask", None)

        dilate_value = args.pop("dilate", None)
        if dilate_value is None:
            if auto_mask_kwargs and "dilate_iterations" in auto_mask_kwargs:
                dilate_value = auto_mask_kwargs["dilate_iterations"]
            elif self.auto_mask_defaults and "dilate_iterations" in self.auto_mask_defaults:
                dilate_value = self.auto_mask_defaults["dilate_iterations"]
            else:
                dilate_value = self.dilate

        args.setdefault("method", self.method)
        args.setdefault("mask_mode", self.mask_mode)
        args.setdefault("dilate", dilate_value)
        args.setdefault("seed", self.seed)
        args.setdefault("window", self.window)
        args.setdefault("overlap", self.overlap)
        args.setdefault("qc", self.qc)
        args.setdefault("retry", self.retry)
        if self.extra:
            args.update(self.extra)
        process_video(Path(input_path), Path(output_path), **args)

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        mask_path: Optional[str | Path] = None,
        **kwargs: Any
    ) -> Path:
        """Process video file (UI-compatible signature)."""
        self.run(input_path, output_path, **kwargs)
        return Path(output_path)
