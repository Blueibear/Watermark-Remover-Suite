"""Thin video wrapper for MVP pipeline - safe to import in tests."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

# Lazy import to avoid heavy work at import time
if TYPE_CHECKING:
    from .pipeline import process_video

@dataclass
class VideoWatermarkRemover:
    """Thin wrapper around MVP process_video for test compatibility."""

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

    @classmethod
    def from_config(cls, config: dict, **kwargs: Any) -> "VideoWatermarkRemover":
        """Create from config dict (for UI compatibility)."""
        # Extract relevant fields from config, using defaults
        video_cfg = config.get("video", {})
        return cls(
            method=video_cfg.get("method", "telea"),
            mask_mode=video_cfg.get("mask_mode", "auto"),
            dilate=video_cfg.get("dilate", 5),
            seed=video_cfg.get("seed", 1234),
            window=video_cfg.get("window", 48),
            overlap=video_cfg.get("overlap", 12),
            qc=video_cfg.get("qc", "warped_ssim>=0.92"),
            retry=video_cfg.get("retry", 1),
            **kwargs
        )

    def run(self, input_path: str | Path, output_path: str | Path, **kwargs: Any) -> None:
        """Process video with optional overrides."""
        from .pipeline import process_video  # Import here to avoid import-time deps

        args: dict[str, Any] = dict(kwargs)
        args.setdefault("method", self.method)
        args.setdefault("mask_mode", self.mask_mode)
        args.setdefault("dilate", self.dilate)
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
