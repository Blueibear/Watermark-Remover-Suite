"""Core processing utilities for the Watermark Remover Suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy imports to avoid heavy work at import time
if TYPE_CHECKING:
    from .batch_manager import BatchItem, BatchResult, BatchWatermarkProcessor
    from .image_remover import ImageWatermarkRemover
    from .pipeline import process_image, process_video
    from .video_remover import VideoWatermarkRemover

__all__ = [
    "process_image",
    "process_video",
    "ImageWatermarkRemover",
    "VideoWatermarkRemover",
    "BatchItem",
    "BatchResult",
    "BatchWatermarkProcessor",
]


def __getattr__(name: str):
    """Lazy-load heavy modules on attribute access."""
    if name == "process_image":
        from .pipeline import process_image
        return process_image
    elif name == "process_video":
        from .pipeline import process_video
        return process_video
    elif name == "ImageWatermarkRemover":
        from .image_remover import ImageWatermarkRemover
        return ImageWatermarkRemover
    elif name == "VideoWatermarkRemover":
        from .video_remover import VideoWatermarkRemover
        return VideoWatermarkRemover
    elif name in {"BatchItem", "BatchResult", "BatchWatermarkProcessor"}:
        from .batch_manager import BatchItem, BatchResult, BatchWatermarkProcessor

        mapping = {
            "BatchItem": BatchItem,
            "BatchResult": BatchResult,
            "BatchWatermarkProcessor": BatchWatermarkProcessor,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
