"""Core processing package for the Watermark Remover Suite."""

from . import utils
from .batch_manager import BatchItem, BatchResult, BatchWatermarkProcessor
from .image_remover import ImageWatermarkRemover
from .video_remover import VideoWatermarkRemover

__all__ = [
    "ImageWatermarkRemover",
    "VideoWatermarkRemover",
    "BatchWatermarkProcessor",
    "BatchItem",
    "BatchResult",
    "utils",
]
