"""Core processing package for the Watermark Remover Suite."""

from .image_remover import ImageWatermarkRemover
from .video_remover import VideoWatermarkRemover
from .batch_manager import BatchItem, BatchResult, BatchWatermarkProcessor
from . import utils

__all__ = [
    "ImageWatermarkRemover",
    "VideoWatermarkRemover",
    "BatchWatermarkProcessor",
    "BatchItem",
    "BatchResult",
    "utils",
]
