"""Batch processing wrapper - safe to import in tests."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .image_remover import ImageWatermarkRemover
from .video_remover import VideoWatermarkRemover


@dataclass
class BatchResult:
    """Result from batch processing a single item."""
    success: bool
    media_type: str
    input_path: Path
    output_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    error: Optional[str] = None


class BatchWatermarkProcessor:
    """Coordinate watermark removal jobs across multiple media items."""

    def __init__(
        self,
        image_remover: Optional[ImageWatermarkRemover] = None,
        video_remover: Optional[VideoWatermarkRemover] = None,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.image_remover = image_remover or ImageWatermarkRemover()
        self.video_remover = video_remover or VideoWatermarkRemover()
        self.config = dict(config or {})

    def process_batch(self, items: list[dict[str, Any]]) -> list[BatchResult]:
        """Process a batch of items (placeholder for MVP)."""
        results = []
        for item in items:
            try:
                media_type = item.get("media_type", "image")
                input_path = Path(item["input_path"])
                output_path = Path(item["output_path"])

                if media_type == "image":
                    self.image_remover.process_file(input_path, output_path)
                elif media_type == "video":
                    self.video_remover.process_file(input_path, output_path)
                else:
                    raise ValueError(f"Unsupported media type: {media_type}")

                results.append(BatchResult(
                    success=True,
                    media_type=media_type,
                    input_path=input_path,
                    output_path=output_path,
                ))
            except Exception as e:
                results.append(BatchResult(
                    success=False,
                    media_type=item.get("media_type", "unknown"),
                    input_path=Path(item.get("input_path", "")),
                    error=str(e),
                ))
        return results
