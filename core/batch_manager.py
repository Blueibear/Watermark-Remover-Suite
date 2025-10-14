"""Batch processing helpers for watermark removal."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .image_remover import ImageWatermarkRemover
from .video_remover import VideoWatermarkRemover

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
MediaType = str


@dataclass
class BatchItem:
    media_type: MediaType
    input_path: PathLike
    output_path: PathLike
    mask_path: Optional[PathLike] = None
    auto_mask_kwargs: Optional[Dict] = None


@dataclass
class BatchResult:
    success: bool
    media_type: MediaType
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
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        config_map = dict(config or {})
        batch_settings = dict(config_map.get("batch", {}))
        self.halt_on_error = bool(batch_settings.get("halt_on_error", False))
        self.max_workers = int(batch_settings.get("max_workers", 1))

        if image_remover is None:
            if config_map:
                image_remover = ImageWatermarkRemover.from_config(config_map)
            else:
                image_remover = ImageWatermarkRemover()
        self.image_remover = image_remover

        if video_remover is None:
            if config_map:
                video_remover = VideoWatermarkRemover.from_config(
                    config_map, image_remover=self.image_remover
                )
            else:
                video_remover = VideoWatermarkRemover(image_remover=self.image_remover)
        self.video_remover = video_remover

    def _resolve_auto_mask_kwargs(self, media_type: MediaType, overrides: Optional[Dict]) -> Optional[Dict]:
        if overrides:
            return overrides
        if media_type == "image":
            return dict(self.image_remover.auto_mask_defaults)
        if media_type == "video":
            return dict(self.video_remover.auto_mask_defaults)
        return overrides

    def _process_image(self, item: BatchItem, auto_mask_kwargs: Optional[Dict]) -> Tuple[Path, Path]:
        return self.image_remover.process_file(
            item.input_path,
            item.output_path,
            mask_path=item.mask_path,
            auto_mask_kwargs=auto_mask_kwargs,
        )

    def _process_video(self, item: BatchItem, auto_mask_kwargs: Optional[Dict]) -> Path:
        return self.video_remover.process_file(
            item.input_path,
            item.output_path,
            mask_path=item.mask_path,
            auto_mask_kwargs=auto_mask_kwargs,
        )

    def process(self, items: Iterable[BatchItem]) -> List[BatchResult]:
        results: List[BatchResult] = []
        for item in items:
            input_path = Path(item.input_path)
            media_type = item.media_type.lower()
            logger.info("Batch processing %s", input_path)
            try:
                auto_kwargs = self._resolve_auto_mask_kwargs(media_type, item.auto_mask_kwargs)
                if media_type == "image":
                    output_path, mask_path = self._process_image(item, auto_kwargs)
                    results.append(
                        BatchResult(
                            success=True,
                            media_type=media_type,
                            input_path=input_path,
                            output_path=output_path,
                            mask_path=mask_path,
                        )
                    )
                elif media_type == "video":
                    output_path = self._process_video(item, auto_kwargs)
                    results.append(
                        BatchResult(
                            success=True,
                            media_type=media_type,
                            input_path=input_path,
                            output_path=output_path,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported media type: {item.media_type}")
            except Exception as exc:  # pragma: no cover - error path
                logger.exception("Failed to process %s: %s", input_path, exc)
                results.append(
                    BatchResult(
                        success=False,
                        media_type=media_type,
                        input_path=input_path,
                        error=str(exc),
                    )
                )
                if self.halt_on_error:
                    break
        return results


__all__ = ["BatchItem", "BatchResult", "BatchWatermarkProcessor"]
