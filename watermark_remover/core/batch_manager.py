"""Batch processing helpers for watermark removal."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _resolve_auto_mask_kwargs(
        self, media_type: MediaType, overrides: Optional[Dict]
    ) -> Optional[Dict]:
        if overrides:
            return overrides
        if media_type == "image":
            return dict(self.image_remover.auto_mask_defaults)
        if media_type == "video":
            return dict(self.video_remover.auto_mask_defaults)
        return overrides

    def _process_image(
        self, item: BatchItem, auto_mask_kwargs: Optional[Dict]
    ) -> Tuple[Path, Path]:
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

    def _execute_item(self, item: BatchItem) -> BatchResult:
        input_path = Path(item.input_path)
        media_type = item.media_type.lower()
        logger.info("Batch processing %s", input_path)
        auto_kwargs = self._resolve_auto_mask_kwargs(media_type, item.auto_mask_kwargs)
        try:
            if media_type == "image":
                output_path, mask_path = self._process_image(item, auto_kwargs)
                return BatchResult(
                    success=True,
                    media_type=media_type,
                    input_path=input_path,
                    output_path=output_path,
                    mask_path=mask_path,
                )
            if media_type == "video":
                output_path = self._process_video(item, auto_kwargs)
                return BatchResult(
                    success=True,
                    media_type=media_type,
                    input_path=input_path,
                    output_path=output_path,
                )
            raise ValueError(f"Unsupported media type: {item.media_type}")
        except Exception as exc:  # pragma: no cover - error path
            logger.exception("Failed to process %s: %s", input_path, exc)
            return BatchResult(
                success=False,
                media_type=media_type,
                input_path=input_path,
                error=str(exc),
            )

    def process(self, items: Iterable[BatchItem]) -> List[BatchResult]:
        item_list = list(items)
        if not item_list:
            return []

        order_map = {Path(item.input_path): idx for idx, item in enumerate(item_list)}

        if self.max_workers <= 1 or self.halt_on_error:
            results: List[BatchResult] = []
            for item in item_list:
                result = self._execute_item(item)
                results.append(result)
                if self.halt_on_error and not result.success:
                    break
            return results

        results: List[BatchResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {executor.submit(self._execute_item, item): item for item in item_list}
            for future in as_completed(future_to_item):
                result = future.result()
                results.append(result)
                if self.halt_on_error and not result.success:
                    executor.shutdown(cancel_futures=True)
                    break
        results.sort(key=lambda r: order_map.get(r.input_path, 0))
        return results


__all__ = ["BatchItem", "BatchResult", "BatchWatermarkProcessor"]
