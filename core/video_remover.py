"""Video watermark removal built on MoviePy frame processing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import cv2
import numpy as np
from moviepy import VideoFileClip

from .image_remover import ImageWatermarkRemover

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class VideoWatermarkRemover:
    """Apply the image watermark remover to every frame in a video clip."""

    def __init__(
        self,
        image_remover: Optional[ImageWatermarkRemover] = None,
        *,
        reuse_mask: bool = True,
        preserve_audio: bool = True,
        codec: str = "libx264",
        audio_codec: str = "aac",
        bitrate: Optional[str] = None,
        auto_mask_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image_remover = image_remover or ImageWatermarkRemover()
        self.reuse_mask = reuse_mask
        self.preserve_audio = preserve_audio
        self.codec = codec
        self.audio_codec = audio_codec
        self.bitrate = bitrate
        self.auto_mask_defaults = dict(auto_mask_defaults or {})
        logger.debug(
            "Initialized VideoWatermarkRemover (reuse_mask=%s, preserve_audio=%s, codec=%s)",
            self.reuse_mask,
            self.preserve_audio,
            self.codec,
        )

    def process_file(
        self,
        input_path: PathLike,
        output_path: PathLike,
        *,
        mask_path: Optional[PathLike] = None,
        auto_mask_kwargs: Optional[dict] = None,
        codec: Optional[str] = None,
        audio_codec: Optional[str] = None,
        bitrate: Optional[str] = None,
    ) -> Path:
        """Remove watermark from a video file."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Video not found: {input_path}")

        logger.info("Processing video %s", input_path)

        detector_kwargs: Dict[str, Any] = dict(self.auto_mask_defaults)
        if auto_mask_kwargs:
            detector_kwargs.update(auto_mask_kwargs)
        clip = VideoFileClip(str(input_path))
        processed_clip = None
        try:
            fps = clip.fps or 24
            original_frame_function = clip.frame_function
            mask_cache: Optional[np.ndarray] = None

            def frame_function(t: float) -> np.ndarray:
                nonlocal mask_cache
                frame_rgb = original_frame_function(t)
                bgr_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if self.reuse_mask and mask_cache is not None:
                    mask_arg: Optional[Union[np.ndarray, PathLike]] = mask_cache
                    inpaint_kwargs = None
                else:
                    mask_arg = mask_path
                    inpaint_kwargs = detector_kwargs if mask_arg is None else None

                result, mask_used = self.image_remover.remove_watermark(
                    bgr_frame,
                    mask=mask_arg,
                    auto_mask_kwargs=inpaint_kwargs,
                )

                if self.reuse_mask and mask_cache is None:
                    mask_cache = mask_used.copy()

                return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            processed_clip = clip.with_updated_frame_function(frame_function)

            audio_flag = False
            if self.preserve_audio and clip.audio is not None:
                processed_clip = processed_clip.with_audio(clip.audio)
                audio_flag = True

            processed_clip.write_videofile(
                str(output_path),
                codec=codec or self.codec,
                audio=audio_flag,
                audio_codec=(audio_codec or self.audio_codec) if audio_flag else None,
                bitrate=self.bitrate if bitrate is None else bitrate,
                fps=fps,
                logger=None,
            )
        finally:
            clip.close()
            if processed_clip is not None:
                processed_clip.close()

        logger.debug("Video written to %s", output_path)
        return output_path

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        image_remover: Optional[ImageWatermarkRemover] = None,
    ) -> "VideoWatermarkRemover":
        video_settings = dict(config.get("video_processing", {}))
        auto_mask_defaults = dict(video_settings.get("auto_mask", {}))
        if image_remover is None:
            image_remover = ImageWatermarkRemover.from_config(config)
        return cls(
            image_remover=image_remover,
            reuse_mask=bool(video_settings.get("reuse_mask", True)),
            preserve_audio=bool(video_settings.get("preserve_audio", True)),
            codec=video_settings.get("codec", "libx264"),
            audio_codec=video_settings.get("audio_codec", "aac"),
            bitrate=video_settings.get("bitrate"),
            auto_mask_defaults=auto_mask_defaults,
        )


__all__ = ["VideoWatermarkRemover"]
