"""Optical flow helpers with RAFT stubs and OpenCV fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class FlowConfig:
    backend: str = "auto"  # auto|raft|tvl1|farneback
    raft_weights: Optional[Path] = None
    device: str = "cuda"


class FlowEstimator:
    """Estimate optical flow using RAFT when available or fall back to OpenCV methods."""

    def __init__(self, cfg: FlowConfig | None = None):
        self.cfg = cfg or FlowConfig()
        self._raft = None
        if self.cfg.backend in ("auto", "raft"):
            try:
                from .raft_stub import RAFT

                weights = self.cfg.raft_weights or (Path.home() / ".wmr" / "models" / "raft.pth")
                if weights.exists():
                    self._raft = RAFT.load_from_checkpoint(weights, device=self.cfg.device)
                elif self.cfg.backend == "raft":
                    raise FileNotFoundError(f"RAFT weights not found at {weights}")
            except Exception:
                self._raft = None

    @staticmethod
    def _to_gray_u8(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    def flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        if self._raft is not None:
            return self._raft.flow(prev, curr)
        if self.cfg.backend in ("tvl1",):
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()  # type: ignore[attr-defined]
            return tvl1.calc(self._to_gray_u8(prev), self._to_gray_u8(curr), None)
        return cv2.calcOpticalFlowFarneback(
            self._to_gray_u8(prev),
            self._to_gray_u8(curr),
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

    @staticmethod
    def warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
