"""Optical flow helpers with RAFT stubs and OpenCV fallbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


@dataclass
class FlowConfig:
    backend: str = "auto"  # auto|raft|tvl1|farneback
    raft_weights: Optional[Path] = None
    device: str = "cuda"


def _cfg():
    """Load YAML config for RAFT model."""
    p = Path("watermark_remover/models/model_config.yaml")
    return yaml.safe_load(open(p, "r")) if p.exists() else {}


class FlowEstimator:
    """Estimate optical flow using RAFT when available or fall back to OpenCV methods."""

    def __init__(self, cfg: FlowConfig | None = None):
        self.cfg = cfg or FlowConfig()
        self._raft = None
        if self.cfg.backend in ("auto", "raft"):
            try:
                from .raft_stub import RAFT

                r = _cfg().get("raft", {})
                wpath = self.cfg.raft_weights or Path(os.path.expanduser(r.get("weights_path", "~/.wmr/models/raft-kitti.torchscript.pt")))
                device = r.get("device", self.cfg.device)
                if wpath.exists():
                    self._raft = RAFT.load_from_checkpoint(wpath, device=device)
                elif self.cfg.backend == "raft":
                    raise FileNotFoundError(f"RAFT weights not found at {wpath}")
            except Exception:
                self._raft = None

    @staticmethod
    def _to_gray_u8(img: np.ndarray) -> np.ndarray:
        """Convert image to grayscale uint8."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return g if g.dtype == np.uint8 else np.clip(g, 0, 255).astype(np.uint8)

    def flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute optical flow from prev to curr frame."""
        if self._raft is not None:
            return self._raft.flow(prev, curr)
        try:
            tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()  # type: ignore[attr-defined]
            return tvl1.calc(self._to_gray_u8(prev), self._to_gray_u8(curr), None)
        except Exception:
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
