"""Temporal helpers for chunked video processing and seam blending."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .flow import FlowEstimator


def make_chunks(n_frames: int, window: int, overlap: int) -> List[Tuple[int, int]]:
    assert window > overlap >= 0
    chunks: List[Tuple[int, int]] = []
    start = 0
    while start < n_frames:
        end = min(n_frames - 1, start + window - 1)
        chunks.append((start, end))
        if end == n_frames - 1:
            break
        start = end - overlap + 1
    return chunks


def blend_overlap(prev_clean: np.ndarray, curr_clean: np.ndarray, flow_prev_to_curr: np.ndarray, alpha: float) -> np.ndarray:
    warped_prev = FlowEstimator.warp(prev_clean, flow_prev_to_curr)
    return (
        warped_prev.astype(np.float32) * alpha + curr_clean.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
