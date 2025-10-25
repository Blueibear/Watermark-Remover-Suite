"""Quick quality check helpers for per-frame validation."""

from __future__ import annotations

import re

import numpy as np

from .metrics import masked_ssim_warped as wssim


def parse_qc(expr: str) -> float | None:
    """Parse QC expression like 'warped_ssim>=0.92' or 'none'."""
    if not expr or expr.strip().lower() == "none":
        return None
    m = re.match(r"\s*warped_ssim\s*>=\s*([0-9.]+)\s*", expr, flags=re.I)
    return float(m.group(1)) if m else None


def qc_pass(prev_orig: np.ndarray, curr_orig: np.ndarray, mask: np.ndarray, prev_clean: np.ndarray, curr_clean: np.ndarray, thr: float) -> bool:
    """Check if cleaned frame passes QC threshold for temporal stability."""
    # Stability inside edited region, using original motion
    val = wssim(prev_clean, curr_clean, mask)
    return val >= thr
