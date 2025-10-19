"""Quality control helpers for reproducible evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

try:
    import lpips  # type: ignore

    _LPIPS_MODEL = lpips.LPIPS(net="alex")
except Exception:  # pragma: no cover - optional dependency
    lpips = None  # type: ignore
    _LPIPS_MODEL = None


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img, 0, 255)
    return (img * 255.0).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)


def _warped(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(_to_gray(prev), _to_gray(curr), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(prev, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def masked_ssim(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    mask = (mask > 0).astype(np.uint8)
    ref = _to_gray(_ensure_uint8(reference))
    cand = _to_gray(_ensure_uint8(candidate))
    score = ssim(ref, cand, data_range=255, gaussian_weights=True)
    return float(score * (mask.mean() / 255.0 if mask.max() else 1.0))


def lpips_metric(reference: np.ndarray, candidate: np.ndarray) -> Optional[float]:
    if _LPIPS_MODEL is None:
        return None
    ref_tensor = _tensor_from_image(reference)
    cand_tensor = _tensor_from_image(candidate)
    score = _LPIPS_MODEL(ref_tensor, cand_tensor)
    return float(score.detach().cpu().numpy())


def _tensor_from_image(img: np.ndarray):
    import torch

    arr = cv2.cvtColor(_ensure_uint8(img), cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


@dataclass
class MetricReport:
    warped_ssim: Optional[float]
    lpips_value: Optional[float]

    def to_json(self) -> str:
        return json.dumps({"warped_ssim": self.warped_ssim, "lpips": self.lpips_value}, indent=2)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
