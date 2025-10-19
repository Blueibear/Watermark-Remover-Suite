"""Minimal pipeline implementation for the Watermark Remover Suite MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
    tqdm = lambda x, **_: x  # type: ignore

try:
    from watermark_remover.core.inpaint_lama import inpaint_lama as lama_run

    _LAMA_AVAILABLE = True
except Exception:  # pragma: no cover - LaMa optional
    lama_run = None  # type: ignore
    _LAMA_AVAILABLE = False

from .flow import FlowEstimator
from .temporal import blend_overlap, make_chunks

Method = Literal["telea", "lama", "sd", "noop"]

_LAMA_MODEL = Path.home() / ".wmr" / "models" / "lama.onnx"
_LAMA_CACHE = None


def _imread(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _auto_mask_bottom_right(img: np.ndarray, *, dilate: int = 5) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = int(h * 0.75)
    x0 = int(w * 0.65)
    roi = img[y0:, x0:].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    threshold = max(180, int(gray.mean() + 0.6 * gray.std()))
    _, mask_roi = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 50, 150)
    mask_roi = cv2.bitwise_or(mask_roi, edges)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:, x0:] = mask_roi
    if dilate > 0:
        kernel_size = max(1, dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)
    return mask


def _inpaint(img: np.ndarray, mask: np.ndarray, method: Method, seed: int) -> np.ndarray:
    if method == "noop":
        return img
    if method == "telea":
        return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    if method == "lama":
        if not _LAMA_AVAILABLE or lama_run is None:
            raise RuntimeError(
                "LaMa backend unavailable. Install onnxruntime and place lama.onnx under ~/.wmr/models/."
            )
        global _LAMA_CACHE
        if _LAMA_CACHE is None:
            if not _LAMA_MODEL.exists():
                raise FileNotFoundError(
                    f"LaMa ONNX not found at {_LAMA_MODEL}. See models/README.md for instructions."
                )
            _LAMA_CACHE = _LAMA_MODEL
        return lama_run(img, mask, model_path=_LAMA_CACHE, device="auto")
    if method == "sd":
        raise NotImplementedError("Stable Diffusion inpainting not available in MVP stub.")
    raise ValueError(f"Unsupported method: {method}")


def process_image(
    path_in: Path,
    path_out: Path,
    *,
    method: Method,
    mask_mode: str,
    dilate: int,
    seed: int,
) -> None:
    img = _imread(path_in)
    if mask_mode == "auto":
        mask = _auto_mask_bottom_right(img, dilate=dilate)
    else:
        mask = _auto_mask_bottom_right(img, dilate=dilate)
    result = _inpaint(img, mask, method, seed)
    _imwrite(path_out, result)


def process_video(
    path_in: Path,
    path_out: Path,
    *,
    method: Method,
    mask_mode: str,
    dilate: int,
    seed: int,
    window: int,
    overlap: int,
) -> None:
    cap = cv2.VideoCapture(str(path_in))
    if not cap.isOpened():
        raise FileNotFoundError(path_in)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if total == 0:
        total = len(frames)
    writer = cv2.VideoWriter(str(path_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    masks = [
        _auto_mask_bottom_right(frame, dilate=dilate) if mask_mode == "auto" else _auto_mask_bottom_right(frame, dilate=dilate)
        for frame in frames
    ]

    flow_estimator = FlowEstimator()
    chunks = make_chunks(len(frames), window, overlap)
    prev_chunk_clean = None

    for chunk_index, (start, end) in enumerate(tqdm(chunks, desc="wmr-video")):
        current_clean = []
        for idx in range(start, end + 1):
            current_clean.append(_inpaint(frames[idx], masks[idx], method, seed))

        if chunk_index == 0:
            last_keep = end - overlap
            for frame in current_clean[: max(0, last_keep - start + 1)]:
                writer.write(frame)
        else:
            for j in range(overlap):
                prev_idx = len(prev_chunk_clean) - overlap + j
                prev_frame = prev_chunk_clean[prev_idx]
                curr_frame = current_clean[j]
                flow = flow_estimator.flow(frames[start - overlap + j], frames[start + j])
                alpha = 1.0 - (j / max(1, overlap - 1))
                blended = blend_overlap(prev_frame, curr_frame, flow, alpha)
                writer.write(blended)
            mid_start = overlap
            mid_end = (end - start + 1) - overlap
            for frame in current_clean[mid_start:mid_end]:
                writer.write(frame)

        prev_chunk_clean = current_clean

    if prev_chunk_clean is not None:
        tail = prev_chunk_clean[-overlap:] if overlap > 0 else prev_chunk_clean
        for frame in tail:
            writer.write(frame)

    writer.release()
