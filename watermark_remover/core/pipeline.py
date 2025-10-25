"""Minimal pipeline implementation for the Watermark Remover Suite MVP."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import tempfile
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

try:
    from watermark_remover.qc.quickcheck import parse_qc, qc_pass

    _QC_AVAILABLE = True
except Exception:  # pragma: no cover - QC optional
    parse_qc = None  # type: ignore
    qc_pass = None  # type: ignore
    _QC_AVAILABLE = False

Method = Literal["telea", "lama", "sd", "noop"]

_LAMA_MODEL = Path.home() / ".wmr" / "models" / "lama.onnx"
_LAMA_CACHE = None


def _frame_seed(base: int, idx: int) -> int:
    """Generate deterministic per-frame seed from base seed and frame index."""
    h = hashlib.blake2b(f"{base}:{idx}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & 0x7FFFFFFF


def _remux_with_ffmpeg(src_mp4: str, dst_mp4: str) -> None:
    """Remux video with ffmpeg to preserve color metadata."""
    cmd = f'ffmpeg -y -i "{src_mp4}" -c:v libx264 -crf 16 -preset medium -pix_fmt yuv420p -movflags +faststart "{dst_mp4}"'
    subprocess.run(shlex.split(cmd), check=True)


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
    qc: str = "warped_ssim>=0.92",
    retry: int = 1,
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

    # Use temporary output for ffmpeg remux
    tmp_out = str(Path(path_out).with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Generate masks and optionally dump them
    masks = []
    for i, frame in enumerate(frames):
        mask = _auto_mask_bottom_right(frame, dilate=dilate) if mask_mode == "auto" else _auto_mask_bottom_right(frame, dilate=dilate)
        masks.append(mask)
        # Optional mask dumping
        if os.environ.get("WMR_DUMP_MASKS") == "1":
            Path("out/masks").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"out/masks/{i:06d}.png", mask)

    # Parse QC threshold
    thr = parse_qc(qc) if _QC_AVAILABLE and parse_qc else None

    flow_estimator = FlowEstimator()
    chunks = make_chunks(len(frames), window, overlap)
    prev_chunk_clean = None

    for chunk_index, (start, end) in enumerate(tqdm(chunks, desc="wmr-video")):
        current_clean = []
        prev_clean = None

        for idx in range(start, end + 1):
            # Use deterministic per-frame seed
            seed_i = _frame_seed(seed, idx)
            cleaned = _inpaint(frames[idx], masks[idx], method, seed_i)

            # QC + retry (skip first frame in a chunk and when QC disabled)
            if thr is not None and _QC_AVAILABLE and qc_pass and idx > start and prev_clean is not None:
                ok = qc_pass(frames[idx - 1], frames[idx], masks[idx], prev_clean, cleaned, thr=thr)
                tries = 0
                while not ok and tries < retry:
                    bigger = min(11, dilate + 2 + tries)
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bigger, bigger))
                    retry_mask = cv2.dilate(masks[idx], k)
                    cleaned = _inpaint(frames[idx], retry_mask, method, _frame_seed(seed, idx) ^ (0x9E3779B1 * (tries + 1)))
                    ok = qc_pass(frames[idx - 1], frames[idx], retry_mask, prev_clean, cleaned, thr=thr)
                    tries += 1

            current_clean.append(cleaned)
            prev_clean = cleaned

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

    # Remux with ffmpeg to preserve color metadata
    try:
        _remux_with_ffmpeg(tmp_out, str(path_out))
        os.remove(tmp_out)
    except Exception:
        # If ffmpeg fails, keep the temp file as the output
        try:
            os.rename(tmp_out, str(path_out))
        except OSError:
            pass
