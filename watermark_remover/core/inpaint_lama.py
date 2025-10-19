"""LaMa ONNX runner used by the MVP pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None  # type: ignore


class LaMaONNX:
    """Minimal LaMa ONNX wrapper supporting tiled inference."""

    def __init__(self, onnx_path: Path, device: str = "auto"):
        if ort is None:
            raise RuntimeError("onnxruntime is required for LaMa inference. Install onnxruntime-gpu or onnxruntime.")
        providers: list[object] = [
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kNextPowerOfTwo"}),
            "CPUExecutionProvider",
        ]
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.tile = 512

    @staticmethod
    def _prepare(img_bgr: np.ndarray, mask_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if mask_u8.ndim == 3:
            mask = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
        else:
            mask = mask_u8
        mask = (mask > 0).astype(np.float32)
        return img_rgb, mask

    def _run(self, img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        inp_img = np.transpose(img_rgb, (2, 0, 1))[None, ...].astype(np.float32)
        inp_mask = mask[None, None, ...].astype(np.float32)
        feeds = {self.input_names[0]: inp_img, self.input_names[1]: inp_mask}
        out = self.session.run(self.output_names, feeds)[0]
        out_rgb = np.transpose(out[0], (1, 2, 0))
        out_bgr = cv2.cvtColor(np.clip(out_rgb, 0, 1), cv2.COLOR_RGB2BGR)
        return (out_bgr * 255.0 + 0.5).astype(np.uint8)

    def inpaint(self, img_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        height, width = img_bgr.shape[:2]
        if max(height, width) <= self.tile:
            prepared = self._prepare(img_bgr, mask_u8)
            return self._run(*prepared)

        overlap = 32
        tile = self.tile
        output = np.zeros_like(img_bgr, dtype=np.float32)
        weight = np.zeros((height, width, 1), dtype=np.float32)

        for y in range(0, height, tile - overlap):
            for x in range(0, width, tile - overlap):
                y1 = min(y + tile, height)
                x1 = min(x + tile, width)
                img_patch = img_bgr[y:y1, x:x1]
                mask_patch = mask_u8[y:y1, x:x1]
                prepared = self._prepare(img_patch, mask_patch)
                patch = self._run(*prepared).astype(np.float32)

                wy = np.linspace(0, 1, y1 - y)[:, None]
                wx = np.linspace(0, 1, x1 - x)[None, :]
                w_patch = (wy * wx)[..., None].astype(np.float32)
                current = output[y:y1, x:x1]
                current_weight = weight[y:y1, x:x1]
                output[y:y1, x:x1] = (current * current_weight + patch * w_patch) / np.clip(
                    current_weight + w_patch, 1e-6, None
                )
                weight[y:y1, x:x1] = current_weight + w_patch

        return np.clip(output, 0, 255).astype(np.uint8)


def inpaint_lama(img_bgr: np.ndarray, mask_u8: np.ndarray, *, model_path: Path, device: str = "auto") -> np.ndarray:
    runner = LaMaONNX(model_path, device=device)
    return runner.inpaint(img_bgr, mask_u8)
