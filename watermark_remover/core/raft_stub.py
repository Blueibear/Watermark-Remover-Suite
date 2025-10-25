"""RAFT TorchScript loader for optical flow estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class RAFT:
    device: str = "cuda"
    model: torch.jit.ScriptModule | None = None

    @classmethod
    def load_from_checkpoint(cls, ckpt: Path, device: str = "cuda") -> "RAFT":
        """Load RAFT model from TorchScript checkpoint."""
        m = torch.jit.load(str(ckpt), map_location=device)
        m.eval()
        return cls(device=device, model=m)

    @staticmethod
    def _to_tensor_bgr(img: np.ndarray) -> torch.Tensor:
        """Convert BGR image to normalized tensor."""
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        t = torch.from_numpy(img).permute(2, 0, 1).float()
        return t[None] / 255.0

    def flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Compute optical flow from prev to curr frame."""
        assert self.model is not None, "RAFT model not loaded"
        with torch.no_grad():
            i0 = self._to_tensor_bgr(prev).to(self.device)
            i1 = self._to_tensor_bgr(curr).to(self.device)
            f = self.model(i0, i1)[0].permute(1, 2, 0).detach().cpu().numpy()
        return f
