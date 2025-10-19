"""RAFT placeholder used until a real model is integrated."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RAFT:
    device: str = "cuda"

    @classmethod
    def load_from_checkpoint(cls, ckpt: Path, device: str = "cuda") -> "RAFT":
        # Replace with the actual RAFT loading logic when integrating the model.
        return cls(device=device)

    def flow(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        raise RuntimeError("RAFT backend not implemented; please add real weights and inference code.")
