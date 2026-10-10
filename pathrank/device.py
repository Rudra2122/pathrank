"""Device selection. "auto" = CUDA when available, else CPU.

MPS (Apple GPU) is opt-in for the two-tower model only (device: mps); LightGCN needs sparse
matmul, which MPS doesn't support, so it falls back to CPU there.
"""
from __future__ import annotations

import torch

from .utils import log


def resolve(name: str = "auto", sparse: bool = False) -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        log.warning("device=cuda requested but CUDA is unavailable; using CPU")
        name = "cpu"
    if name == "mps":
        if sparse or not torch.backends.mps.is_available():
            log.warning("MPS unavailable for this model; using CPU")
            name = "cpu"
    return torch.device(name)


def describe(dev: torch.device) -> str:
    if dev.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(dev)}"
    return dev.type


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
