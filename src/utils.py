"""
Utility helpers for Dimensional ABSA 2026.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, map_location: str | torch.device = "cpu") -> None:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)


def pearson_corr(preds: Iterable[float], targets: Iterable[float]) -> float:
    """
    Compute Pearson correlation; returns 0.0 if undefined (e.g., constant).
    """
    try:
        coef, _ = pearsonr(list(preds), list(targets))
        if np.isnan(coef):
            return 0.0
        return float(coef)
    except Exception:
        return 0.0


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def num_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable



