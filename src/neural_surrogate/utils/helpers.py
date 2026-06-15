"""Utility functions for the surrogate pipeline."""

from __future__ import annotations

import numpy as np
import torch
import random
from typing import Optional


def set_seed(seed: int = 42):
    """Make results reproducible across numpy, torch, and python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quick_predict(
    model: torch.nn.Module,
    continuous: np.ndarray,
    categorical: np.ndarray,
    input_scaler=None,
    target_scaler=None,
    device: str = "cpu",
) -> np.ndarray:
    """One-shot prediction for a batch of inputs (returns real-scale targets)."""
    model.eval()
    if input_scaler:
        continuous = input_scaler.transform(continuous)
    cont_t = torch.from_numpy(continuous.astype(np.float32)).to(device)
    cat_t = torch.from_numpy(categorical.astype(np.int64)).to(device)
    with torch.no_grad():
        pred = model(cont_t, cat_t).cpu().numpy()
    if target_scaler:
        pred = target_scaler.inverse_transform(pred)
    return pred
