"""Input / output normalization — fit on the training set, transform everything else.

Single shared implementation used by both the XGBoost (gbt) and neural_surrogate
pipelines. numpy-only (no torch), so safe to import anywhere.
"""

from __future__ import annotations

import numpy as np
import joblib
from typing import Optional


class StandardScaler:
    """Standardise to zero mean, unit variance (z-score)."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        # Avoid division by zero for constant columns
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def save(self, path: str):
        joblib.dump({"mean": self.mean_, "std": self.std_}, path)

    def load(self, path: str):
        d = joblib.load(path)
        self.mean_ = d["mean"]
        self.std_ = d["std"]
        return self


class MinMaxScaler:
    """Scale per-column to the [0, 1] range."""

    def __init__(self):
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        x = np.asarray(x, dtype=np.float64)
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        span = self.max_ - self.min_
        # Avoid division by zero for constant columns
        self.max_ = np.where(span < 1e-8, self.min_ + 1.0, self.max_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return (x - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.max_ - self.min_) + self.min_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def save(self, path: str):
        joblib.dump({"min": self.min_, "max": self.max_}, path)

    def load(self, path: str):
        d = joblib.load(path)
        self.min_ = d["min"]
        self.max_ = d["max"]
        return self
