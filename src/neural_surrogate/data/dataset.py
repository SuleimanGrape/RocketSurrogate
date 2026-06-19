"""PyTorch Dataset / DataLoader for rocket simulation JSONL data.

Used by both the pure NN trainer and the XGBoost trainer (via make_splits).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "rocket_sim"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "gbt"))
from dataio import load_jsonl  # noqa: F401  (re-exported for callers)
import schema  # noqa: E402

from models.scalers import StandardScaler
from models.surrogate import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    TARGETS,
    ENCODING_MAPS,
)


def _is_positive(rec: Dict) -> bool:
    """A within-bounds (computable) record. Negatives carry within_bounds=False
    and no numeric targets; legacy records without the field are positive.
    Mirrors gbt/data_loader.py so both trainers filter identically."""
    return rec.get("output", {}).get("within_bounds") is not False


def engineered_continuous(records: List[Dict]) -> Tuple[np.ndarray, List[str]]:
    """Compute the SAME engineered continuous features the XGBoost trees use
    (total mass, ratios, Barrowman cg/cp/stability margin, ...) so the two model
    families train on identical information. Reuses gbt/preprocess as the single
    source of truth — never re-derives the formulas here.

    Returns (array of shape (n, n_engineered) float32, engineered feature names).
    """
    import pandas as pd
    from preprocess import add_engineered_features  # gbt — on sys.path above

    raw = pd.DataFrame([r["input"] for r in records])[schema.INPUT_FIELDS]
    full, names = add_engineered_features(raw, list(schema.INPUT_FIELDS))
    new_names = names[len(schema.INPUT_FIELDS):]
    return full[new_names].to_numpy(dtype=np.float32), new_names


def records_to_arrays(
    records: List[Dict],
    continuous_keys: List[str] = CONTINUOUS_FEATURES,
    categorical_keys: List[str] = CATEGORICAL_FEATURES,
    target_keys: List[str] = TARGETS,
    encoding_maps: Dict = ENCODING_MAPS,
    positive_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert JSONL records to (continuous, categorical, targets) arrays.

    Each record is expected to have top-level "input" and "output" keys. By
    default, not-computable (within_bounds=False) records are dropped first —
    they carry no numeric targets, so feeding them to the regressor would KeyError
    (and they aren't regression data). Pass positive_only=False only if the caller
    has already filtered.
    """
    if positive_only:
        records = [r for r in records if _is_positive(r)]
    n = len(records)
    cont = np.zeros((n, len(continuous_keys)), dtype=np.float32)
    cat = np.zeros((n, len(categorical_keys)), dtype=np.int64)
    tgt = np.zeros((n, len(target_keys)), dtype=np.float32)

    for i, rec in enumerate(records):
        inp = rec["input"]
        out = rec["output"]
        for j, k in enumerate(continuous_keys):
            cont[i, j] = float(inp[k])
        for j, k in enumerate(categorical_keys):
            cat[i, j] = int(encoding_maps[k][inp[k]])
        for j, k in enumerate(target_keys):
            tgt[i, j] = float(out[k])

    return cont, cat, tgt


# ---------------------------------------------------------------------------
# Shared split logic (used by PyTorch AND XGBoost trainers)
# ---------------------------------------------------------------------------

def make_splits(
    n: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test index arrays for a dataset of size n.

    Single source of truth for data splitting — use from both the PyTorch
    DataLoader path and the XGBoost training path so results are comparable.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:n_train + n_val + n_test],
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class RocketDataset(Dataset):
    """In-memory dataset: continuous params + categorical codes → flight targets."""

    def __init__(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
        targets: np.ndarray,
        continuous_names: Optional[List[str]] = None,
        log1p_indices: Optional[List[int]] = None,
    ):
        assert len(continuous) == len(categorical) == len(targets)
        self.continuous = continuous.astype(np.float32)
        self.categorical = categorical.astype(np.int64)
        self.targets = targets.astype(np.float32)
        # Names of the continuous columns (base + any engineered). The trainer
        # reads len(continuous_names) to size the model's continuous input.
        self.continuous_names = (
            list(continuous_names) if continuous_names is not None
            else list(CONTINUOUS_FEATURES))
        # Target column indices stored in log1p space (heavy-tailed targets).
        # evaluate() applies expm1 to these before computing metrics.
        self.log1p_indices = list(log1p_indices) if log1p_indices else []

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.continuous[idx]),
            torch.from_numpy(self.categorical[idx]),
            torch.from_numpy(self.targets[idx]),
        )

    # ------------------------------------------------------------------
    # Split + scale + DataLoaders
    # ------------------------------------------------------------------

    @staticmethod
    def from_jsonl(
        path: str,
        continuous_keys=CONTINUOUS_FEATURES,
        categorical_keys=CATEGORICAL_FEATURES,
        target_keys=TARGETS,
        encoding_maps=ENCODING_MAPS,
        engineer_features: bool = True,
    ) -> "RocketDataset":
        """Load a JSONL file produced by the generator into a RocketDataset.

        Not-computable (within_bounds=False) records are dropped. When
        engineer_features is True (default) the same engineered continuous
        features the XGBoost trees use are appended, for an apples-to-apples
        comparison and to give the NN the Barrowman stability signal.
        """
        records = [r for r in load_jsonl(path) if _is_positive(r)]
        cont, cat, tgt = records_to_arrays(
            records, continuous_keys, categorical_keys, target_keys,
            encoding_maps, positive_only=False)
        cont_names = list(continuous_keys)
        if engineer_features:
            eng, eng_names = engineered_continuous(records)
            cont = np.hstack([cont, eng]).astype(np.float32)
            cont_names = cont_names + eng_names
        # Heavy-tailed targets are fit in log1p space (same gbt LOG1P_TARGETS set
        # the trees use); evaluate() inverts them with expm1. Single source of
        # truth, so the two model families treat the tail identically.
        from model import LOG1P_TARGETS  # gbt (on sys.path above)
        log1p_indices = [i for i, k in enumerate(target_keys) if k in LOG1P_TARGETS]
        for i in log1p_indices:
            tgt[:, i] = np.log1p(tgt[:, i])
        return RocketDataset(cont, cat, tgt, continuous_names=cont_names,
                             log1p_indices=log1p_indices)

    @staticmethod
    def make_loaders(
        dataset: "RocketDataset",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        batch_size: int = 256,
        num_workers: int = 0,
        scale_inputs: bool = True,
        scale_targets: bool = True,
        seed: int = 42,
    ) -> dict:
        """Split → scale → wrap in DataLoaders.

        Returns dict with keys: "train", "val", "test" (DataLoaders),
        "input_scaler", "target_scaler" (fitted scalers),
        "train_idx", "val_idx", "test_idx" (numpy index arrays).

        ROCm notes:
            - Set num_workers > 0 when training on GPU for async data loading.
            - pin_memory=True is applied when CUDA/ROCm is available for faster
              host-to-device transfers.
        """
        train_idx, val_idx, test_idx = make_splits(len(dataset), train_frac, val_frac, test_frac, seed)

        # Keep using random_split for Subset compatibility with _ScoredSubset
        gen = torch.Generator().manual_seed(seed)
        n_train = len(train_idx)
        n_val = len(val_idx)
        n_test = len(test_idx)
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=gen)

        cont_scaler = StandardScaler() if scale_inputs else None
        tgt_scaler = StandardScaler() if scale_targets else None

        if cont_scaler:
            cont_scaler.fit(dataset.continuous[train_idx])
        if tgt_scaler:
            tgt_scaler.fit(dataset.targets[train_idx])

        # ROCm/CUDA: pin_memory speeds up host-to-GPU transfers
        pin = torch.cuda.is_available()

        loaders = {}
        for split_name, subset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            scaled = _ScaledSubset(subset, cont_scaler, tgt_scaler)
            loaders[split_name] = DataLoader(
                scaled,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                pin_memory=pin,
            )

        loaders["input_scaler"] = cont_scaler
        loaders["target_scaler"] = tgt_scaler
        loaders["train_idx"] = train_idx
        loaders["val_idx"] = val_idx
        loaders["test_idx"] = test_idx
        return loaders


class _ScaledSubset(Dataset):
    """Applies scaling lazily to a random_split Subset."""

    def __init__(self, subset, input_scaler, target_scaler):
        self.subset = subset
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        cont, cat, tgt = self.subset[idx]
        if self.input_scaler:
            cont = torch.from_numpy(
                self.input_scaler.transform(cont.unsqueeze(0).numpy())
            ).squeeze(0).float()
        if self.target_scaler:
            tgt = torch.from_numpy(
                self.target_scaler.transform(tgt.unsqueeze(0).numpy())
            ).squeeze(0).float()
        return cont, cat, tgt
