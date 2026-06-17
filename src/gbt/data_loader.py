#!/usr/bin/env python3
"""Load and split rocket data from JSONL files matching generator.py format."""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import schema
from dataio import load_jsonl  # noqa: F401  (re-exported for callers)


# Categorical columns that XGBoost handles natively
CATEGORICAL_COLS = schema.XGB_CATEGORICAL_COLS


def _is_positive(rec: Dict) -> bool:
    """A within-bounds (computable) record. Negatives carry within_bounds=False
    and no numeric targets; legacy records without the field are positive."""
    return rec.get("output", {}).get("within_bounds") is not False


def extract_arrays(
    records: List[Dict],
    input_features: Optional[List[str]] = None,
    output_targets: Optional[List[str]] = None,
    classification: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Extract feature matrix X and target matrix Y from records.

    Args:
        records: list of {"input": {...}, "output": {...}} dicts
        input_features: which input keys to use (None = all)
        output_targets: which output keys to predict (None = all; regression only)
        classification: if True, keep ALL records and return the binary
            within_bounds label (1=computable, 0=not) as a 1-D Y. If False
            (default, regression), drop negative records first so missing-target
            designs never reach the regressors.

    Returns:
        X: (n_samples, n_features) DataFrame with categorical dtypes preserved
        Y: regression -> (n_samples, n_targets) float array;
           classification -> (n_samples,) int array of within_bounds labels
        feature_names: list of feature name strings
        target_names: list of target name strings (["within_bounds"] if classifying)
    """
    if not records:
        raise ValueError("No records to extract.")

    if classification:
        used = records
        y = np.array([1 if _is_positive(r) else 0 for r in used], dtype=np.int64)
        target_names = ["within_bounds"]
    else:
        used = [r for r in records if _is_positive(r)]
        if not used:
            raise ValueError("No positive (within_bounds) records to extract.")

    # Auto-detect feature/target names from the first used record
    if input_features is None:
        input_features = sorted(used[0]["input"].keys())
    if not classification and output_targets is None:
        # Canonical numeric targets (excludes the motor_class passthrough and the
        # within_bounds label, which are not regression targets).
        output_targets = list(schema.TARGETS)

    # Build X as DataFrame with categorical dtypes for XGBoost native support
    X_dict = {}
    for k in input_features:
        col = [r["input"][k] for r in used]
        if k in CATEGORICAL_COLS:
            X_dict[k] = pd.Categorical(col)
        else:
            X_dict[k] = col
    X = pd.DataFrame(X_dict, columns=input_features)

    if classification:
        return X, y, input_features, target_names

    # Build Y as numpy array (targets are all numeric — motor_class excluded from targets)
    Y = np.array([[r["output"][k] for k in output_targets] for r in used], dtype=np.float64)
    return X, Y, input_features, output_targets


def train_val_test_split(
    X: pd.DataFrame,
    Y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """Split data into train/val/test sets."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # test gets the remainder to avoid off-by-one

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": (X.iloc[train_idx].reset_index(drop=True), Y[train_idx]),
        "val": (X.iloc[val_idx].reset_index(drop=True), Y[val_idx]),
        "test": (X.iloc[test_idx].reset_index(drop=True), Y[test_idx]),
    }
