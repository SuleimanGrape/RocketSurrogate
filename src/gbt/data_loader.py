#!/usr/bin/env python3
"""Load and split rocket data from JSONL files matching generator.py format."""

import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional


# Categorical columns that XGBoost handles natively
CATEGORICAL_COLS = {"nose_type", "motor_class"}


def load_jsonl(path: str) -> List[Dict]:
    """Load records from a JSONL file. Each line is {"input": {...}, "output": {...}}."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_arrays(
    records: List[Dict],
    input_features: Optional[List[str]] = None,
    output_targets: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    """
    Extract feature matrix X and target matrix Y from records.

    Args:
        records: list of {"input": {...}, "output": {...}} dicts
        input_features: which input keys to use (None = all)
        output_targets: which output keys to predict (None = all)

    Returns:
        X: (n_samples, n_features) DataFrame with categorical dtypes preserved
        Y: (n_samples, n_targets) array
        feature_names: list of feature name strings
        target_names: list of target name strings
    """
    if not records:
        raise ValueError("No records to extract.")

    # Auto-detect feature/target names from first record
    if input_features is None:
        input_features = sorted(records[0]["input"].keys())
    if output_targets is None:
        output_targets = sorted(records[0]["output"].keys())

    # Build X as DataFrame with categorical dtypes for XGBoost native support
    X_dict = {}
    for k in input_features:
        col = [r["input"][k] for r in records]
        if k in CATEGORICAL_COLS:
            X_dict[k] = pd.Categorical(col)
        else:
            X_dict[k] = col
    X = pd.DataFrame(X_dict, columns=input_features)

    # Build Y as numpy array (targets are all numeric — motor_class excluded from targets)
    Y = np.array([[r["output"][k] for k in output_targets] for r in records], dtype=np.float64)

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
