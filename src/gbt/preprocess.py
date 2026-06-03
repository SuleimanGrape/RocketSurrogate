#!/usr/bin/env python3
"""Feature preprocessing: scaling, feature engineering, and train/transform consistency."""

import numpy as np
from typing import Dict, Tuple, Optional


class FeatureScaler:
    """Standard scaler (z-score) that fits on train and transforms all splits."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # Avoid division by zero for constant features
        self.std_[self.std_ < 1e-10] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class TargetScaler:
    """Min-max scaler for targets, per-column, fits on train only."""

    def __init__(self):
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, Y: np.ndarray) -> "TargetScaler":
        self.min_ = Y.min(axis=0)
        self.max_ = Y.max(axis=0)
        range_ = self.max_ - self.min_
        range_[range_ < 1e-10] = 1.0
        self.range_ = range_
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("TargetScaler not fitted. Call fit() first.")
        return (Y - self.min_) / self.range_

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        if self.min_ is None:
            raise RuntimeError("TargetScaler not fitted.")
        return Y * self.range_ + self.min_

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        return self.fit(Y).transform(Y)


def add_engineered_features(
    X: np.ndarray,
    feature_names: list,
) -> Tuple[np.ndarray, list]:
    """
    Add physically-motivated engineered features for rocket data.
    These help gradient boost trees capture non-linear relationships.
    """
    import pandas as pd

    df = pd.DataFrame(X, columns=feature_names)
    new_features = []
    new_names = []

    # Diameter in meters (derived from diameter_mm)
    if "diameter_mm" in df.columns:
        df["diameter_m"] = df["diameter_mm"] / 1000.0
        new_names.append("diameter_m")

    # Total mass
    if "dry_mass_kg" in df.columns and "propellant_mass_kg" in df.columns:
        df["total_mass_kg"] = df["dry_mass_kg"] + df["propellant_mass_kg"]
        new_names.append("total_mass_kg")

    # Propellant fraction
    if "propellant_mass_kg" in df.columns and "dry_mass_kg" in df.columns:
        df["propellant_frac"] = df["propellant_mass_kg"] / (df["dry_mass_kg"] + df["propellant_mass_kg"] + 1e-9)
        new_names.append("propellant_frac")

    # Aspect ratio (length / diameter)
    if "length_m" in df.columns and "diameter_m" in df.columns:
        df["aspect_ratio"] = df["length_m"] / (df["diameter_m"] + 1e-9)
        new_names.append("aspect_ratio")

    # Motor impulse (thrust * burn_time) and average thrust re-derived
    if "avg_thrust_N" in df.columns and "burn_time_s" in df.columns:
        df["motor_impulse_ns"] = df["avg_thrust_N"] * df["burn_time_s"]
        new_names.append("motor_impulse_ns")

    # Thrust-to-weight ratio estimate
    if "avg_thrust_N" in df.columns and "total_mass_kg" in df.columns:
        df["tw_ratio_est"] = df["avg_thrust_N"] / (df["total_mass_kg"] * 9.81 + 1e-9)
        new_names.append("tw_ratio_est")

    # Fin area ratio
    if all(k in df.columns for k in ["fin_root_chord_m", "fin_tip_chord_m", "fin_span_m", "fin_count", "diameter_m"]):
        fin_area = 0.5 * (df["fin_root_chord_m"] + df["fin_tip_chord_m"]) * df["fin_span_m"] * df["fin_count"]
        body_area = 3.14159 * (df["diameter_m"] / 2) ** 2
        df["fin_area_ratio"] = fin_area / (body_area + 1e-9)
        new_names.append("fin_area_ratio")

    # Nose-body ratio
    if "nose_length_m" in df.columns and "length_m" in df.columns:
        df["nose_body_ratio"] = df["nose_length_m"] / (df["length_m"] + 1e-9)
        new_names.append("nose_body_ratio")

    # Slenderness (volume proxy)
    if "length_m" in df.columns and "diameter_m" in df.columns:
        df["slenderness"] = df["length_m"] ** 2 / (df["diameter_m"] + 1e-9)
        new_names.append("slenderness")

    result_names = feature_names + new_names
    return df[result_names], result_names


def preprocess(
    splits: Dict[str, Tuple],
    feature_names: list,
    scale_features: bool = True,
    scale_targets: bool = False,
    engineer_features: bool = True,
) -> Dict:
    """
    Full preprocessing pipeline.

    Args:
        splits: {"train": (X, Y), "val": (X, Y), "test": (X, Y)}
                X can be pd.DataFrame (with categorical columns) or np.ndarray
        feature_names: list of original feature name strings
        scale_features: apply z-score normalization to features
        scale_targets: apply min-max scaling to targets
        engineer_features: add derived features

    Returns:
        Dictionary with processed arrays, fitted scalers, and updated feature_names.
    """
    current_names = list(feature_names)

    # ── Feature engineering (same transform on all splits) ────────────
    if engineer_features:
        X_train, Y_train = splits["train"]
        X_train, current_names = add_engineered_features(X_train, current_names)
        X_val, _ = splits["val"]
        X_val, _ = add_engineered_features(X_val, feature_names)
        X_test, _ = splits["test"]
        X_test, _ = add_engineered_features(X_test, feature_names)
        splits = {
            "train": (X_train, Y_train),
            "val": (X_val, splits["val"][1]),
            "test": (X_test, splits["test"][1]),
        }

    # ── Fit scalers on training data ─────────────────────────────────
    if scale_features:
        feat_scaler = FeatureScaler()
        feat_scaler.fit(splits["train"][0])
    else:
        feat_scaler = None

    if scale_targets:
        targ_scaler = TargetScaler()
        targ_scaler.fit(splits["train"][1])
    else:
        targ_scaler = None

    # ── Transform all splits ─────────────────────────────────────────
    result = {
        "feature_names": current_names,
        "feature_scaler": feat_scaler,
        "target_scaler": targ_scaler,
        "engineered": engineer_features,
        "n_features": splits["train"][0].shape[1],
        "n_targets": splits["train"][1].shape[1],
    }
    for split_name, (X, Y) in splits.items():
        if feat_scaler is not None:
            X = feat_scaler.transform(X)
        if targ_scaler is not None:
            Y = targ_scaler.transform(Y)
        result[split_name] = {"X": X, "Y": Y}
        result[f"X_{split_name}"] = X
        result[f"Y_{split_name}"] = Y

    return result
