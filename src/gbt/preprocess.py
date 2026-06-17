#!/usr/bin/env python3
"""Feature preprocessing: scaling, feature engineering, and train/transform consistency."""

import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional

# Shared scalers (z-score for features, min-max for targets).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from scalers import StandardScaler as FeatureScaler, MinMaxScaler as TargetScaler  # noqa: F401

# Closed-form Barrowman cg/cp/stability margin (exact functions of the geometry).
# Adding them as features lets the trees read off the otherwise hard-to-predict
# stability margin (a small (cp-cg) difference) without any target leakage.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rocket_sim"))
from utils import compute_cp_barrowman, stability_margin_calibers  # noqa: E402

# Inputs compute_cp_barrowman needs; the features are only added when all present.
_BARROWMAN_INPUTS = [
    "diameter_mm", "length_m", "nose_length_m", "nose_type", "fin_count",
    "fin_root_chord_m", "fin_tip_chord_m", "fin_span_m",
    "dry_mass_kg", "propellant_mass_kg",
]


def add_engineered_features(
    X: np.ndarray,
    feature_names: list,
) -> Tuple[np.ndarray, list]:
    """
    Add physically-motivated engineered features for rocket data.
    These help gradient boost trees capture non-linear relationships.

    Notes:
        - Computes diameter_m on the fly for derived features but does NOT
          add it as a separate column (avoids linear dependence on diameter_mm).
        - Categorical columns (nose_type, motor_class) are passed through
          unchanged — XGBoost handles them natively.
    """
    import pandas as pd

    df = pd.DataFrame(X, columns=feature_names)
    new_features = []
    new_names = []

    # Inline diameter conversion (not stored as a separate feature)
    def _diam_m():
        if "diameter_mm" in df.columns:
            return df["diameter_mm"] / 1000.0
        return None

    d_m = None  # lazy compute once below

    # Total mass
    if "dry_mass_kg" in df.columns and "propellant_mass_kg" in df.columns:
        df["total_mass_kg"] = df["dry_mass_kg"] + df["propellant_mass_kg"]
        new_names.append("total_mass_kg")

    # Propellant fraction
    if "propellant_mass_kg" in df.columns and "dry_mass_kg" in df.columns:
        df["propellant_frac"] = df["propellant_mass_kg"] / (df["dry_mass_kg"] + df["propellant_mass_kg"] + 1e-9)
        new_names.append("propellant_frac")

    # Aspect ratio (length / diameter)
    if "length_m" in df.columns and "diameter_mm" in df.columns:
        if d_m is None:
            d_m = _diam_m()
        df["aspect_ratio"] = df["length_m"] / (d_m + 1e-9)
        new_names.append("aspect_ratio")

    # Motor impulse (thrust * burn_time)
    if "avg_thrust_N" in df.columns and "burn_time_s" in df.columns:
        df["motor_impulse_ns"] = df["avg_thrust_N"] * df["burn_time_s"]
        new_names.append("motor_impulse_ns")

    # Thrust-to-weight ratio estimate
    if "avg_thrust_N" in df.columns and "dry_mass_kg" in df.columns and "propellant_mass_kg" in df.columns:
        total_m = df["dry_mass_kg"] + df["propellant_mass_kg"]
        df["tw_ratio_est"] = df["avg_thrust_N"] / (total_m * 9.81 + 1e-9)
        new_names.append("tw_ratio_est")

    # Fin area ratio
    if all(k in df.columns for k in ["fin_root_chord_m", "fin_tip_chord_m", "fin_span_m", "fin_count", "diameter_mm"]):
        if d_m is None:
            d_m = _diam_m()
        fin_area = 0.5 * (df["fin_root_chord_m"] + df["fin_tip_chord_m"]) * df["fin_span_m"] * df["fin_count"]
        body_area = 3.14159 * (d_m / 2) ** 2
        df["fin_area_ratio"] = fin_area / (body_area + 1e-9)
        new_names.append("fin_area_ratio")

    # Nose-body ratio
    if "nose_length_m" in df.columns and "length_m" in df.columns:
        df["nose_body_ratio"] = df["nose_length_m"] / (df["length_m"] + 1e-9)
        new_names.append("nose_body_ratio")

    # Slenderness (volume proxy)
    if "length_m" in df.columns and "diameter_mm" in df.columns:
        if d_m is None:
            d_m = _diam_m()
        df["slenderness"] = df["length_m"] ** 2 / (d_m + 1e-9)
        new_names.append("slenderness")

    # Ballistic coefficient proxy: mass / (Cd * cross_section)
    # Higher = better coasting performance
    if "dry_mass_kg" in df.columns and "propellant_mass_kg" in df.columns and "diameter_mm" in df.columns:
        if d_m is None:
            d_m = _diam_m()
        cross_section = 3.14159 * (d_m / 2) ** 2
        df["ballistic_coeff"] = (df["dry_mass_kg"] + df["propellant_mass_kg"]) / (cross_section + 1e-9)
        new_names.append("ballistic_coeff")

    # Fin loading: fin area per unit span (indicator of pitch damping)
    if all(k in df.columns for k in ["fin_root_chord_m", "fin_tip_chord_m", "fin_span_m", "fin_count"]):
        df["fin_loading"] = (df["fin_root_chord_m"] + df["fin_tip_chord_m"]) * df["fin_span_m"] * df["fin_count"]
        new_names.append("fin_loading")

    # Barrowman cg/cp/stability margin — exact closed-form from the geometry.
    if all(k in df.columns for k in _BARROWMAN_INPUTS):
        def _barrowman(row):
            params = {k: row[k] for k in _BARROWMAN_INPUTS}
            cg, cp = compute_cp_barrowman(params)
            return pd.Series({
                "barrowman_cg_m": cg,
                "barrowman_cp_m": cp,
                "barrowman_margin_cal": stability_margin_calibers(cg, cp, row["diameter_mm"]),
            })
        bw = df.apply(_barrowman, axis=1)
        for col in ("barrowman_cg_m", "barrowman_cp_m", "barrowman_margin_cal"):
            df[col] = bw[col]
            new_names.append(col)

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
