#!/usr/bin/env python3
"""Shared feature engineering — the single source of truth for the engineered
input features, used by both the neural surrogate and the within_bounds
classifier (and verified, in torch, by neural_surrogate/optim/diff_features.py).

Lives in ``common`` so no model package owns it. Depends only on numpy/pandas and
the pure-math Barrowman helpers in ``rocket_sim/utils.py``.
"""

import os
import sys
import numpy as np
from typing import Tuple

# Closed-form Barrowman cg/cp/stability margin (exact functions of the geometry).
# Adding them as features lets a model read off the otherwise hard-to-predict
# stability margin (a small (cp-cg) difference) without any target leakage.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rocket_sim"))
from utils import compute_cp_barrowman, stability_margin_calibers  # noqa: E402

# Targets fit on log1p(y) and inverted with expm1 at predict time. Used for
# heavy-tailed targets where typical (relative) accuracy matters more than R^2:
# max_acceleration_mps2 (mean ~260, rare spikes to ~2000) — log1p cuts MAE ~23%
# and MAPE 4.4%->2.7%, at the cost of a slight R^2 dip (R^2 is dominated by the
# rare large-magnitude points the transform deliberately deweights). Consumers
# invert these so everything downstream stays in the original units; the saved
# model metadata records the log1p target list.
LOG1P_TARGETS = {"max_acceleration_mps2"}

# Inputs compute_cp_barrowman needs; the Barrowman features are only added when
# all are present.
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

    Notes:
        - Computes diameter_m on the fly for derived features but does NOT
          add it as a separate column (avoids linear dependence on diameter_mm).
        - Categorical columns (nose_type, motor_class) are passed through
          unchanged.
    """
    import pandas as pd

    df = pd.DataFrame(X, columns=feature_names)
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

    # Burnout acceleration proxy (m/s^2): analytic upper envelope of peak accel.
    # Peak acceleration occurs near burnout, where the propellant is spent and the
    # mass is ~dry_mass, so a ~ thrust/dry_mass - g. The heavy tail in
    # max_acceleration_mps2 is driven by 1/dry_mass blowing up for light rockets;
    # handing the model this quantity (in the target's own units) linearises that
    # tail instead of forcing it to learn 1/m. tw_ratio_est above uses *total*
    # (liftoff) mass, so this burnout-condition term is independent information.
    if "avg_thrust_N" in df.columns and "dry_mass_kg" in df.columns:
        df["burnout_accel_proxy_mps2"] = df["avg_thrust_N"] / (df["dry_mass_kg"] + 1e-9) - 9.80665
        new_names.append("burnout_accel_proxy_mps2")

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

    # Ballistic coefficient proxy: mass / (Cd * cross_section). Higher = better coast.
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
