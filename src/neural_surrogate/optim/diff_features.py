"""Differentiable re-implementation of the engineered continuous features.

The XGBoost / NN training path builds 14 engineered continuous features in
numpy/pandas (``gbt/preprocess.add_engineered_features``), including the exact
closed-form Barrowman cg / cp / stability margin. That code is correct but
breaks autograd: gradients cannot flow from a target back to a raw design input
through pandas. For gradient-based design optimization we need exactly that, so
this module re-derives the identical formulas in torch.

Contract (verified by ``tests`` / ``eval_gradients``):
    ENGINEERED(x_cont_raw, x_cat) reproduces preprocess.add_engineered_features
    column-for-column, in the same order, to ~1e-5 relative error — while being
    differentiable w.r.t. every continuous raw input.

Categorical inputs (diameter, nose type, fin count, motor class) are discrete,
so gradients are taken only w.r.t. the 17 continuous raw inputs; the categoricals
enter as per-row constants decoded from their integer codes (matching the values
the numpy path reads from the raw record).
"""

from __future__ import annotations

import os
import sys
from typing import List

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "common"))
import schema  # noqa: E402

# ── Column layout (single source of truth in common/schema.py) ────────────────
CONT = list(schema.INPUT_CONTINUOUS)          # 17 base continuous, in order
CAT = list(schema.INPUT_CATEGORICAL)          # [diameter_mm, nose_type, fin_count, motor_class]
_CI = {name: i for i, name in enumerate(CONT)}
_KI = {name: i for i, name in enumerate(CAT)}

# Names of the 14 engineered columns, in the exact order add_engineered_features
# appends them. ENGINEERED_NAMES + CONT == the model's continuous feature order.
ENGINEERED_NAMES: List[str] = [
    "total_mass_kg", "propellant_frac", "aspect_ratio", "motor_impulse_ns",
    "tw_ratio_est", "burnout_accel_proxy_mps2", "fin_area_ratio",
    "nose_body_ratio", "slenderness", "ballistic_coeff", "fin_loading",
    "barrowman_cg_m", "barrowman_cp_m", "barrowman_margin_cal",
]
CONTINUOUS_NAMES: List[str] = CONT + ENGINEERED_NAMES   # 31 columns

_PI = 3.14159   # preprocess.py uses the literal 3.14159, not math.pi — match it.
_G = 9.80665


# ── Categorical decode tables (code → physical value / coefficient) ────────────
def _code_order(field: str) -> list:
    """Values of a categorical field in code order (code 0, 1, 2, ...)."""
    return [v for v, _ in sorted(schema.ENCODING_MAPS[field].items(),
                                 key=lambda kv: kv[1])]


_DIAM_MM_BY_CODE = _code_order("diameter_mm")     # e.g. [24, 29, 38, 54, 75, 98]
_FIN_N_BY_CODE = _code_order("fin_count")         # e.g. [3, 4]
_NOSE_BY_CODE = _code_order("nose_type")          # e.g. [conical, ogive, von_karman, elliptical]


def _nose_cg_coeff(nose_type: str) -> float:
    """nose_cg / L_nose — replicates the dispatch in utils.estimate_cg exactly,
    including that config's 'von_karman' string falls through to the else branch
    (the code only matches 'von karman'/'vonkarman')."""
    if nose_type == "conical":
        return 1.0 / 3.0
    elif nose_type in ("tangent", "ogive"):
        return 0.44
    elif nose_type in ("von karman", "vonkarman"):
        return 0.5
    elif nose_type == "elliptical":
        return 0.4
    return 1.0 / 3.0


def _nose_xcp_coeff(nose_type: str) -> float:
    """x_cp_nose / L_nose — replicates utils._cp_nose exactly (same quirk)."""
    if nose_type == "conical":
        return 2.0 / 3.0
    elif nose_type in ("tangent", "ogive"):
        return 0.56
    elif nose_type in ("von karman", "vonkarman"):
        return 0.56
    elif nose_type == "elliptical":
        return 0.5
    return 0.5


_NOSE_CG_COEFF_BY_CODE = [_nose_cg_coeff(nt) for nt in _NOSE_BY_CODE]
_NOSE_XCP_COEFF_BY_CODE = [_nose_xcp_coeff(nt) for nt in _NOSE_BY_CODE]


def _decode_cat(x_cat: torch.Tensor):
    """Map (B,4) integer codes to per-row physical constants used by the
    continuous feature formulas. Returned tensors are constants (no grad)."""
    dev, dt = x_cat.device, torch.float64
    diam_lut = torch.tensor(_DIAM_MM_BY_CODE, device=dev, dtype=dt)
    finn_lut = torch.tensor(_FIN_N_BY_CODE, device=dev, dtype=dt)
    cg_lut = torch.tensor(_NOSE_CG_COEFF_BY_CODE, device=dev, dtype=dt)
    xcp_lut = torch.tensor(_NOSE_XCP_COEFF_BY_CODE, device=dev, dtype=dt)

    diam_mm = diam_lut[x_cat[:, _KI["diameter_mm"]]]
    n_fins = finn_lut[x_cat[:, _KI["fin_count"]]]
    nose_cg_c = cg_lut[x_cat[:, _KI["nose_type"]]]
    nose_xcp_c = xcp_lut[x_cat[:, _KI["nose_type"]]]
    return diam_mm, n_fins, nose_cg_c, nose_xcp_c


# ── Barrowman cg / cp / margin (torch; differentiable in the continuous inputs) ─
def _mac(root: torch.Tensor, tip: torch.Tensor) -> torch.Tensor:
    """Mean aerodynamic chord. The general Barrowman formula already equals
    `root` when tip == root, so a single (guarded) expression matches both
    branches of utils._cp_fins / _cn_alpha_fins."""
    s = root + tip
    safe = torch.where(s > 0, s, torch.ones_like(s))
    general = (2.0 / 3.0) * (root + tip - (root * tip) / safe)
    return torch.where(s > 0, general, root)


def barrowman(L_body, L_nose, root, tip, span, dry_mass, prop_mass,
              diam_mm, n_fins, nose_cg_c, nose_xcp_c):
    """Return (cg, cp, margin_calibers) — exact torch port of
    utils.compute_cp_barrowman + estimate_cg + stability_margin_calibers."""
    d = diam_mm / 1000.0
    L_total = L_nose + L_body

    # ---- estimate_cg ----
    slenderness = torch.where(d > 0, L_body / d, torch.full_like(d, 10.0))
    is_short = slenderness < 8.0
    nose_body_ratio = torch.where(L_body > 0, L_nose / L_body, torch.ones_like(L_body))

    nose_mass_frac = torch.clamp(0.15 + 0.15 * nose_body_ratio, max=0.40)
    nose_mass_frac = torch.where(
        is_short,
        torch.clamp(nose_mass_frac + 0.10 * (1.0 - slenderness / 8.0), max=0.50),
        nose_mass_frac)

    body_mass_frac = torch.clamp(0.50 - 0.10 * nose_body_ratio, min=0.25)
    body_mass_frac = torch.where(
        is_short, torch.clamp(body_mass_frac - 0.05, min=0.20), body_mass_frac)

    fin_mass_frac = 0.08
    recovery_frac = 0.07
    electronics_frac = torch.where(is_short, torch.full_like(d, 0.20),
                                   torch.full_like(d, 0.15))
    motor_casing_frac = torch.clamp(
        1.0 - nose_mass_frac - body_mass_frac - fin_mass_frac
        - recovery_frac - electronics_frac, min=0.02)

    nose_cg = nose_cg_c * L_nose
    body_cg = L_nose + L_body / 2.0
    fin_cg = L_total - root + span / 3.0
    recovery_cg = L_nose + L_body * 0.30
    electronics_cg = L_nose + L_body * torch.where(
        is_short, torch.full_like(d, 0.30), torch.full_like(d, 0.35))
    motor_casing_cg = L_nose + L_body * 0.80
    propellant_cg = L_nose + L_body * 0.60

    def _pos(x):  # max(0, x)
        return torch.clamp(x, min=0.0)

    m_nose = _pos(nose_mass_frac * dry_mass)
    m_body = _pos(body_mass_frac * dry_mass)
    m_fin = _pos(fin_mass_frac * dry_mass)
    m_rec = _pos(recovery_frac * dry_mass)
    m_elec = _pos(electronics_frac * dry_mass)
    m_motor = _pos(motor_casing_frac * dry_mass)
    m_prop = prop_mass

    total_mass = m_nose + m_body + m_fin + m_rec + m_elec + m_motor + m_prop
    moment = (m_nose * nose_cg + m_body * body_cg + m_fin * fin_cg
              + m_rec * recovery_cg + m_elec * electronics_cg
              + m_motor * motor_casing_cg + m_prop * propellant_cg)
    cg = torch.where(total_mass > 0, moment / torch.where(total_mass > 0, total_mass,
                                                          torch.ones_like(total_mass)),
                     torch.zeros_like(total_mass))

    # ---- cp (Barrowman) ----
    cn_nose = 2.0
    x_cp_nose = nose_xcp_c * L_nose

    r_body = d / 2.0
    denom = r_body + tip
    mac = _mac(root, tip)
    a = torch.where(denom > 0, 2.0 * mac / torch.where(denom > 0, denom,
                                                       torch.ones_like(denom)),
                    torch.ones_like(denom))
    cn_fins = (4.0 * n_fins * (span / d) ** 2) / (1.0 + torch.sqrt(1.0 + a ** 2))

    s = root + tip
    safe_s = torch.where(s > 0, s, torch.ones_like(s))
    x_from_le = torch.where(
        s > 0,
        mac / 4.0 + (span / 6.0) * (root + 2.0 * tip) / safe_s,
        mac / 4.0)

    cn_total = cn_nose + cn_fins
    x_cp_fins = L_total - root + x_from_le
    cp_main = (cn_nose * x_cp_nose + cn_fins * x_cp_fins) / cn_total
    cp = torch.where(cn_total > 0, cp_main, L_total / 2.0)

    margin = torch.where(d > 0, (cp - cg) / torch.where(d > 0, d, torch.ones_like(d)),
                         torch.zeros_like(d))
    return cg, cp, margin


# ── Full engineered feature block ─────────────────────────────────────────────
def engineered_block(x_cont_raw: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
    """Compute the 14 engineered columns (B, 14) from raw inputs.

    x_cont_raw : (B, 17) continuous raw inputs in schema.INPUT_CONTINUOUS order.
    x_cat      : (B, 4)  integer codes in schema.INPUT_CATEGORICAL order.
    Differentiable w.r.t. x_cont_raw. Output dtype follows x_cont_raw.
    """
    out_dtype = x_cont_raw.dtype
    x = x_cont_raw.to(torch.float64)

    length = x[:, _CI["length_m"]]
    nose_len = x[:, _CI["nose_length_m"]]
    root = x[:, _CI["fin_root_chord_m"]]
    tip = x[:, _CI["fin_tip_chord_m"]]
    span = x[:, _CI["fin_span_m"]]
    dry = x[:, _CI["dry_mass_kg"]]
    prop = x[:, _CI["propellant_mass_kg"]]
    burn = x[:, _CI["burn_time_s"]]
    thrust = x[:, _CI["avg_thrust_N"]]

    diam_mm, n_fins, nose_cg_c, nose_xcp_c = _decode_cat(x_cat)
    diam_mm = diam_mm.to(torch.float64)
    n_fins = n_fins.to(torch.float64)
    nose_cg_c = nose_cg_c.to(torch.float64)
    nose_xcp_c = nose_xcp_c.to(torch.float64)
    d_m = diam_mm / 1000.0

    total_mass = dry + prop
    propellant_frac = prop / (dry + prop + 1e-9)
    aspect_ratio = length / (d_m + 1e-9)
    motor_impulse = thrust * burn
    tw_ratio = thrust / (total_mass * 9.81 + 1e-9)
    burnout_accel = thrust / (dry + 1e-9) - _G
    fin_area = 0.5 * (root + tip) * span * n_fins
    body_area = _PI * (d_m / 2.0) ** 2
    fin_area_ratio = fin_area / (body_area + 1e-9)
    nose_body_ratio = nose_len / (length + 1e-9)
    slenderness = length ** 2 / (d_m + 1e-9)
    cross_section = _PI * (d_m / 2.0) ** 2
    ballistic_coeff = (dry + prop) / (cross_section + 1e-9)
    fin_loading = (root + tip) * span * n_fins

    cg, cp, margin = barrowman(length, nose_len, root, tip, span, dry, prop,
                               diam_mm, n_fins, nose_cg_c, nose_xcp_c)

    cols = [total_mass, propellant_frac, aspect_ratio, motor_impulse, tw_ratio,
            burnout_accel, fin_area_ratio, nose_body_ratio, slenderness,
            ballistic_coeff, fin_loading, cg, cp, margin]
    return torch.stack(cols, dim=1).to(out_dtype)


def continuous_block(x_cont_raw: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
    """Full (B, 31) continuous feature matrix = raw 17 ++ engineered 14, in the
    exact order the model's input scaler / first layer expect."""
    return torch.cat([x_cont_raw, engineered_block(x_cont_raw, x_cat)], dim=1)
