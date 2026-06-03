"""CG computation and Barrowman CP calculation.

All positions measured from the nose tip (forward end),
with positive direction toward the tail.
"""

import math
import numpy as np
from typing import Sequence, Tuple

import config as cfg


def compute_cg(components: Sequence[Tuple[float, float]]) -> float:
    """Compute center of gravity from (mass, position) pairs."""
    total_mass = sum(m for m, _ in components)
    if total_mass == 0:
        return 0.0
    return sum(m * x for m, x in components) / total_mass


def estimate_cg(d_m: float, L_body: float, L_nose: float, nose_type: str,
                 n_fins: int, root: float, tip: float, span: float,
                 dry_mass: float, prop_mass: float) -> float:
    """Estimate CG position from nose tip.

    Shared by compute_cp_barrowman() and estimate_max_fin_span_for_stability().
    Mass fractions are body-length-aware: short rockets get more forward mass
    to keep CG ahead of CP for positive stability.
    """
    L_total = L_nose + L_body
    slenderness = L_body / d_m if d_m > 0 else 10.0
    is_short = slenderness < 8.0

    nose_body_ratio = L_nose / L_body if L_body > 0 else 1.0
    nose_mass_frac = min(0.40, 0.15 + 0.15 * nose_body_ratio)
    if is_short:
        nose_mass_frac = min(0.50, nose_mass_frac + 0.10 * (1.0 - slenderness / 8.0))

    body_mass_frac = max(0.25, 0.50 - 0.10 * nose_body_ratio)
    if is_short:
        body_mass_frac = max(0.20, body_mass_frac - 0.05)

    fin_mass_frac = 0.08
    recovery_frac = 0.07
    electronics_frac = 0.20 if is_short else 0.15
    motor_casing_frac = max(0.02, 1.0 - nose_mass_frac - body_mass_frac -
                            fin_mass_frac - recovery_frac - electronics_frac)

    if nose_type == "conical":
        nose_cg = L_nose / 3.0
    elif nose_type in ("tangent", "ogive"):
        nose_cg = 0.44 * L_nose
    elif nose_type in ("von karman", "vonkarman"):
        nose_cg = 0.5 * L_nose
    elif nose_type == "elliptical":
        nose_cg = 0.4 * L_nose
    else:
        nose_cg = L_nose / 3.0

    body_cg = L_nose + L_body / 2.0
    fin_cg = L_total - root + span / 3.0
    recovery_cg = L_nose + L_body * 0.30
    electronics_cg = L_nose + L_body * (0.30 if is_short else 0.35)
    motor_casing_cg = L_nose + L_body * 0.80
    propellant_cg = L_nose + L_body * 0.60

    components = [
        (max(0, nose_mass_frac * dry_mass), nose_cg),
        (max(0, body_mass_frac * dry_mass), body_cg),
        (max(0, fin_mass_frac * dry_mass), fin_cg),
        (max(0, recovery_frac * dry_mass), recovery_cg),
        (max(0, electronics_frac * dry_mass), electronics_cg),
        (max(0, motor_casing_frac * dry_mass), motor_casing_cg),
        (prop_mass, propellant_cg),
    ]
    return compute_cg(components)


def _cp_nose(nose_type: str, nose_length: float) -> Tuple[float, float]:
    """Return (CN_alpha, x_cp_from_tip) for the nose cone."""
    cn = 2.0
    if nose_type == "conical":
        x_cp = (2.0 / 3.0) * nose_length
    elif nose_type in ("tangent", "ogive"):
        x_cp = 0.56 * nose_length
    elif nose_type in ("von karman", "vonkarman"):
        x_cp = 0.56 * nose_length
    elif nose_type == "elliptical":
        x_cp = 0.5 * nose_length
    else:
        x_cp = 0.5 * nose_length
    return cn, x_cp


def _cn_alpha_fins(n_fins: int, root_chord: float, tip_chord: float,
                   span: float, body_diameter: float) -> float:
    """Barrowman CN_alpha for fins only."""
    r_body = body_diameter / 2.0
    denom = r_body + tip_chord
    if tip_chord == root_chord:
        mac = root_chord
    else:
        mac = (2.0 / 3.0) * (root_chord + tip_chord -
                               (root_chord * tip_chord) / (root_chord + tip_chord))
    a = 2.0 * mac / denom if denom > 0 else 1.0
    return (4.0 * n_fins * (span / body_diameter) ** 2) / (
        1.0 + np.sqrt(1.0 + a ** 2))


def _cp_fins(n_fins: int, root_chord: float, tip_chord: float,
             span: float, body_diameter: float) -> Tuple[float, float, float]:
    """Return (CN_alpha, x_cp_from_fin_le, total_area) for fins."""
    if tip_chord == root_chord:
        mac = root_chord
    else:
        mac = (2.0 / 3.0) * (root_chord + tip_chord -
                               (root_chord * tip_chord) / (root_chord + tip_chord))
    area_single = 0.5 * (root_chord + tip_chord) * span
    total_fins_area = n_fins * area_single
    cn = _cn_alpha_fins(n_fins, root_chord, tip_chord, span, body_diameter)
    if (root_chord + tip_chord) > 0:
        x_from_le = (mac / 4.0 + (span / 6.0) *
                     (root_chord + 2.0 * tip_chord) / (root_chord + tip_chord))
    else:
        x_from_le = mac / 4.0
    return cn, x_from_le, total_fins_area


def compute_fin_cn_limit(n_fins: int, root_chord: float, tip_chord: float,
                         body_diameter: float, nose_cn: float, nose_cp: float,
                         max_sm_cal: float, d_m: float) -> Tuple[float, float]:
    """Placeholder for max fin span CN limit. Returns (0, 0)."""
    return 0.0, 0.0


def compute_cp_barrowman(params: dict) -> Tuple[float, float]:
    """Compute CP and CG positions from nose tip via Barrowman equations."""
    d = params["diameter_mm"] / 1000.0
    L_body = params["length_m"]
    L_nose = params["nose_length_m"]
    nose_type = params["nose_type"]
    n_fins = params["fin_count"]
    root = params["fin_root_chord_m"]
    tip = params["fin_tip_chord_m"]
    span = params["fin_span_m"]
    dry_mass = params["dry_mass_kg"]
    prop_mass = params["propellant_mass_kg"]
    L_total = L_nose + L_body

    cg = estimate_cg(d_m=d, L_body=L_body, L_nose=L_nose, nose_type=nose_type,
                     n_fins=n_fins, root=root, tip=tip, span=span,
                     dry_mass=dry_mass, prop_mass=prop_mass)

    cn_nose, x_cp_nose = _cp_nose(nose_type, L_nose)
    cn_fins, x_cp_fins_from_le, _ = _cp_fins(n_fins, root, tip, span, d)
    cn_total = cn_nose + cn_fins

    if cn_total <= 0:
        cp = L_total / 2.0
    else:
        x_cp_fins = L_total - root + x_cp_fins_from_le
        cp = (cn_nose * x_cp_nose + cn_fins * x_cp_fins) / cn_total

    return cg, cp


def stability_margin_calibers(cg: float, cp: float, diameter_mm: float) -> float:
    """Static stability margin in calibers. Target: 0.5 to 4.0."""
    d_m = diameter_mm / 1000.0
    if d_m == 0:
        return 0.0
    return (cp - cg) / d_m


def estimate_max_fin_span_for_stability(d_m: float, L_nose: float, L_body: float,
                                        n_fins: int, root_chord: float,
                                        tip_chord: float, nose_type: str,
                                        dry_mass_kg: float, prop_mass_kg: float,
                                        min_sm: float = 0.5,
                                        max_sm: float = 4.0) -> Tuple[float, float]:
    """Find fin span range [min, max] that gives stability in [min_sm, max_sm].

    Uses the same CG model as compute_cp_barrowman(). Returns best-effort
    range even when CG is too far aft for a valid solution.
    """
    cn_nose, x_cp_nose = _cp_nose(nose_type, L_nose)
    L_total = L_nose + L_body
    abs_min_span = d_m * cfg.FIN_SPAN_MIN_DIAMETERS
    abs_max_span = d_m * cfg.FIN_SPAN_MAX_DIAMETERS

    nominal_span = (abs_min_span + abs_max_span) / 2.0
    cg_estimate = estimate_cg(
        d_m=d_m, L_body=L_body, L_nose=L_nose, nose_type=nose_type,
        n_fins=n_fins, root=root_chord, tip=tip_chord, span=nominal_span,
        dry_mass=dry_mass_kg, prop_mass=prop_mass_kg,
    )

    def _sm_for_span(s):
        cn_f = _cn_alpha_fins(n_fins, root_chord, tip_chord, s, d_m)
        if cn_nose + cn_f <= 0:
            return 0.0
        if (root_chord + tip_chord) > 0:
            x_cp_f = (L_total - root_chord +
                      root_chord / 4.0 + (s / 6.0) *
                      (root_chord + 2.0 * tip_chord) / (root_chord + tip_chord))
        else:
            x_cp_f = L_total - root_chord + root_chord / 4.0
        cp = (cn_nose * x_cp_nose + cn_f * x_cp_f) / (cn_nose + cn_f)
        return (cp - cg_estimate) / d_m

    sm_at_max = _sm_for_span(abs_max_span)
    sm_at_min = _sm_for_span(abs_min_span)

    if sm_at_max < min_sm:
        return abs_max_span * 0.9, abs_max_span
    if sm_at_min > max_sm:
        return abs_min_span, abs_min_span * 1.1 if abs_min_span > 0 else abs_min_span + 0.001

    lo, hi = abs_min_span, abs_max_span
    for _ in range(40):
        mid = (lo + hi) / 2.0
        sm = _sm_for_span(mid)
        if sm < max_sm:
            lo = mid
        elif sm > max_sm:
            hi = mid
        else:
            lo = hi = mid
            break
    found_max = min(lo, abs_max_span)

    lo, hi = abs_min_span, found_max
    for _ in range(40):
        mid = (lo + hi) / 2.0
        sm = _sm_for_span(mid)
        if sm < min_sm:
            lo = mid
        elif sm > max_sm:
            hi = mid
        else:
            lo = hi = mid
            break
    found_min = max(hi, abs_min_span)

    if found_min > found_max:
        mid = (found_min + found_max) / 2.0
        found_min = found_max = mid

    return found_min, found_max
