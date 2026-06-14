"""Two-stage validity filtering.

Pre-validation (cheap, before RocketPy) catches ~90%+ of invalid designs.
Post-validation filters simulation outputs for physical plausibility.
"""

import math
import numpy as np
import rocketpy

import config as cfg
from utils import compute_cp_barrowman, stability_margin_calibers

DRY_MASS_TO_PROPELLANT_MIN = 1.5


def prevalidate(params: dict) -> tuple:
    """Check parameter physicality before simulation. Returns (ok, reason)."""
    d_mm = params["diameter_mm"]
    d_m = d_mm / 1000.0
    body_l = params["length_m"]
    nose_l = params["nose_length_m"]
    root = params["fin_root_chord_m"]
    tip = params["fin_tip_chord_m"]
    span = params["fin_span_m"]
    n_fins = params["fin_count"]
    dry_mass = params["dry_mass_kg"]
    prop_mass = params["propellant_mass_kg"]
    burn_time = params["burn_time_s"]
    avg_thrust = params["avg_thrust_N"]
    total_mass = dry_mass + prop_mass
    total_length = nose_l + body_l

    if body_l <= 0:
        return False, "body_length <= 0"
    # Slenderness ratio: length / diameter > 80 is structurally unrealistic
    # and breaks Barrowman CP approximations
    if d_m > 0 and total_length / d_m > 80:
        return False, f"slenderness {total_length/d_m:.0f}:1 > 80"
    if span > body_l:
        return False, "fin_span > body_length"
    if tip > root:
        return False, "tip_chord > root_chord"
    if root <= 0 or tip <= 0 or span <= 0:
        return False, "non-positive fin geometry"
    if root > body_l:
        return False, "fin_root_chord > body_length"
    if dry_mass <= DRY_MASS_TO_PROPELLANT_MIN * prop_mass:
        return False, f"dry_mass <= {DRY_MASS_TO_PROPELLANT_MIN}x propellant_mass"
    if dry_mass < cfg.DRY_MASS_MIN_KG:
        return False, "dry_mass below minimum"
    if total_mass <= 0:
        return False, "total mass <= 0"

    weight = total_mass * 9.81
    if weight <= 0 or avg_thrust / weight < cfg.THRUST_TO_WEIGHT_MIN:
        return False, f"T/W < {cfg.THRUST_TO_WEIGHT_MIN}"
    if avg_thrust <= 0:
        return False, "avg_thrust <= 0"
    if burn_time <= 0:
        return False, "burn_time <= 0"
    if prop_mass <= 0:
        return False, "propellant_mass <= 0"

    try:
        cg, cp = compute_cp_barrowman(params)
        sm = stability_margin_calibers(cg, cp, d_mm)
        if sm < cfg.STABILITY_MARGIN_MIN_CAL or sm > cfg.STABILITY_MARGIN_MAX_CAL:
            return False, f"stability_margin {sm:.2f} cal outside [{cfg.STABILITY_MARGIN_MIN_CAL}, {cfg.STABILITY_MARGIN_MAX_CAL}]"
    except Exception as e:
        return False, f"Barrowman CP/CG error: {e}"

    rail = params["rail_length_m"]
    if total_length > 0 and rail > total_length * 3.0:
        return False, f"rail_length {rail:.1f}m > 3x rocket_length {total_length:.1f}m"
    # Estimated rail exit velocity: simple constant-acceleration model
    # Prevents designs that would barely clear the rail (unstable, ODE solver struggles)
    try:
        net_force = avg_thrust - total_mass * 9.81
        if net_force > 0 and rail > 0:
            v_exit = math.sqrt(2.0 * net_force * rail / total_mass)
            if v_exit < 10.0:
                return False, f"estimated rail_exit_velocity {v_exit:.1f} m/s < 10"
    except Exception:
        pass
    if nose_l <= 0 or d_m <= 0:
        return False, "non-positive geometry"
    if nose_l > body_l * 1.5:
        return False, "nose_length > 1.5x body_length"

    body_volume = math.pi * (d_m / 2) ** 2 * body_l
    prop_density = 1815.0
    if prop_mass > body_volume * prop_density * 0.8:
        return False, "propellant_mass exceeds body capacity"

    return True, "ok"


def is_valid(params: dict, flight: rocketpy.Flight | None) -> bool:
    """Post-simulation validity check."""
    if flight is None:
        return False
    try:
        apogee = float(flight.apogee)
        if apogee > cfg.MAX_APOGEE_KM * 1000 or apogee <= 0:
            return False
        if _get_max_mach(flight) > cfg.MAX_MACH:
            return False
        try:
            if float(flight.max_acceleration) > 2000.0:
                return False
        except Exception:
            pass
        # Minimum rail exit velocity for safe stable flight
        try:
            if hasattr(flight, 'out_of_rail_velocity') and float(flight.out_of_rail_velocity) < 10.0:
                return False
        except Exception:
            pass
        # Maximum slenderness ratio (length/diameter) for structural realism
        d_m = params["diameter_mm"] / 1000.0
        if d_m > 0 and params["length_m"] / d_m > 80:
            return False
        return True
    except Exception:
        return False


def _get_max_mach(flight: rocketpy.Flight) -> float:
    try:
        if hasattr(flight, 'max_mach_number') and flight.max_mach_number is not None:
            return float(flight.max_mach_number)
    except Exception:
        pass
    try:
        return float(flight.max_speed) / 340.0
    except Exception:
        return 0.0
