"""Parameter sampling: random, Latin Hypercube (LHS), and Sobol sequence.

Fin span is constrained to produce stability margins in [0.5, 4.0] calibers.
Body length is constrained per (diameter, motor_class) to keep
dry_mass / propellant_mass in [1.5, 10.0].
"""

import math
import numpy as np
from scipy.stats import qmc
from typing import List, Dict, Any

import config as cfg
from utils import estimate_max_fin_span_for_stability


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _compute_body_length_range(diameter_mm: int, motor_class: str) -> tuple:
    """Allowed body length range for a (diameter, motor_class) pair.

    Ensures dry_mass / propellant_mass is in [1.5, 10.0].
    Widens the k range by 20% if constraints are tight.
    """
    d_m = diameter_mm / 1000.0
    d2 = d_m ** 2
    if d2 <= 0:
        return cfg.BODY_LENGTH_MIN_M, cfg.BODY_LENGTH_MAX_M

    mspec = cfg.MOTOR_SPECS[motor_class]
    prop_min, prop_max = mspec[0], mspec[1]

    k_min, k_max = cfg.DRY_MASS_K_MIN, cfg.DRY_MASS_K_MAX
    L_min = 1.5 * prop_max / (k_max * d2)
    L_max = 10.0 * prop_min / (k_min * d2)

    if L_min > L_max:
        L_min = 1.5 * prop_max / (k_max * 1.2 * d2)
        L_max = 10.0 * prop_min / (k_min * 0.8 * d2)

    L_min = max(cfg.BODY_LENGTH_MIN_M, L_min)
    L_max = min(cfg.BODY_LENGTH_MAX_M, L_max)

    if L_min > L_max:
        return cfg.BODY_LENGTH_MIN_M, cfg.BODY_LENGTH_MAX_M
    return L_min, L_max


def _compute_fin_span_range(diameter_mm: int, body_length: float,
                            nose_length: float, nose_type: str,
                            root_chord_m: float, tip_chord_m: float,
                            fin_count: int, dry_mass_kg: float,
                            prop_mass_kg: float) -> tuple:
    """Fin span range giving stability in [0.5, 4.0] calibers via Barrowman."""
    d_m = diameter_mm / 1000.0

    min_span, max_span = estimate_max_fin_span_for_stability(
        d_m=d_m, L_nose=nose_length, L_body=body_length,
        n_fins=fin_count, root_chord=root_chord_m, tip_chord=tip_chord_m,
        nose_type=nose_type, dry_mass_kg=dry_mass_kg, prop_mass_kg=prop_mass_kg,
        min_sm=cfg.STABILITY_MARGIN_MIN_CAL, max_sm=cfg.STABILITY_MARGIN_MAX_CAL,
    )

    abs_min = d_m * cfg.FIN_SPAN_MIN_DIAMETERS
    abs_max = d_m * cfg.FIN_SPAN_MAX_DIAMETERS
    min_span = max(min_span, abs_min)
    max_span = min(max_span, abs_max)

    if min_span > max_span:
        return abs_max * 0.95, abs_max
    return min_span, max_span


def _map_sample_to_params(sample: np.ndarray) -> Dict[str, Any]:
    """Map a unit-hypercube sample [0,1]^21 to physical parameters."""
    p: Dict[str, Any] = {}

    p["diameter_mm"] = cfg.BODY_DIAMETERS_MM[
        int(sample[0] * len(cfg.BODY_DIAMETERS_MM)) % len(cfg.BODY_DIAMETERS_MM)]
    p["nose_type"] = cfg.NOSE_TYPES[
        int(sample[1] * len(cfg.NOSE_TYPES)) % len(cfg.NOSE_TYPES)]
    p["fin_count"] = cfg.FIN_COUNTS[
        int(sample[2] * len(cfg.FIN_COUNTS)) % len(cfg.FIN_COUNTS)]

    allowed = cfg.ALLOWED_MOTORS_BY_DIAMETER.get(
        p["diameter_mm"], list(range(len(cfg.MOTOR_CLASSES))))
    motor_pick = int(sample[3] * len(allowed)) % len(allowed)
    p["motor_class"] = cfg.MOTOR_CLASSES[allowed[motor_pick]]

    diam_m = p["diameter_mm"] / 1000.0
    L_min, L_max = _compute_body_length_range(p["diameter_mm"], p["motor_class"])
    p["length_m"] = _clamp(L_min + sample[4] * (L_max - L_min),
                           cfg.BODY_LENGTH_MIN_M, cfg.BODY_LENGTH_MAX_M)

    nose_len_diam = cfg.NOSE_LENGTH_MIN_DIAMETERS + sample[5] * (
        cfg.NOSE_LENGTH_MAX_DIAMETERS - cfg.NOSE_LENGTH_MIN_DIAMETERS)
    p["nose_length_m"] = nose_len_diam * diam_m

    root_chord_diam = cfg.FIN_ROOT_CHORD_MIN_DIAMETERS + sample[6] * (
        cfg.FIN_ROOT_CHORD_MAX_DIAMETERS - cfg.FIN_ROOT_CHORD_MIN_DIAMETERS)
    root_chord_m = root_chord_diam * diam_m
    p["fin_root_chord_m"] = root_chord_m
    tip_frac = cfg.FIN_TIP_CHORD_MIN_FRAC + sample[7] * (1.0 - cfg.FIN_TIP_CHORD_MIN_FRAC)
    p["fin_tip_chord_m"] = tip_frac * root_chord_m
    p["fin_sweep_m"] = sample[9] * root_chord_m
    p["fin_thickness_mm"] = cfg.FIN_THICKNESS_MIN_MM + sample[10] * (
        cfg.FIN_THICKNESS_MAX_MM - cfg.FIN_THICKNESS_MIN_MM)

    dry_mass_k = cfg.DRY_MASS_K_MIN + sample[11] * (cfg.DRY_MASS_K_MAX - cfg.DRY_MASS_K_MIN)
    p["dry_mass_kg"] = _clamp(dry_mass_k * diam_m ** 2 * p["length_m"],
                              cfg.DRY_MASS_MIN_KG, cfg.DRY_MASS_MAX_KG)

    mspec = cfg.MOTOR_SPECS[p["motor_class"]]
    p["propellant_mass_kg"] = mspec[0] + sample[12] * (mspec[1] - mspec[0])
    p["burn_time_s"] = mspec[2] + sample[13] * (mspec[3] - mspec[2])
    p["avg_thrust_N"] = mspec[4] + sample[14] * (mspec[5] - mspec[4])

    span_min, span_max = _compute_fin_span_range(
        diameter_mm=p["diameter_mm"], body_length=p["length_m"],
        nose_length=p["nose_length_m"], nose_type=p["nose_type"],
        root_chord_m=p["fin_root_chord_m"], tip_chord_m=p["fin_tip_chord_m"],
        fin_count=p["fin_count"], dry_mass_kg=p["dry_mass_kg"],
        prop_mass_kg=p["propellant_mass_kg"])
    p["fin_span_m"] = _clamp(span_min + sample[8] * (span_max - span_min),
                              diam_m * cfg.FIN_SPAN_MIN_DIAMETERS,
                              diam_m * cfg.FIN_SPAN_MAX_DIAMETERS)

    p["elevation_m"] = cfg.ELEVATION_MIN_M + sample[15] * (cfg.ELEVATION_MAX_M - cfg.ELEVATION_MIN_M)
    p["temperature_c"] = cfg.TEMPERATURE_MIN_C + sample[16] * (cfg.TEMPERATURE_MAX_C - cfg.TEMPERATURE_MIN_C)
    p["wind_speed_mps"] = cfg.WIND_SPEED_MIN_MS + sample[17] * (cfg.WIND_SPEED_MAX_MS - cfg.WIND_SPEED_MIN_MS)
    p["wind_direction_deg"] = cfg.WIND_DIRECTION_MIN_DEG + sample[18] * (cfg.WIND_DIRECTION_MAX_DEG - cfg.WIND_DIRECTION_MIN_DEG)

    total_length = p["nose_length_m"] + p["length_m"]
    rail_max = max(cfg.RAIL_LENGTH_MIN_M, min(cfg.RAIL_LENGTH_MAX_M, total_length * 2.0))
    p["rail_length_m"] = _clamp(
        cfg.RAIL_LENGTH_MIN_M + sample[19] * (rail_max - cfg.RAIL_LENGTH_MIN_M),
        cfg.RAIL_LENGTH_MIN_M, cfg.RAIL_LENGTH_MAX_M)

    p["launch_angle_deg"] = cfg.LAUNCH_ANGLE_MIN_DEG + sample[20] * (cfg.LAUNCH_ANGLE_MAX_DEG - cfg.LAUNCH_ANGLE_MIN_DEG)
    return p


NUM_CONTINUOUS = 21


def random_sample(n: int, seed: int) -> List[Dict[str, Any]]:
    """Uniform random sampling."""
    rng = np.random.default_rng(seed)
    samples = rng.random((n, NUM_CONTINUOUS))
    return [_map_sample_to_params(s) for s in samples]


def lhs_sample(n: int, seed: int) -> List[Dict[str, Any]]:
    """Latin Hypercube Sampling."""
    sampler = qmc.LatinHypercube(d=NUM_CONTINUOUS, seed=seed)
    return [_map_sample_to_params(s) for s in sampler.random(n=n)]


def sobol_sample(n: int, seed: int) -> List[Dict[str, Any]]:
    """Sobol sequence sampling."""
    try:
        import sobol_seq
    except ImportError:
        raise ImportError("sobol_seq package required: pip install sobol-seq")
    m = 1
    while m < n:
        m <<= 1
    samples = sobol_seq.i4_sobol_generate(NUM_CONTINUOUS, m, seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    return [_map_sample_to_params(s) for s in samples[:n]]


def balanced_sample(n: int, seed: int) -> List[Dict[str, Any]]:
    """Sampling with balanced discrete categories."""
    rng = np.random.default_rng(seed)

    balanced_keys = ["diameter_mm", "nose_type", "fin_count"]
    balanced_options = [cfg.BODY_DIAMETERS_MM, cfg.NOSE_TYPES, cfg.FIN_COUNTS]

    assignments = {}
    for key, options in zip(balanced_keys, balanced_options):
        base, rem = divmod(n, len(options))
        col = []
        for j, opt in enumerate(options):
            col.extend([opt] * (base + (1 if j < rem else 0)))
        rng.shuffle(col)
        assignments[key] = col

    diameter_assignments = assignments["diameter_mm"]
    motor_assignments = [None] * n

    from collections import Counter
    diameter_positions = {}
    for idx, d in enumerate(diameter_assignments):
        diameter_positions.setdefault(d, []).append(idx)

    for d, positions in diameter_positions.items():
        allowed = cfg.ALLOWED_MOTORS_BY_DIAMETER.get(d, list(range(len(cfg.MOTOR_CLASSES))))
        n_motors = len(allowed)
        base, rem = divmod(len(positions), n_motors)
        motor_choices = []
        for j, mi in enumerate(allowed):
            motor_choices.extend([cfg.MOTOR_CLASSES[mi]] * (base + (1 if j < rem else 0)))
        rng.shuffle(motor_choices)
        for pos, mc in zip(positions, motor_choices):
            motor_assignments[pos] = mc

    continuous = rng.random((n, NUM_CONTINUOUS))
    params_list = []

    for idx in range(n):
        sample = continuous[idx]
        diameter_mm = assignments["diameter_mm"][idx]
        nose_type = assignments["nose_type"][idx]
        fin_count = assignments["fin_count"][idx]
        motor_class = motor_assignments[idx]

        diam_idx = cfg.BODY_DIAMETERS_MM.index(diameter_mm)
        allowed_motor_indices = cfg.ALLOWED_MOTORS_BY_DIAMETER.get(diameter_mm, [])

        sc = sample.copy()
        sc[0] = (diam_idx + 0.5) / len(cfg.BODY_DIAMETERS_MM)
        sc[1] = (cfg.NOSE_TYPES.index(nose_type) + 0.5) / len(cfg.NOSE_TYPES)
        sc[2] = (cfg.FIN_COUNTS.index(fin_count) + 0.5) / len(cfg.FIN_COUNTS)

        if allowed_motor_indices:
            mi = [cfg.MOTOR_CLASSES[m] for m in allowed_motor_indices].index(motor_class)
            sc[3] = (mi + 0.5) / len(allowed_motor_indices)

        p = _map_sample_to_params(sc)
        p["diameter_mm"] = diameter_mm
        p["nose_type"] = nose_type
        p["fin_count"] = fin_count
        p["motor_class"] = motor_class

        diam_m = p["diameter_mm"] / 1000.0
        L_min, L_max = _compute_body_length_range(diameter_mm, motor_class)
        p["length_m"] = _clamp(L_min + sample[4] * (L_max - L_min),
                               cfg.BODY_LENGTH_MIN_M, cfg.BODY_LENGTH_MAX_M)

        nose_len_diam = cfg.NOSE_LENGTH_MIN_DIAMETERS + sample[5] * (
            cfg.NOSE_LENGTH_MAX_DIAMETERS - cfg.NOSE_LENGTH_MIN_DIAMETERS)
        p["nose_length_m"] = nose_len_diam * diam_m

        root_chord_diam = cfg.FIN_ROOT_CHORD_MIN_DIAMETERS + sample[6] * (
            cfg.FIN_ROOT_CHORD_MAX_DIAMETERS - cfg.FIN_ROOT_CHORD_MIN_DIAMETERS)
        p["fin_root_chord_m"] = root_chord_diam * diam_m
        tip_frac = cfg.FIN_TIP_CHORD_MIN_FRAC + sample[7] * (1.0 - cfg.FIN_TIP_CHORD_MIN_FRAC)
        p["fin_tip_chord_m"] = tip_frac * p["fin_root_chord_m"]
        p["fin_sweep_m"] = sample[9] * p["fin_root_chord_m"]

        dry_mass_k = cfg.DRY_MASS_K_MIN + sample[11] * (cfg.DRY_MASS_K_MAX - cfg.DRY_MASS_K_MIN)
        p["dry_mass_kg"] = _clamp(dry_mass_k * diam_m ** 2 * p["length_m"],
                                  cfg.DRY_MASS_MIN_KG, cfg.DRY_MASS_MAX_KG)

        mspec = cfg.MOTOR_SPECS[motor_class]
        p["propellant_mass_kg"] = mspec[0] + sample[12] * (mspec[1] - mspec[0])
        p["burn_time_s"] = mspec[2] + sample[13] * (mspec[3] - mspec[2])
        p["avg_thrust_N"] = mspec[4] + sample[14] * (mspec[5] - mspec[4])

        span_min, span_max = _compute_fin_span_range(
            diameter_mm=diameter_mm, body_length=p["length_m"],
            nose_length=p["nose_length_m"], nose_type=nose_type,
            root_chord_m=p["fin_root_chord_m"], tip_chord_m=p["fin_tip_chord_m"],
            fin_count=fin_count, dry_mass_kg=p["dry_mass_kg"],
            prop_mass_kg=p["propellant_mass_kg"])
        p["fin_span_m"] = _clamp(span_min + sample[8] * (span_max - span_min),
                                  diam_m * cfg.FIN_SPAN_MIN_DIAMETERS,
                                  diam_m * cfg.FIN_SPAN_MAX_DIAMETERS)

        params_list.append(p)

    return params_list
