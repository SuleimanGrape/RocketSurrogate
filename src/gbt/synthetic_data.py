#!/usr/bin/env python3
"""Synthetic data generation wrapper for the XGBoost training pipeline.

Reuses the existing rocket_sim pipeline to generate JSONL data files
compatible with data_loader.py's expected format.
"""

import json
import os
import sys
import time

# Add rocket_sim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rocket_sim'))

import parameters as params_mod
import rocket_builder
import simulator
import validator
import outputs


# All input keys from outputs.extract_input()
# XGBoost supports categorical features natively via enable_categorical=True,
# so we keep nose_type and motor_class in the feature set.
INPUT_FEATURES = [
    "diameter_mm",
    "length_m",
    "nose_type",
    "nose_length_m",
    "fin_count",
    "fin_root_chord_m",
    "fin_tip_chord_m",
    "fin_span_m",
    "fin_sweep_m",
    "fin_thickness_mm",
    "dry_mass_kg",
    "motor_class",
    "propellant_mass_kg",
    "burn_time_s",
    "avg_thrust_N",
    "wind_speed_mps",
    "wind_direction_deg",
    "elevation_m",
    "temperature_c",
    "rail_length_m",
    "launch_angle_deg",
]

# Categorical feature names (for dtype conversion)
CATEGORICAL_FEATURES = ["nose_type", "motor_class"]

# Output keys for prediction (motor_class is an input feature, not a prediction target)
TARGET_FEATURES = [
    "apogee_m",
    "max_velocity_mps",
    "max_mach",
    "max_acceleration_mps2",
    "burnout_altitude_m",
    "burnout_velocity_mps",
    "flight_time_s",
    "landing_velocity_mps",
    "stability_margin_calibers",
    "rail_exit_velocity_mps",
    "max_dynamic_pressure_pa",
    "cg_m",
    "cp_m",
]


def generate(count: int = 5000, seed: int = 42, output_path: str = None) -> str:
    """Generate synthetic rocket data and save as JSONL.

    Args:
        count: Number of valid records to generate
        seed: Random seed
        output_path: Output file path (default: outputs/synthetic_data.jsonl)

    Returns:
        Path to the generated JSONL file
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'outputs', 'synthetic_data.jsonl'
        )
        output_path = os.path.abspath(output_path)

    print(f"Generating {count} synthetic rocket records (seed={seed})...")
    t0 = time.time()

    # Oversample to account for rejections
    n_to_sample = int(count * 3.0)
    param_list = params_mod.balanced_sample(n_to_sample, seed)

    # Pre-validate
    pre_valids = []
    for p in param_list:
        ok, _ = validator.prevalidate(p)
        if ok:
            pre_valids.append(p)

    print(f"  Pre-validated: {len(pre_valids)}/{len(param_list)} passed")

    # Simulate
    results = []
    for p in pre_valids:
        if len(results) >= count:
            break
        try:
            rocket = rocket_builder.build_rocket(p)
            flight = simulator.run_simulation(rocket, p)
            if flight is None:
                continue
            if not validator.is_valid(p, flight):
                continue
            out = outputs.extract_output(p, flight)
            inp = outputs.extract_input(p)
            results.append({"input": inp, "output": out})
        except Exception:
            continue

    print(f"  Generated {len(results)} valid records in {time.time()-t0:.1f}s")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"  Saved to {output_path}")
    return output_path
