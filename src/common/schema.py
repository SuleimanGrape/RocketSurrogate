"""Single source of truth for the rocket dataset schema.

Every package (rocket_sim, gbt, neural_surrogate) imports field names, categorical
encodings, and target lists from here so they can never silently drift. The lists
mirror the dicts produced by ``rocket_sim/outputs.py`` (extract_input / extract_output)
and the discrete choices in ``rocket_sim/config.py``; ``tests/test_schema.py`` asserts
they stay consistent.

Import from any package with::

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))  # depth-1
    import schema
"""

import os
import sys

# config.py lives in rocket_sim; pull the discrete choices from there so the
# encodings below are derived, never re-hardcoded.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rocket_sim"))
import config as cfg  # noqa: E402


# ── Inputs ────────────────────────────────────────────────────────────────────
# Continuous scalar inputs, in the order the neural surrogate expects them.
INPUT_CONTINUOUS = [
    "length_m",
    "nose_length_m",
    "fin_root_chord_m",
    "fin_tip_chord_m",
    "fin_span_m",
    "fin_sweep_m",
    "fin_thickness_mm",
    "dry_mass_kg",
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

# Categorical inputs (string/discrete labels → integer codes via ENCODING_MAPS).
INPUT_CATEGORICAL = ["diameter_mm", "nose_type", "fin_count", "motor_class"]

# Full ordered input field list, matching outputs.extract_input() exactly.
INPUT_FIELDS = [
    "diameter_mm", "length_m", "nose_type", "nose_length_m", "fin_count",
    "fin_root_chord_m", "fin_tip_chord_m", "fin_span_m", "fin_sweep_m",
    "fin_thickness_mm", "dry_mass_kg", "motor_class", "propellant_mass_kg",
    "burn_time_s", "avg_thrust_N", "wind_speed_mps", "wind_direction_deg",
    "elevation_m", "temperature_c", "rail_length_m", "launch_angle_deg",
]

# ── Categorical encodings (derived from config) ────────────────────────────────
ENCODING_MAPS = {
    "diameter_mm": {v: i for i, v in enumerate(cfg.BODY_DIAMETERS_MM)},
    "nose_type":   {v: i for i, v in enumerate(cfg.NOSE_TYPES)},
    "fin_count":   {v: i for i, v in enumerate(cfg.FIN_COUNTS)},
    "motor_class": {v: i for i, v in enumerate(cfg.MOTOR_CLASSES)},
}
CATEGORICAL_CARDINALITIES = {k: len(v) for k, v in ENCODING_MAPS.items()}

# Categorical columns XGBoost handles natively as pandas Categorical. diameter_mm
# and fin_count are left numeric/ordinal for the tree models (embeddings for the NN).
XGB_CATEGORICAL_COLS = {"nose_type", "motor_class"}

# ── Outputs ────────────────────────────────────────────────────────────────────
# Numeric regression targets, in outputs.extract_output() order. Note: the sim
# terminates at apogee, so there is no landing_velocity_mps. time_to_apogee_s IS
# produced and belongs here.
TARGETS = [
    "apogee_m",
    "max_velocity_mps",
    "max_mach",
    "max_acceleration_mps2",
    "burnout_altitude_m",
    "burnout_velocity_mps",
    "flight_time_s",
    "time_to_apogee_s",
    "stability_margin_calibers",
    "rail_exit_velocity_mps",
    "max_dynamic_pressure_pa",
    "cg_m",
    "cp_m",
]

# Non-target field carried through extract_output for bookkeeping (an input echo).
OUTPUT_PASSTHROUGH = ["motor_class"]

# Binary computability label written on every output record: True when the
# simulator resolved the design within the validity/timeout bounds, False for
# designs that time out or are otherwise not computable (that negative-class data
# is generated separately so the surrogate can learn what is not computable).
# This is a CLASSIFICATION target, deliberately kept out of the numeric
# regression TARGETS above.
WITHIN_BOUNDS_FIELD = "within_bounds"

# ── Display subsets (for plotter.py distribution plots) ────────────────────────
PLOT_INPUT_KEYS = [
    "diameter_mm", "length_m", "nose_length_m", "fin_count",
    "fin_root_chord_m", "fin_span_m", "dry_mass_kg", "motor_class",
    "wind_speed_mps", "elevation_m",
]
PLOT_OUTPUT_KEYS = [
    "apogee_m", "max_velocity_mps", "max_mach", "max_acceleration_mps2",
    "burnout_altitude_m", "flight_time_s",
    "stability_margin_calibers", "rail_exit_velocity_mps",
    "max_dynamic_pressure_pa",
]
