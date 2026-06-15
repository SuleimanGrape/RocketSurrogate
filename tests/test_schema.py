#!/usr/bin/env python3
"""Schema consistency: the single source of truth (common/schema.py) must agree
with what the simulator actually produces and what the models consume.

Run: python tests/test_schema.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "common"))
sys.path.insert(0, os.path.join(ROOT, "src", "rocket_sim"))

import schema          # noqa: E402
import config as cfg   # noqa: E402
import parameters as params_mod  # noqa: E402
import outputs         # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def main():
    # 1) extract_input keys == schema.INPUT_FIELDS (exact order).
    sample = params_mod.balanced_sample(64, seed=0)[0]
    inp_keys = list(outputs.extract_input(sample).keys())
    check(inp_keys == schema.INPUT_FIELDS,
          "extract_input keys match schema.INPUT_FIELDS")

    # 2) continuous + categorical partition the input fields.
    check(set(schema.INPUT_CONTINUOUS) | set(schema.INPUT_CATEGORICAL) == set(schema.INPUT_FIELDS),
          "continuous + categorical == all input fields")
    check(not (set(schema.INPUT_CONTINUOUS) & set(schema.INPUT_CATEGORICAL)),
          "continuous and categorical are disjoint")

    # 3) ENCODING_MAPS derived correctly from config.
    check(list(schema.ENCODING_MAPS["motor_class"]) == list(cfg.MOTOR_CLASSES),
          "motor_class encoding matches config order")
    check(list(schema.ENCODING_MAPS["diameter_mm"]) == list(cfg.BODY_DIAMETERS_MM),
          "diameter_mm encoding matches config order")

    # 4) targets: no landing_velocity_mps, time_to_apogee_s present.
    check("landing_velocity_mps" not in schema.TARGETS,
          "landing_velocity_mps removed from TARGETS")
    check("time_to_apogee_s" in schema.TARGETS,
          "time_to_apogee_s present in TARGETS")

    # 5) neural surrogate consumes the same schema.
    sys.path.insert(0, os.path.join(ROOT, "src", "neural_surrogate"))
    from models import surrogate  # noqa: E402
    check(surrogate.TARGETS == schema.TARGETS, "surrogate.TARGETS == schema.TARGETS")
    check(surrogate.CONTINUOUS_FEATURES == schema.INPUT_CONTINUOUS,
          "surrogate.CONTINUOUS_FEATURES == schema.INPUT_CONTINUOUS")
    check(surrogate.CATEGORICAL_FEATURES == schema.INPUT_CATEGORICAL,
          "surrogate.CATEGORICAL_FEATURES == schema.INPUT_CATEGORICAL")

    # 6) gbt synthetic_data re-exports the schema fields.
    sys.path.insert(0, os.path.join(ROOT, "src", "gbt"))
    import synthetic_data  # noqa: E402
    check(synthetic_data.INPUT_FEATURES == schema.INPUT_FIELDS,
          "synthetic_data.INPUT_FEATURES == schema.INPUT_FIELDS")
    check(synthetic_data.TARGET_FEATURES == schema.TARGETS,
          "synthetic_data.TARGET_FEATURES == schema.TARGETS")

    print("\nAll schema consistency checks passed.")


if __name__ == "__main__":
    main()
