#!/usr/bin/env python3
"""Pre-sim performance gate: the cheap drag-aware estimate must reject designs
that clearly bust the Mach/apogee caps, while leaving plausible designs alone.

Run: python tests/test_gate.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "rocket_sim"))

import config as cfg          # noqa: E402
import parameters as params_mod  # noqa: E402
import validator              # noqa: E402
from utils import estimate_peak_performance  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def _first_valid_design():
    """A real design that passes pre-validation, drawn from the sampler."""
    for p in params_mod.balanced_sample(256, seed=0):
        if validator.prevalidate(p)[0]:
            return p
    raise AssertionError("no valid design in balanced sample")


def main():
    good = _first_valid_design()

    # 1) estimate is finite and positive for a real design.
    apogee, mach = estimate_peak_performance(good)
    check(apogee > 0 and mach > 0, "estimate is positive for a real design")

    # 2) sub-thrust-to-weight design returns (0, 0) rather than blowing up.
    weak = dict(good, avg_thrust_N=1.0)
    check(estimate_peak_performance(weak) == (0.0, 0.0),
          "estimate returns (0,0) when thrust <= weight")

    # 3) the good design passes pre-validation (incl. the new gate).
    ok, reason = validator.prevalidate(good)
    check(ok, f"good design passes pre-validation (reason={reason})")

    # 4) cranking thrust/propellant to the M-class ceiling forces a high
    #    estimated Mach, which must trip the gate.
    fast = dict(good, motor_class="M", avg_thrust_N=cfg.MOTOR_SPECS["M"][5],
                propellant_mass_kg=cfg.MOTOR_SPECS["M"][1])
    est_a, est_m = estimate_peak_performance(fast)
    check(est_m > cfg.PREVAL_EST_MACH_MAX,
          f"high-thrust design estimates Mach {est_m:.1f} > cap")
    ok, reason = validator.prevalidate(fast)
    check(not ok and ("Mach" in reason or "apogee" in reason),
          f"gate rejects the high-thrust design (reason={reason})")

    # 5) thresholds are configured.
    check(cfg.PREVAL_EST_MACH_MAX > cfg.MAX_MACH,
          "Mach gate threshold sits above the hard Mach cap")
    check(cfg.PREVAL_EST_APOGEE_MAX_M > cfg.MAX_APOGEE_KM * 1000,
          "apogee gate threshold sits above the hard apogee cap")

    print("\nAll gate checks passed.")


if __name__ == "__main__":
    main()
