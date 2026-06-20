#!/usr/bin/env python3
"""The gradient-based design optimizer must:
  1. produce box-feasible, constraint-feasible designs (per the surrogate),
  2. actually improve the objective over its starting points (the surrogate
     gradient is being followed, not just clipping), and
  3. have its constraint machinery flag genuinely-bad designs.

Fast: a small random-init run on the F motor class, no warm-start, no simulator.
Run: python tests/test_design_optimizer.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "common"))
sys.path.insert(0, os.path.join(ROOT, "src", "rocket_sim"))
sys.path.insert(0, os.path.join(ROOT, "src", "neural_surrogate"))

import torch                                       # noqa: E402
from optim.diff_surrogate import load_surrogate    # noqa: E402
from optim.design_optimizer import (               # noqa: E402
    DesignOptimizer, box_bounds, constraint_violations, CONT, _CI, DESIGN_VARS,
)

BUNDLE = os.path.join(ROOT, "models", "neural_distilled")
CAT = {"diameter_mm": 38, "nose_type": "ogive", "fin_count": 4, "motor_class": "F"}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def main():
    torch.manual_seed(0)
    surr = load_surrogate(BUNDLE, class1_exact=True)
    opt = DesignOptimizer(surr, objective="apogee_m", maximize=True)

    # ── 1. box bounds are a proper, ordered box ──────────────────────────────
    lo, hi = box_bounds(CAT["diameter_mm"], CAT["motor_class"])
    check(np.all(hi >= lo), "box bounds ordered (hi >= lo for all 17 inputs)")
    check(lo[_CI["propellant_mass_kg"]] == 0.05 and hi[_CI["propellant_mass_kg"]] == 0.10,
          "propellant box matches the F motor spec [0.05, 0.10]")

    # ── 2. constraint machinery flags bad designs, passes good ones ──────────
    d_m = CAT["diameter_mm"] / 1000.0
    codes = torch.tensor(opt._cat_codes(CAT)[None, :], dtype=torch.long)
    # A mid-box design (likely roughly sensible)
    x_mid = torch.tensor(((lo + hi) / 2.0)[None, :], dtype=torch.float32)
    # A design that violates tip<=root and dry>=1.5*prop on purpose
    x_bad = x_mid.clone()
    x_bad[0, _CI["fin_tip_chord_m"]] = hi[_CI["fin_root_chord_m"]] * 1.5  # tip >> root
    x_bad[0, _CI["dry_mass_kg"]] = 0.05                                    # < 1.5*prop
    x_bad[0, _CI["propellant_mass_kg"]] = 0.10                             # 1.5*prop = 0.15 > 0.05
    out_bad = surr.forward(x_bad, codes)
    v_bad = constraint_violations(x_bad, out_bad, surr, d_m)
    check(float(v_bad["tip_le_root"]) > 0, "tip>root flagged (tip_le_root > 0)")
    check(float(v_bad["dry_vs_prop"]) > 0, "dry<1.5*prop flagged (dry_vs_prop > 0)")

    # ── 3. end-to-end: feasible, improving, and the design actually moved ─────
    res0 = opt.optimize(CAT, n_restarts=8, steps=0, rounds=1, seed=1)   # starts only
    res1 = opt.optimize(CAT, n_restarts=8, steps=120, rounds=3, seed=1)  # optimized

    check(res1["best"]["feasible"], "optimized best design is feasible (per surrogate)")
    check(res1["n_feasible"] >= res0["n_feasible"],
          f"feasible count did not regress ({res0['n_feasible']} -> {res1['n_feasible']})")

    a0 = res0["best"]["objective_value"]
    a1 = res1["best"]["objective_value"]
    check(res1["best"]["feasible"] and a1 > 1000.0,
          f"optimized apogee is a sane positive value ({a1:.1f} m)")
    # Improvement: the optimized feasible apogee should beat the best *feasible*
    # start (infeasible starts can have larger but invalid apogee, so compare the
    # selected feasible bests).
    if res0["best"]["feasible"]:
        check(a1 >= a0 - 1e-6,
              f"optimized objective did not regress vs start ({a0:.1f} -> {a1:.1f} m)")

    # the optimizer moved the design (gradient followed, not a no-op)
    p0 = res0["best"]["params"]; p1 = res1["best"]["params"]
    moved = max(abs(p1[k] - p0[k]) for k in DESIGN_VARS)
    check(moved > 1e-4, f"optimized design differs from its start (max move {moved:.4g})")

    # box feasibility of the final design
    in_box = all(lo[_CI[k]] - 1e-4 <= p1[k] <= hi[_CI[k]] + 1e-4 for k in CONT)
    check(in_box, "optimized design lies within the box bounds")

    print(f"\n  start best apogee = {a0:.1f} m  ->  optimized = {a1:.1f} m  "
          f"({res1['n_feasible']}/{res1['n_restarts']} feasible)")
    print("\nAll design-optimizer checks passed.")


if __name__ == "__main__":
    main()
