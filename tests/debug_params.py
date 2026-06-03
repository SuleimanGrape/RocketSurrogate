"""Quick diagnostic: check parameter distributions and prevalidation pass rate."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'rocket_sim'))

import config as cfg
import parameters as params_mod
import validator
from utils import compute_cp_barrowman, stability_margin_calibers
from collections import Counter

params = params_mod.balanced_sample(200, seed=42)

print("=== Stability margin distribution ===")
sm_vals = []
for p in params:
    cg, cp = compute_cp_barrowman(p)
    sm = stability_margin_calibers(cg, cp, p['diameter_mm'])
    sm_vals.append(sm)

sm_vals.sort()
in_range = sum(1 for s in sm_vals if 0.5 <= s <= 4.0)
print(f"  min={sm_vals[0]:.2f}  max={sm_vals[-1]:.2f}  median={sm_vals[len(sm_vals)//2]:.2f}")
print(f"  in [0.5, 4.0]: {in_range}/200 ({100*in_range/200:.0f}%)")

print("\n=== Prevalidation ===")
failures = Counter()
pass_examples = []
for p in params:
    ok, reason = validator.prevalidate(p)
    if not ok:
        failures[reason] += 1
    elif len(pass_examples) < 5:
        pass_examples.append(p)

total_fail = sum(failures.values())
print(f"  Pass: {200 - total_fail}/200  ({100*(200-total_fail)/200:.0f}%)")
for reason, cnt in failures.most_common(10):
    print(f"  {cnt:>3}x {reason}")

print("\n=== Passing examples ===")
for p in pass_examples:
    cg, cp = compute_cp_barrowman(p)
    sm = stability_margin_calibers(cg, cp, p['diameter_mm'])
    ratio = p['dry_mass_kg'] / p['propellant_mass_kg'] if p['propellant_mass_kg'] > 0 else 0
    print(f"  d={p['diameter_mm']} mot={p['motor_class']} L={p['length_m']:.2f} "
          f"nose={p['nose_length_m']:.2f} root={p['fin_root_chord_m']:.3f}d "
          f"span={p['fin_span_m']:.3f} dry={p['dry_mass_kg']:.1f} "
          f"prop={p['propellant_mass_kg']:.3f} ratio={ratio:.1f} sm={sm:.2f}")
