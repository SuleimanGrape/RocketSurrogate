#!/usr/bin/env python3
"""Diagnostic: test prevalidation and simulation pass rates for seed 897, using multiprocessing."""
import sys, time, json
from multiprocessing import Pool
sys.path.insert(0, "src/rocket_sim")
import validator
import parameters as params_mod
from generator import _prevalidate_and_simulate

# Use 30 designs across 6 workers
param_list = params_mod.balanced_sample(30, 897)

# Pre-validate sequentially (fast)
param_list_prevalidated = []
pre_rejects = {}
for p in param_list:
    ok, reason = validator.prevalidate(p)
    if ok:
        param_list_prevalidated.append(p)
    else:
        pre_rejects[reason] = pre_rejects.get(reason, 0) + 1

print(f"Pre-validated: {len(param_list_prevalidated)}/{len(param_list)}")
if pre_rejects:
    print(f"Pre-rejects: {json.dumps(pre_rejects, indent=2)}")

# Simulate with 6 workers
print(f"\nSimulating {len(param_list_prevalidated)} designs with 6 workers...")
t0 = time.time()
valid = 0
total = 0
times = []
with Pool(processes=min(6, len(param_list_prevalidated))) as pool:
    for r in pool.imap_unordered(_prevalidate_and_simulate, param_list_prevalidated):
        total += 1
        elapsed = time.time() - t0
        times.append(elapsed)
        if r is not None:
            valid += 1
            print(f"  VALID [{valid}]: {r['input']['motor_class']} {r['input']['diameter_mm']}mm at {elapsed:.1f}s")
        else:
            print(f"  FAIL: ({total}/{len(param_list_prevalidated)}) at {elapsed:.1f}s")
        if (total) % 5 == 0:
            print(f"  [{total}/{len(param_list_prevalidated)}] {time.time()-t0:.0f}s elapsed")

print(f"\n=== Results ===")
print(f"Pre-validated: {len(param_list_prevalidated)}/{len(param_list)}")
print(f"Sim valid: {valid}/{total}")
print(f"Time: {time.time()-t0:.0f}s")