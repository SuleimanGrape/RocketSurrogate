#!/usr/bin/env python3
"""Synthetic data generation wrapper for the XGBoost training pipeline.

Reuses the rocket_sim pipeline (via the process-isolated gen_worker) to generate
JSONL data files compatible with data_loader.py's expected format. Field lists are
imported from common/schema.py so they never drift from the simulator.
"""

import json
import os
import sys
import time

# rocket_sim (parameters, validator, gen_worker) and common (schema) on path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rocket_sim"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import schema
import parameters as params_mod
import validator
import gen_worker

# Field lists — single source of truth in common/schema.py.
INPUT_FEATURES = schema.INPUT_FIELDS
CATEGORICAL_FEATURES = sorted(schema.XGB_CATEGORICAL_COLS)
TARGET_FEATURES = schema.TARGETS


def generate(count: int = 5000, seed: int = 42, output_path: str = None,
             workers: int = 4, oversample: float = 3.0) -> str:
    """Generate synthetic rocket data and save as JSONL.

    Args:
        count: Number of valid records to generate
        seed: Random seed
        output_path: Output file path (default: outputs/synthetic_data.jsonl)
        workers: Worker processes for the isolated simulation pool
        oversample: Sampling multiplier to cover validation rejections

    Returns:
        Path to the generated JSONL file
    """
    if output_path is None:
        output_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "outputs", "synthetic_data.jsonl"))

    print(f"Generating {count} synthetic rocket records (seed={seed})...")
    t0 = time.time()

    # Sample + pre-validate (cheap, before RocketPy).
    n_to_sample = int(count * oversample)
    param_list = params_mod.balanced_sample(n_to_sample, seed)
    pre_valids = [p for p in param_list if validator.prevalidate(p)[0]]
    print(f"  Pre-validated: {len(pre_valids)}/{len(param_list)} passed")

    # Simulate via the process-isolated pool (hard-kill timeouts + recycling).
    results = []
    for _idx, res in gen_worker.run_batch(pre_valids, workers=workers):
        if res is not None:
            results.append(res)
            if len(results) >= count:
                break

    print(f"  Generated {len(results)} valid records in {time.time()-t0:.1f}s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"  Saved to {output_path}")
    return output_path
