#!/usr/bin/env python3
"""Synthetic rocket data generation pipeline.

Sample -> Pre-validate -> Simulate -> Post-validate -> Save

Two-stage validation catches invalid designs before they reach RocketPy.
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime, timezone
from multiprocessing import cpu_count
from typing import Optional
from collections import Counter

import numpy as np
import rocketpy

import config as cfg
import parameters as params_mod
import rocket_builder
import simulator
import validator
import outputs
from utils import compute_cp_barrowman


def _prevalidate_and_simulate(param_dict: dict) -> Optional[dict]:
    """Full pipeline for one rocket: validate, build, simulate, validate, extract."""
    ok, reason = validator.prevalidate(param_dict)
    if not ok:
        return None

    try:
        rocket = rocket_builder.build_rocket(param_dict)
    except Exception:
        return None

    try:
        flight = simulator.run_simulation(rocket, param_dict)
    except Exception:
        return None

    if flight is None:
        return None

    if not validator.is_valid(param_dict, flight):
        return None

    try:
        out = outputs.extract_output(param_dict, flight)
        inp = outputs.extract_input(param_dict)
        return {"input": inp, "output": out}
    except Exception:
        return None


def _init_worker():
    import warnings
    warnings.filterwarnings("ignore")


def generate(
    count: int = 10000,
    method: str = "random",
    seed: int = 42,
    output: str = "rocket_data.jsonl",
    workers: int = 1,
    splits_dir: Optional[str] = None,
    plots_dir: Optional[str] = None,
    balanced: bool = True,
    oversample_factor: float = 4.0,
) -> None:
    print(f"=== Rocket Data Generator ===")
    print(f"  Target count  : {count}")
    print(f"  Method        : {method}")
    print(f"  Seed          : {seed}")
    print(f"  Workers       : {workers}")
    print(f"  Output        : {output}")
    print(f"  Balanced      : {balanced}")
    print()

    # Sampling
    print("[1/5] Sampling parameters...")
    t0 = time.time()
    n_to_sample = int(count * oversample_factor)
    if balanced:
        param_list = params_mod.balanced_sample(n_to_sample, seed)
    elif method == "lhs":
        param_list = params_mod.lhs_sample(n_to_sample, seed)
    elif method == "sobol":
        param_list = params_mod.sobol_sample(n_to_sample, seed)
    else:
        param_list = params_mod.random_sample(n_to_sample, seed)
    print(f"  Sampled {len(param_list)} parameter sets in {time.time()-t0:.1f}s")

    # Pre-validation
    print("[2/5] Pre-validating parameters...")
    t0 = time.time()
    pre_valids = []
    pre_reject_reasons = Counter()
    for p in param_list:
        ok, reason = validator.prevalidate(p)
        if ok:
            pre_valids.append(p)
        else:
            pre_reject_reasons[reason] += 1
    print(f"  Pre-validated in {time.time()-t0:.1f}s")
    print(f"  Passed: {len(pre_valids)}/{len(param_list)} "
          f"({100*len(pre_valids)/len(param_list):.1f}%)")
    if pre_reject_reasons:
        print(f"  Top rejections:")
        for reason, cnt in pre_reject_reasons.most_common(5):
            print(f"    {reason}: {cnt}")

    if not pre_valids:
        print("ERROR: No designs passed pre-validation.")
        return

    max_to_sim = min(len(pre_valids), int(count * 3))
    param_list_prevalidated = pre_valids[:max_to_sim]

    # Simulation — process-isolated, streaming, resumable.
    # Each sim runs in a child process that the parent kills on overrun, so a
    # hung/leaky RocketPy solve can never accumulate in a long-lived worker.
    # Workers recycle every cfg.MAXTASKSPERCHILD tasks. Results are appended to
    # the JSONL as they complete (crash-safe), and an existing output file is
    # resumed (already-simulated indices are skipped).
    print("[3/5] Running simulations...")
    t0 = time.time()
    import gen_worker

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    # Resume: each record carries its stable index into param_list_prevalidated
    # (deterministic for fixed seed/count/method), so we can skip completed work.
    results = []
    done_ids = set()
    if os.path.exists(output):
        with open(output) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line from a hard crash
                if "id" in rec:
                    done_ids.add(rec["id"])
                results.append(rec)
        if done_ids:
            print(f"  Resume: {len(results)} records already in {output}; "
                  f"skipping those indices")

    tasks = [(i, p) for i, p in enumerate(param_list_prevalidated) if i not in done_ids]
    task_ids = [i for i, _ in tasks]
    task_params = [p for _, p in tasks]

    actual_workers = max(1, min(workers, cpu_count() - 1)) if workers > 1 else 1
    sim_rejected = 0
    processed = 0

    if len(results) < count and task_params:
        with open(output, "a") as fout:
            for local_idx, res in gen_worker.run_batch(
                task_params,
                workers=actual_workers,
                maxtasksperchild=cfg.MAXTASKSPERCHILD,
            ):
                processed += 1
                if res is not None:
                    rec = {"id": task_ids[local_idx], **res}
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    os.fsync(fout.fileno())
                    results.append(rec)
                else:
                    sim_rejected += 1
                if processed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  {processed}/{len(task_params)} simulated, "
                          f"{len(results)} valid, {sim_rejected} rejected "
                          f"({rate:.1f} sims/s)")
                if len(results) >= count:
                    break

    t_sim = time.time() - t0
    print(f"  Completed in {t_sim:.1f}s — {len(results)} valid designs "
          f"({sim_rejected} sim/post-sim rejections)")

    if not results:
        print("ERROR: No valid designs generated after simulation.")
        return

    # Save — records were already streamed to `output` above. Trim to target if
    # we overshot (resume + new run can exceed count), keeping the file in sync.
    print("[4/5] Saving data...")
    if len(results) > count:
        results = results[:count]
        with open(output, "w") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(results)} records to {output}")

    meta = {
        "generation_seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "balanced": balanced,
        "rocketpy_version": "1.12.1",
        "target_count": count,
        "actual_count": len(results),
        "sampling": {
            "n_sampled": n_to_sample,
            "n_prevalidated": len(pre_valids),
            "n_simulated": len(param_list_prevalidated),
            "prevalidation_rate": round(len(pre_valids) / n_to_sample, 3),
            "simulation_rate": round(len(results) / max(1, len(param_list_prevalidated)), 3),
        },
        "timing": {"simulation_seconds": round(t_sim, 1)},
        "validity_filters": {
            "stability_margin_cal": [cfg.STABILITY_MARGIN_MIN_CAL, cfg.STABILITY_MARGIN_MAX_CAL],
            "tw_min": cfg.THRUST_TO_WEIGHT_MIN,
            "max_mach": cfg.MAX_MACH,
            "max_apogee_km": cfg.MAX_APOGEE_KM,
            "sim_timeout_s": cfg.SIM_TIMEOUT_S,
        },
    }
    meta_path = output.replace(".jsonl", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    print("\n  Dataset balance:")
    for key in ["diameter_mm", "nose_type", "fin_count", "motor_class"]:
        counter = Counter(r["input"][key] for r in results)
        print(f"    {key}: {dict(sorted(counter.items()))}")

    if splits_dir:
        print("\n[5/5] Creating train/val/test splits...")
        from splitter import split_dataset
        split_dataset(results, splits_dir, train_frac=0.7, val_frac=0.15, test_frac=0.15)

    if plots_dir:
        print("\n[5/5] Generating summary plots...")
        try:
            from plotter import generate_plots
            generate_plots(results, plots_dir)
        except Exception as e:
            print(f"  Plot generation skipped: {e}")

    print(f"\n=== Done ===")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic rocket design data.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--method", choices=["random", "lhs", "sobol"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="rocket_data.jsonl")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--plots-dir", type=str, default=None)
    parser.add_argument("--no-balanced", action="store_true")
    parser.add_argument("--oversample", type=float, default=4.0)
    args = parser.parse_args()

    generate(
        count=args.count, method=args.method, seed=args.seed,
        output=args.output, workers=args.workers, splits_dir=args.splits_dir,
        plots_dir=args.plots_dir, balanced=not args.no_balanced,
        oversample_factor=args.oversample,
    )


if __name__ == "__main__":
    main()
