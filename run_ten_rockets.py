#!/usr/bin/env python3
"""Run 10 rocket simulations and record timing + results to JSON.

Usage:
    python run_ten_rockets.py [--seed 42] [--output results.json]
"""

import argparse
import json
import time
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rocket_sim'))

import parameters as params_mod
import rocket_builder
import simulator
import validator
import outputs
from utils import compute_cp_barrowman, stability_margin_calibers


def main():
    parser = argparse.ArgumentParser(description="Run 10 rocket simulations with timing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/ten_rocket_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("RocketSurrogate — 10-Rocket Timing Benchmark")
    print("=" * 60)

    print("\n[1] Sampling parameters...")
    all_params = params_mod.balanced_sample(30, seed=args.seed)
    params = []
    for p in all_params:
        ok, _ = validator.prevalidate(p)
        if ok:
            params.append(p)
        if len(params) >= 10:
            break
    print(f"    Sampled {len(params)} pre-validated designs (seed={args.seed})")

    print("\n[2] Running simulations...")
    results = []
    total_start = time.time()

    for i, p in enumerate(params):
        record = {"rocket_id": i + 1, "input": {}, "output": {}, "timing": {}, "status": "pending"}
        t0 = time.time()
        try:
            result = simulator.run_and_extract(p)
            t_total = time.time() - t0
            record["timing"]["total_seconds"] = round(t_total, 3)

            if result is None:
                record["status"] = "simulation_failed_or_timeout"
                results.append(record)
                print(f"    [{i+1:>2}/10] {record['timing']['total_seconds']:.1f}s  FAILED/TIMEOUT")
                continue

            inp = result["input"]
            out = result["output"]
            cg, cp = compute_cp_barrowman(p)
            sm = stability_margin_calibers(cg, cp, p["diameter_mm"])
            record["input"] = inp
            record["output"] = out
            record["output"]["cg_m"] = round(cg, 4)
            record["output"]["cp_m"] = round(cp, 4)
            record["output"]["stability_margin_calibers"] = round(sm, 2)
            record["status"] = "success"
            print(f"    [{i+1:>2}/10] {record['timing']['total_seconds']:.1f}s  OK  "
                  f"apogee={out['apogee_m']:.0f}m  mach={out['max_mach']:.2f}  sm={sm:.2f}cal")

        except Exception as e:
            record["status"] = f"error: {str(e)}"
            record["timing"]["total_seconds"] = round(time.time() - t_build_start, 3)
            print(f"    [{i+1:>2}/10] {record['timing']['total_seconds']:.1f}s  ERROR: {e}")

        results.append(record)

    total_elapsed = time.time() - total_start
    successes = [r for r in results if r["status"] == "success"]
    sim_times = sorted([r["timing"]["sim_seconds"] for r in successes if "sim_seconds" in r["timing"]])
    n = len(sim_times)

    summary = {
        "benchmark": "RocketSurrogate 10-Rocket Timing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "rocketpy_version": "1.12.1",
        "results": results,
        "summary": {
            "total_rockets": len(params),
            "successful": len(successes),
            "failed": len(params) - len(successes),
            "total_wall_time_seconds": round(total_elapsed, 1),
            "simulation_times": {
                "average_seconds": round(sum(sim_times) / n, 1) if n else None,
                "min_seconds": round(sim_times[0], 1) if n else None,
                "max_seconds": round(sim_times[-1], 1) if n else None,
                "median_seconds": round(sim_times[n // 2], 1) if n else None,
            },
        },
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[3] Results saved to {args.output}")
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Successful:    {len(successes)}/{len(params)}")
    print(f"  Total time:    {total_elapsed:.1f}s")
    if sim_times:
        print(f"  Sim average:   {sum(sim_times)/n:.1f}s")
        print(f"  Sim median:    {sim_times[n//2]:.1f}s")
        print(f"  Sim range:     {sim_times[0]:.1f}s – {sim_times[-1]:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
