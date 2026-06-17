#!/usr/bin/env python3
"""Consolidate multiple rocket-data JSONL runs into one deduplicated dataset.

Three passes, all content-based (independent of the per-file `id`):
  1. Exact / sim-equivalent dedup — drop records whose input parameters are
     identical after rounding each field to a precision finer than the
     simulator could resolve (catches true re-draws, resumes, concatenation).
  2. Near-duplicate dedup — within each categorical group (same diameter, nose,
     fin count, motor class), drop a design that lies within --near-dist of a
     design already kept, in min-max-normalized continuous-parameter space.
     Distance is Euclidean / sqrt(n_dims), so 0 = identical and ~1 = opposite
     corner of the design box. Default 0.05 is conservative (removes only the
     closest twins); raising it removes more but eventually deletes legitimately
     distinct samples — measure before trusting a large value.
  3. Re-id — the merged records are renumbered 0..N-1 so ids are unique again.

Usage:
    python consolidate_dataset.py \
        --inputs outputs/rocket_data_2k_v2.jsonl outputs/rocket_data_5k.jsonl \
                 outputs/rocket_data_5k_s2026.jsonl outputs/rocket_data_10k_s2027.jsonl \
        --output outputs/rocket_data_all.jsonl
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

# Categorical inputs: must match exactly for two designs to be "the same".
CAT_FIELDS = ["diameter_mm", "nose_type", "fin_count", "motor_class"]

# Continuous inputs and the decimal precision used for the exact/sim-equivalent
# fingerprint (finer than the simulator resolves, so equal fingerprint == same).
CONT_PRECISION = {
    "length_m": 3, "nose_length_m": 3, "fin_root_chord_m": 3,
    "fin_tip_chord_m": 3, "fin_span_m": 3, "fin_sweep_m": 3,
    "fin_thickness_mm": 1, "dry_mass_kg": 2, "propellant_mass_kg": 3,
    "burn_time_s": 2, "avg_thrust_N": 0, "wind_speed_mps": 1,
    "wind_direction_deg": 0, "elevation_m": 0, "temperature_c": 1,
    "rail_length_m": 2, "launch_angle_deg": 1,
}
CONT_FIELDS = list(CONT_PRECISION)


def _exact_fingerprint(inp: dict) -> str:
    parts = [f"{k}={inp[k]}" for k in CAT_FIELDS]
    for k, prec in CONT_PRECISION.items():
        parts.append(f"{k}={round(float(inp[k]), prec)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def load_records(paths):
    recs, per_file = [], {}
    for path in paths:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line
                rec["_source"] = os.path.basename(path)
                recs.append(rec)
                n += 1
        per_file[path] = n
    return recs, per_file


def dedup_exact(recs):
    seen, kept, dropped = set(), [], 0
    for r in recs:
        fp = _exact_fingerprint(r["input"])
        if fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        kept.append(r)
    return kept, dropped


def dedup_near(recs, near_dist):
    """Greedy: keep a record unless it is within `near_dist` of one already kept
    (same categorical group, normalized continuous space)."""
    if near_dist <= 0:
        return recs, 0

    X = np.array([[float(r["input"][k]) for k in CONT_FIELDS] for r in recs],
                 dtype=float)
    mn, mx = X.min(0), X.max(0)
    span = np.where(mx > mn, mx - mn, 1.0)
    Xn = (X - mn) / span
    scale = np.sqrt(len(CONT_FIELDS))

    groups = defaultdict(list)
    for i, r in enumerate(recs):
        groups[tuple(r["input"][k] for k in CAT_FIELDS)].append(i)

    keep_mask = np.ones(len(recs), dtype=bool)
    for idxs in groups.values():
        kept_vecs = []
        for i in idxs:
            v = Xn[i]
            if kept_vecs:
                d = np.sqrt(((np.array(kept_vecs) - v) ** 2).sum(1)).min() / scale
                if d < near_dist:
                    keep_mask[i] = False
                    continue
            kept_vecs.append(v)

    kept = [r for r, m in zip(recs, keep_mask) if m]
    return kept, int((~keep_mask).sum())


def main():
    ap = argparse.ArgumentParser(description="Consolidate + dedupe rocket datasets.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--near-dist", type=float, default=0.05,
                    help="normalized near-duplicate threshold (0 disables; "
                         "default 0.05 is conservative)")
    ap.add_argument("--keep-source", action="store_true",
                    help="keep the _source field in the output records")
    args = ap.parse_args()

    recs, per_file = load_records(args.inputs)
    print(f"Loaded {len(recs)} records from {len(args.inputs)} files:")
    for path, n in per_file.items():
        print(f"  {n:>6}  {path}")

    recs, n_exact = dedup_exact(recs)
    print(f"\nExact / sim-equivalent dedup: removed {n_exact} "
          f"-> {len(recs)} remain")

    recs, n_near = dedup_near(recs, args.near_dist)
    print(f"Near-duplicate dedup (dist<{args.near_dist}): removed {n_near} "
          f"-> {len(recs)} remain")

    # Re-id and write.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for new_id, r in enumerate(recs):
            out = {"id": new_id, "input": r["input"], "output": r["output"]}
            if args.keep_source:
                out["_source"] = r["_source"]
            f.write(json.dumps(out) + "\n")

    meta_path = args.output.replace(".jsonl", "_metadata.json")
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "consolidated_from": {os.path.basename(p): n for p, n in per_file.items()},
        "total_input_records": sum(per_file.values()),
        "removed_exact_duplicates": n_exact,
        "removed_near_duplicates": n_near,
        "near_dist_threshold": args.near_dist,
        "final_count": len(recs),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {len(recs)} unique records -> {args.output}")
    print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
