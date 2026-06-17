#!/usr/bin/env python3
"""Backfill the ``within_bounds`` computability label onto existing datasets.

Every record already in our JSONL files is a design the simulator resolved
successfully, so by definition it is within bounds. This stamps
``output.within_bounds = True`` on each record (idempotent — records that
already carry the field are left unchanged) so the existing corpus matches the
schema going forward, ahead of generating the not-computable (False) class.

Usage:
    python backfill_within_bounds.py outputs/rocket_data_all.jsonl \
                                      outputs/rocket_data_10k_s2028.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "common"))
import schema  # noqa: E402

FIELD = schema.WITHIN_BOUNDS_FIELD


def backfill(path: str) -> tuple:
    """Rewrite ``path`` in place, adding within_bounds=True where missing.
    Returns (total, added)."""
    total = added = 0
    out_lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            out = rec.get("output")
            if isinstance(out, dict) and FIELD not in out:
                out[FIELD] = True
                added += 1
            out_lines.append(json.dumps(rec))

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    os.replace(tmp, path)  # atomic on the same volume
    return total, added


def main():
    ap = argparse.ArgumentParser(description="Stamp within_bounds=True on existing data.")
    ap.add_argument("files", nargs="+", help="JSONL dataset file(s) to label in place")
    args = ap.parse_args()

    for path in args.files:
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue
        total, added = backfill(path)
        print(f"  {path}: {total} records, labelled {added} "
              f"({total - added} already had '{FIELD}')")


if __name__ == "__main__":
    main()
