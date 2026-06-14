#!/usr/bin/env python3
"""Memory benchmark for the data-generation pipeline.

Proves the overnight OOM by sampling RSS (and live thread count) over a short
batch of simulations, for two architectures:

  --mode baseline : current design — sims run sequentially in ONE long-lived
                    process; per-sim timeout is thread-based (threads cannot be
                    killed, so timed-out RocketPy solves keep running and
                    accumulate). This mirrors a Pool worker that is never
                    recycled (no maxtasksperchild).

  --mode fixed    : new design — sims run in child processes via
                    multiprocessing.Pool(maxtasksperchild=N); on overrun the
                    child is killed so the OS reclaims everything. RSS measured
                    across the whole process tree (parent + workers).

Both modes draw the SAME deterministic, pre-validated parameter batch so the
before/after comparison is apples-to-apples. We report peak/final RSS of the
*process tree* in each case — that is what actually OOMs the machine.

Usage:
    python bench_memory.py --mode baseline --n 50 --out outputs/bench_baseline.csv
    python bench_memory.py --mode fixed    --n 50 --out outputs/bench_fixed.csv

Optional: --force-timeout S  shrinks every per-class timeout to S seconds so
larger-class solves overrun, exposing the thread-leak mechanism quickly.
"""

import argparse
import os
import sys
import time
import threading

import psutil

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "rocket_sim")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import config as cfg          # noqa: E402
import parameters as params_mod  # noqa: E402
import validator              # noqa: E402


def build_batch(n: int, seed: int = 42):
    """Deterministic, pre-validated parameter batch shared by both modes."""
    raw = params_mod.balanced_sample(n * 6, seed)
    out = []
    for p in raw:
        ok, _ = validator.prevalidate(p)
        if ok:
            out.append(p)
        if len(out) >= n:
            break
    return out


def tree_rss_mb(proc: psutil.Process) -> float:
    """RSS of proc + all live children, in MiB."""
    total = 0
    try:
        total += proc.memory_info().rss
    except psutil.Error:
        pass
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total / (1024 * 1024)


def run_baseline(batch, out_csv, force_timeout):
    """Sequential, in-process — mimics a never-recycled worker."""
    import simulator
    import rocket_builder
    import outputs

    if force_timeout is not None:
        cfg.SIM_TIMEOUT_S = force_timeout
        cfg.SIM_TIMEOUT_BY_CLASS = {k: force_timeout for k in cfg.SIM_TIMEOUT_BY_CLASS}

    me = psutil.Process()
    rows = [("i", "t_s", "rss_mb", "threads", "valid")]
    t0 = time.time()
    base_threads = threading.active_count()
    rss0 = tree_rss_mb(me)
    rows.append((0, 0.0, round(rss0, 1), base_threads, ""))
    print(f"  start: RSS={rss0:.1f} MiB, threads={base_threads}")

    valid = 0
    for i, p in enumerate(batch, 1):
        ok = False
        try:
            rocket = rocket_builder.build_rocket(p)
            flight = simulator.run_simulation(rocket, p)
            if flight is not None and validator.is_valid(p, flight):
                outputs.extract_output(p, flight)
                outputs.extract_input(p)
                ok = True
        except Exception:
            pass
        if ok:
            valid += 1
        rss = tree_rss_mb(me)
        rows.append((i, round(time.time() - t0, 2), round(rss, 1),
                     threading.active_count(), int(ok)))
        if i % 5 == 0 or i == len(batch):
            print(f"  {i:3d}/{len(batch)}  RSS={rss:7.1f} MiB  "
                  f"threads={threading.active_count():3d}  valid={valid}")
    _write_csv(out_csv, rows)
    _summary(rows)


def run_fixed(batch, out_csv, force_timeout, maxtasks, workers, per_sim_timeout):
    """Process-isolated pool with recycling + hard-kill timeout.

    Samples tree RSS from the parent while the pool churns through the batch.
    """
    import gen_worker  # new module (process-isolated worker)

    if force_timeout is not None:
        per_sim_timeout = force_timeout

    me = psutil.Process()
    rows = [("i", "t_s", "rss_mb", "threads", "valid")]
    t0 = time.time()
    rss0 = tree_rss_mb(me)
    rows.append((0, 0.0, round(rss0, 1), threading.active_count(), ""))
    print(f"  start: RSS={rss0:.1f} MiB")

    done = 0
    valid = 0
    for _idx, res in gen_worker.run_batch(
        batch, workers=workers, maxtasksperchild=maxtasks,
        per_sim_timeout=per_sim_timeout, force_timeout=force_timeout,
    ):
        done += 1
        ok = res is not None
        if ok:
            valid += 1
        rss = tree_rss_mb(me)
        rows.append((done, round(time.time() - t0, 2), round(rss, 1),
                     threading.active_count(), int(bool(ok))))
        if done % 5 == 0 or done == len(batch):
            print(f"  {done:3d}/{len(batch)}  RSS(tree)={rss:7.1f} MiB  valid={valid}")
    _write_csv(out_csv, rows)
    _summary(rows)


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"  wrote {path}")


def _slope_mib_per_sim(idxs, vals):
    """Least-squares slope of RSS vs sim index (MiB per sim)."""
    k = len(idxs)
    if k < 2:
        return 0.0
    mx = sum(idxs) / k
    my = sum(vals) / k
    num = sum((x - mx) * (y - my) for x, y in zip(idxs, vals))
    den = sum((x - mx) ** 2 for x in idxs)
    return num / den if den else 0.0


def _summary(rows):
    data = rows[1:]
    idxs = [r[0] for r in data]
    rss_vals = [r[2] for r in data]
    thr_vals = [r[3] for r in data]
    rss_start, rss_end, rss_peak = rss_vals[0], rss_vals[-1], max(rss_vals)

    # Steady-state slope over the second half — ignores worker-pool warmup so it
    # measures genuine leak rate, not one-time interpreter spin-up cost.
    half = len(data) // 2
    ss_idxs, ss_vals = idxs[half:], rss_vals[half:]
    slope = _slope_mib_per_sim(ss_idxs, ss_vals)

    print(f"\n  --- summary ---")
    print(f"  RSS start          : {rss_start:.1f} MiB")
    print(f"  RSS end            : {rss_end:.1f} MiB")
    print(f"  RSS peak           : {rss_peak:.1f} MiB")
    print(f"  RSS growth (raw)   : {rss_end - rss_start:+.1f} MiB "
          f"({100*(rss_end-rss_start)/max(rss_start,1):+.0f}%)")
    print(f"  steady-state slope : {slope:+.2f} MiB/sim "
          f"(2nd half, sims {ss_idxs[0]}-{ss_idxs[-1]})")
    print(f"  projected +10k sims: {slope*10000:+.0f} MiB at steady-state rate")
    print(f"  threads            : {thr_vals[0]} -> {thr_vals[-1]} (peak {max(thr_vals)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "fixed"], required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--force-timeout", type=float, default=None,
                    help="shrink every per-sim timeout to S seconds to force overruns")
    ap.add_argument("--maxtasks", type=int, default=10, help="(fixed) maxtasksperchild")
    ap.add_argument("--workers", type=int, default=4, help="(fixed) worker processes")
    ap.add_argument("--per-sim-timeout", type=float, default=None,
                    help="(fixed) override per-sim timeout seconds")
    args = ap.parse_args()

    out = args.out or f"outputs/bench_{args.mode}.csv"
    print(f"=== bench_memory mode={args.mode} n={args.n} "
          f"force_timeout={args.force_timeout} ===")
    batch = build_batch(args.n, args.seed)
    print(f"  built deterministic batch: {len(batch)} pre-validated params")

    if args.mode == "baseline":
        run_baseline(batch, out, args.force_timeout)
    else:
        run_fixed(batch, out, args.force_timeout, args.maxtasks,
                  args.workers, args.per_sim_timeout)


if __name__ == "__main__":
    main()
