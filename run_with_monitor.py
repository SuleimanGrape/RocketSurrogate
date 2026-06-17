#!/usr/bin/env python3
"""Resilient rocket data generator — periodic JSONL flush + system health monitoring.

Saves incremental JSONL every N valid samples so a crash only loses ~N records.
Logs CPU load, RAM, GPU vitals, and (if available) CPU temperatures via
LibreHardwareMonitor WMI to a companion _health.csv for post-crash diagnosis.

Usage:
    python run_with_monitor.py --count 5000 --seed 897 --workers 15
"""

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import cpu_count
from typing import Optional

import psutil

# ── Reuse existing rocket_sim modules ────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "rocket_sim"))
import config as cfg
import parameters as params_mod
import validator
import gen_worker  # process-isolated pool with hard-kill timeouts + recycling

# ── Global state (accessible from signal handler) ────────────────────────────
_output_file = None
_results_buffer = []
_total_simmed = 0
_total_valid = 0
_target_count = 0
_start_time = 0.0
_results_lock = threading.Lock()
_monitor = None  # SystemHealthMonitor instance

# CLI globals for metadata
_seed = 42
_method = "random"
_balanced = True


# =============================================================================
#  Duplicate exclusion (content fingerprint — matches consolidate_dataset.py)
# =============================================================================

_FP_CAT_FIELDS = ["diameter_mm", "nose_type", "fin_count", "motor_class"]
_FP_CONT_PRECISION = {
    "length_m": 3, "nose_length_m": 3, "fin_root_chord_m": 3,
    "fin_tip_chord_m": 3, "fin_span_m": 3, "fin_sweep_m": 3,
    "fin_thickness_mm": 1, "dry_mass_kg": 2, "propellant_mass_kg": 3,
    "burn_time_s": 2, "avg_thrust_N": 0, "wind_speed_mps": 1,
    "wind_direction_deg": 0, "elevation_m": 0, "temperature_c": 1,
    "rail_length_m": 2, "launch_angle_deg": 1,
}


def _design_fingerprint(inp: dict) -> str:
    """Sim-equivalent fingerprint of a design's input parameters."""
    parts = [f"{k}={inp[k]}" for k in _FP_CAT_FIELDS]
    for k, prec in _FP_CONT_PRECISION.items():
        parts.append(f"{k}={round(float(inp[k]), prec)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _load_exclude_fingerprints(paths) -> set:
    """Collect input fingerprints from prior dataset JSONL files to skip."""
    fps = set()
    for path in paths or []:
        if not os.path.exists(path):
            print(f"  WARNING: exclude file not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inp = rec.get("input")
                if inp:
                    fps.add(_design_fingerprint(inp))
    return fps


# =============================================================================
#  System Health Monitor
# =============================================================================

LHM_NAMESPACE = "root\\LibreHardwareMonitor"


def _query_libre_hardware_monitor() -> dict:
    """Query LibreHardwareMonitor via WMI for CPU/GPU/motherboard sensor data.

    Returns a dict with keys like
        "cpu_package_temp", "cpu_max_core_temp", "cpu_total_load", etc.
    Returns empty dict if not available.
    """
    sensors = {
        "cpu_package_temp": [],
        "cpu_core_max_temp": [],
        "gpu_core_temp": [],
        "cpu_total_load": [],
        "fan_rpm": [],
    }
    try:
        script = fr"""
$sensors = Get-CimInstance -Namespace "{LHM_NAMESPACE}" -ClassName Sensor -ErrorAction SilentlyContinue
if (-not $sensors) {{ exit }}
foreach ($s in $sensors) {{
    $parent = $s.Parent | Out-String
    if ($s.SensorType -eq "Temperature") {{
        if ($s.Name -match "(CPU Package|CPU Die|Tctl|Tdie)") {{ Write-Output "CPU_PACKAGE_TEMP=$($s.Value)" }}
        elseif ($s.Name -match "(CPU Core #)") {{ Write-Output "CPU_CORE_TEMP=$($s.Value)" }}
        elseif ($s.Name -match "(GPU Core|GPU)") {{ Write-Output "GPU_CORE_TEMP=$($s.Value)" }}
    }}
    elseif ($s.SensorType -eq "Load" -and $s.Name -match "(CPU Total)") {{ Write-Output "CPU_TOTAL_LOAD=$($s.Value)" }}
    elseif ($s.SensorType -eq "Fan") {{ Write-Output "FAN_RPM=$($s.Name)=$($s.Value)" }}
}}
"""
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("CPU_PACKAGE_TEMP="):
                sensors["cpu_package_temp"].append(float(line.split("=", 1)[1]))
            elif line.startswith("CPU_CORE_TEMP="):
                sensors["cpu_core_max_temp"].append(float(line.split("=", 1)[1]))
            elif line.startswith("GPU_CORE_TEMP="):
                sensors["gpu_core_temp"].append(float(line.split("=", 1)[1]))
            elif line.startswith("CPU_TOTAL_LOAD="):
                sensors["cpu_total_load"].append(float(line.split("=", 1)[1]))
            elif line.startswith("FAN_RPM="):
                sensors["fan_rpm"].append(line.split("=", 1)[1])
    except Exception:
        pass

    out = {}
    if sensors["cpu_package_temp"]:
        out["lhm_cpu_package_temp_c"] = round(max(sensors["cpu_package_temp"]), 1)
    if sensors["cpu_core_max_temp"]:
        out["lhm_cpu_max_core_temp_c"] = round(max(sensors["cpu_core_max_temp"]), 1)
    if sensors["gpu_core_temp"]:
        out["lhm_gpu_temp_c"] = round(max(sensors["gpu_core_temp"]), 1)
    if sensors["cpu_total_load"]:
        out["lhm_cpu_load_pct"] = round(max(sensors["cpu_total_load"]), 1)
    if sensors["fan_rpm"]:
        out["lhm_fans"] = ", ".join(sensors["fan_rpm"])
    return out


def _query_nvidia_smi() -> dict:
    """Query nvidia-smi for GPU vitals. Returns empty dict on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,"
             "utilization.memory,power.draw,power.limit,"
             "clocks.current.graphics,clocks.current.memory,fan.speed",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        parts = [p.strip() for p in result.stdout.strip().split(", ")]
        if len(parts) < 8:
            return {}
        return {
            "gpu_temp_c": parts[0],
            "gpu_util_pct": parts[1],
            "gpu_mem_util_pct": parts[2],
            "gpu_power_w": parts[3],
            "gpu_power_limit_w": parts[4],
            "gpu_core_clock_mhz": parts[5],
            "gpu_mem_clock_mhz": parts[6],
            "gpu_fan_speed_pct": parts[7],
        }
    except Exception:
        return {}


class SystemHealthMonitor:
    """Daemon thread sampling CPU, RAM, GPU, and WMI sensors to a CSV log."""

    def __init__(self, csv_path: str, interval: float = 5.0):
        self.csv_path = csv_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file = None
        self._writer = None
        self._last_disk = psutil.disk_io_counters()

    def start(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)

        # Probe for LibreHardwareMonitor once to note if available
        lhm_test = _query_libre_hardware_monitor()
        self._lhm_available = bool(lhm_test)

        fields = [
            "timestamp_utc", "elapsed_s",
            "cpu_percent", "cpu_per_core_json",
            "ram_percent", "ram_used_gb", "ram_total_gb",
            "disk_read_mb", "disk_write_mb",
            "gpu_temp_c", "gpu_util_pct", "gpu_mem_util_pct",
            "gpu_power_w", "gpu_power_limit_w",
            "gpu_core_clock_mhz", "gpu_mem_clock_mhz", "gpu_fan_speed_pct",
        ]
        if self._lhm_available:
            fields += [
                "lhm_cpu_package_temp_c", "lhm_cpu_max_core_temp_c",
                "lhm_cpu_load_pct", "lhm_fans",
            ]

        self._writer.writerow(fields)
        self._file.flush()
        if self._lhm_available:
            print(f"  LibreHardwareMonitor WMI detected — CPU temps enabled")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._file and not self._file.closed:
            self._file.close()

    def _run(self):
        t0 = time.time()
        while not self._stop.wait(self.interval):
            elapsed = time.time() - t0
            try:
                cpu_pct = psutil.cpu_percent(interval=0.2)
                cpu_per_core = psutil.cpu_percent(interval=0.0, percpu=True)
            except Exception:
                cpu_pct = ""
                cpu_per_core = []
            try:
                mem = psutil.virtual_memory()
                ram_pct = mem.percent
                ram_used = round(mem.used / 1e9, 2)
                ram_total = round(mem.total / 1e9, 1)
            except Exception:
                ram_pct = ram_used = ram_total = ""

            # Disk delta
            try:
                disk = psutil.disk_io_counters()
                disk_read = round((disk.read_bytes - self._last_disk.read_bytes) / 1e6, 1)
                disk_write = round((disk.write_bytes - self._last_disk.write_bytes) / 1e6, 1)
                self._last_disk = disk
            except Exception:
                disk_read = disk_write = ""

            gpu = _query_nvidia_smi()
            lhm = _query_libre_hardware_monitor() if self._lhm_available else {}

            row = [
                datetime.now(timezone.utc).isoformat(),
                round(elapsed, 1),
                cpu_pct,
                json.dumps(cpu_per_core) if cpu_per_core else "",
                ram_pct, ram_used, ram_total,
                disk_read, disk_write,
                gpu.get("gpu_temp_c", ""),
                gpu.get("gpu_util_pct", ""),
                gpu.get("gpu_mem_util_pct", ""),
                gpu.get("gpu_power_w", ""),
                gpu.get("gpu_power_limit_w", ""),
                gpu.get("gpu_core_clock_mhz", ""),
                gpu.get("gpu_mem_clock_mhz", ""),
                gpu.get("gpu_fan_speed_pct", ""),
            ]
            if self._lhm_available:
                row += [
                    lhm.get("lhm_cpu_package_temp_c", ""),
                    lhm.get("lhm_cpu_max_core_temp_c", ""),
                    lhm.get("lhm_cpu_load_pct", ""),
                    lhm.get("lhm_fans", ""),
                ]
            try:
                self._writer.writerow(row)
                self._file.flush()
            except Exception:
                pass


# =============================================================================
#  Signal handler
# =============================================================================

def _handle_signal(signum, frame):
    """SIGINT handler: flush remaining data and exit cleanly."""
    global _output_file, _results_buffer, _total_simmed, _total_valid
    print(f"\n\n=== Interrupted after {_total_simmed} sims ({_total_valid} valid) ===")
    _flush_results()
    _write_partial_metadata()
    if _monitor:
        _monitor.stop()
    print("Partial data saved. Exiting.")
    sys.exit(0)


# =============================================================================
#  Generator with incremental write
# =============================================================================

def _flush_results():
    """Flush buffered results to JSONL + fsync."""
    global _output_file, _results_buffer, _total_valid
    with _results_lock:
        if not _results_buffer:
            return
        if _output_file and not _output_file.closed:
            for record in _results_buffer:
                _output_file.write(json.dumps(record) + "\n")
            _output_file.flush()
            os.fsync(_output_file.fileno())
            n = len(_results_buffer)
            _results_buffer = []
            elapsed = time.time() - _start_time
            rate = _total_simmed / elapsed if elapsed > 0 else 0
            print(f"  Flushed {n} records ({_total_valid}/{_target_count} valid, "
                  f"{_total_simmed} sims, {rate:.0f} sims/s)")


def _init_worker():
    import warnings
    warnings.filterwarnings("ignore")


def _write_partial_metadata():
    """Write metadata after interrupt signal."""
    global _output_file, _seed, _method, _balanced
    meta_path = _output_file.name.replace(".jsonl", "_metadata.json") if _output_file else ""
    if meta_path:
        meta = {
            "generation_seed": _seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": _method,
            "balanced": _balanced,
            "target_count": _target_count,
            "actual_count": _total_valid,
            "completed": False,
            "note": "Interrupted — partial dataset",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Partial metadata saved to {meta_path}")


def run_with_monitor(
    count: int = 10000,
    method: str = "random",
    seed: int = 42,
    output: str = "rocket_data.jsonl",
    workers: int = 1,
    splits_dir: Optional[str] = None,
    plots_dir: Optional[str] = None,
    balanced: bool = True,
    oversample_factor: float = 5.0,
    monitor_interval: float = 5.0,
    flush_every: int = 100,
    exclude: Optional[list] = None,
):
    global _output_file, _results_buffer, _total_simmed, _total_valid
    global _target_count, _start_time, _monitor, _seed, _method, _balanced

    _seed = seed
    _method = method
    _balanced = balanced
    _target_count = count

    # ── Launch health monitor ────────────────────────────────────────────────
    health_path = output.replace(".jsonl", "_health.csv")
    _monitor = SystemHealthMonitor(health_path, interval=monitor_interval)
    _monitor.start()
    print(f"  Health log   : {health_path}")

    print(f"=== Rocket Data Generator (resilient) ===")
    print(f"  Target       : {count}")
    print(f"  Method       : {method}")
    print(f"  Seed         : {seed}")
    print(f"  Workers      : {workers}")
    print(f"  Output       : {output}")
    print(f"  Flush every  : {flush_every} records")
    print(f"  Monitor int  : {monitor_interval}s")
    print()

    # ── Step 1: Sample parameters ────────────────────────────────────────────
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
    print(f"  Sampled {len(param_list)} sets in {time.time()-t0:.1f}s")

    # ── Step 1b: Load duplicate-exclusion fingerprints ───────────────────────
    # Skip any design already present in prior dataset files (--exclude) and
    # drop intra-run exact/sim-equivalent collisions, so a new seed never
    # re-adds a design already in the corpus.
    exclude_fps = _load_exclude_fingerprints(exclude)
    if exclude:
        print(f"  Excluding {len(exclude_fps)} fingerprints from "
              f"{len(exclude)} prior file(s)")
    seen_fps = set(exclude_fps)

    # ── Step 2: Pre-validation ───────────────────────────────────────────────
    print("[2/5] Pre-validating parameters...")
    t0 = time.time()
    pre_valids = []
    pre_reject_reasons = Counter()
    n_dup_skipped = 0
    for p in param_list:
        fp = _design_fingerprint(p)
        if fp in seen_fps:
            n_dup_skipped += 1
            continue
        seen_fps.add(fp)
        ok, reason = validator.prevalidate(p)
        if ok:
            pre_valids.append(p)
        else:
            pre_reject_reasons[reason] += 1
    if n_dup_skipped:
        print(f"  Skipped {n_dup_skipped} duplicate designs "
              f"(already in corpus or sampled twice)")
    print(f"  Pre-validated in {time.time()-t0:.1f}s")
    print(f"  Passed: {len(pre_valids)}/{len(param_list)} "
          f"({100*len(pre_valids)/len(param_list):.1f}%)")
    if pre_reject_reasons:
        print(f"  Top rejections:")
        for reason, cnt in pre_reject_reasons.most_common(5):
            print(f"    {reason}: {cnt}")

    if not pre_valids:
        print("ERROR: No designs passed pre-validation.")
        _monitor.stop()
        return

    max_to_sim = min(len(pre_valids), int(count * 3))
    param_list_prevalidated = pre_valids[:max_to_sim]

    # ── Step 3: Run simulations (process-isolated, incremental save, resumable) ─
    # Each sim runs in a child process that the parent kills on overrun, so a
    # hung/leaky RocketPy solve can never accumulate across a long overnight run.
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    # Resume: skip indices already present in the output file from a prior run.
    done_ids = set()
    if os.path.exists(output):
        with open(output, encoding="utf-8") as f:
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
        if done_ids:
            print(f"  Resume: {len(done_ids)} records already in {output}; skipping those")

    _output_file = open(output, "a", encoding="utf-8")

    print("[3/5] Running simulations (incremental save)...")
    _start_time = time.time()
    _total_simmed = 0
    _total_valid = len(done_ids)
    sim_rejected = 0

    signal.signal(signal.SIGINT, _handle_signal)

    actual_workers = max(1, min(workers, cpu_count() - 1))

    tasks = [(i, p) for i, p in enumerate(param_list_prevalidated) if i not in done_ids]
    task_ids = [i for i, _ in tasks]
    task_params = [p for _, p in tasks]

    if _total_valid < count and task_params:
        for local_idx, r in gen_worker.run_batch(
            task_params,
            workers=actual_workers,
            maxtasksperchild=cfg.MAXTASKSPERCHILD,
        ):
            _total_simmed += 1
            if r is not None:
                with _results_lock:
                    _results_buffer.append({"id": task_ids[local_idx], **r})
                _total_valid += 1
            else:
                sim_rejected += 1
            # Periodic flush
            if len(_results_buffer) >= flush_every:
                _flush_results()
            if _total_valid >= count:
                break
            if _total_simmed % 50 == 0:
                elapsed = time.time() - _start_time
                rate = _total_simmed / elapsed if elapsed > 0 else 0
                print(f"  {_total_simmed} sims, {_total_valid} valid, "
                      f"{sim_rejected} rejected ({rate:.0f} sims/s)")

    t_sim = time.time() - _start_time

    # Final flush
    _flush_results()
    _output_file.close()

    print(f"  Completed in {t_sim:.1f}s — {_total_valid} valid "
          f"({sim_rejected} rejections)")

    if _total_valid == 0:
        print("ERROR: No valid designs.")
        _monitor.stop()
        return

    # ── Step 4: Reload & truncate ────────────────────────────────────────────
    all_results = []
    with open(output, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))
    if len(all_results) > count:
        all_results = all_results[:count]
        with open(output, "w") as f:
            for rec in all_results:
                f.write(json.dumps(rec) + "\n")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta_path = output.replace(".jsonl", "_metadata.json")
    meta = {
        "generation_seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "balanced": balanced,
        "rocketpy_version": "1.12.1",
        "target_count": count,
        "actual_count": len(all_results),
        "completed": True,
        "sampling": {
            "n_sampled": n_to_sample,
            "n_prevalidated": len(pre_valids),
            "n_simulated": len(param_list_prevalidated),
            "prevalidation_rate": round(len(pre_valids) / max(1, n_to_sample), 3),
            "simulation_rate": round(len(all_results) / max(1, len(param_list_prevalidated)), 3),
        },
        "dedup": {
            "exclude_files": exclude or [],
            "excluded_fingerprints": len(exclude_fps),
            "duplicates_skipped": n_dup_skipped,
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
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata     : {meta_path}")

    print("\n  Dataset balance:")
    for key in ["diameter_mm", "nose_type", "fin_count", "motor_class"]:
        counter = Counter(r["input"][key] for r in all_results)
        print(f"    {key}: {dict(sorted(counter.items()))}")

    if splits_dir:
        print(f"\n[4/5] Creating train/val/test splits...")
        try:
            from splitter import split_dataset
            split_dataset(all_results, splits_dir, train_frac=0.7, val_frac=0.15, test_frac=0.15)
        except Exception as e:
            print(f"  Splits skipped: {e}")

    if plots_dir:
        print(f"[5/5] Generating summary plots...")
        try:
            from plotter import generate_plots
            generate_plots(all_results, plots_dir)
        except Exception as e:
            print(f"  Plots skipped: {e}")

    _monitor.stop()
    print(f"\n=== Done ===")
    print(f"  Health log  : {health_path}")


# =============================================================================
#  CLI
# =============================================================================

def main():
    global _seed, _method, _balanced

    parser = argparse.ArgumentParser(
        description="Resilient rocket data generator with health monitoring.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--method", choices=["random", "lhs", "sobol"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="rocket_data.jsonl")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--plots-dir", type=str, default=None)
    parser.add_argument("--no-balanced", action="store_true")
    parser.add_argument("--oversample", type=float, default=5.0)
    parser.add_argument("--monitor-interval", type=float, default=5.0)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--exclude", nargs="+", default=None,
                        help="prior dataset JSONL file(s) whose designs to skip "
                             "(prevents cross-run duplicates)")
    args = parser.parse_args()

    _seed = args.seed
    _method = args.method
    _balanced = not args.no_balanced

    run_with_monitor(
        count=args.count, method=args.method, seed=args.seed,
        output=args.output, workers=args.workers,
        splits_dir=args.splits_dir, plots_dir=args.plots_dir,
        balanced=not args.no_balanced,
        oversample_factor=args.oversample,
        monitor_interval=args.monitor_interval,
        flush_every=args.flush_every,
        exclude=args.exclude,
    )


if __name__ == "__main__":
    main()