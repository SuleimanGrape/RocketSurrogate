"""Gradient-quality evaluation for the differentiable surrogate.

For gradient-based design optimization, *values* being accurate is not enough —
the surrogate's gradient d(metric)/d(design) must point the right way. This tool
compares the NN's autograd gradients against ground truth:

  Class 1 (cg_m, cp_m, stability_margin_calibers) — closed-form Barrowman.
      Ground-truth gradient is the analytic derivative, obtained by autograd
      through the exact torch Barrowman (optim.diff_features), to machine
      precision. Cheap, runs on a large sample.

  Class 2 (apogee, velocities, accel, pressure, ...) — 6-DOF flight outputs.
      Ground-truth gradient is a central finite-difference of the actual RocketPy
      simulator (--with-sim). Expensive (≈2·17 solves per design), so it runs on
      a small sample.

Metrics per target: mean cosine similarity between NN and true gradient vectors
(direction — what the optimizer follows), median relative L2 error (magnitude),
and the surrogate's value accuracy (R²/MAE) on the same sample for reference.

Usage:
    python eval_gradients.py --bundle ../../models/neural --data ../../outputs/rocket_data_full.jsonl
    python eval_gradients.py --bundle ../../models/neural --data ... --with-sim --n-sim 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# rocket_sim FIRST so the bare `utils` name resolves to rocket_sim/utils (the
# Barrowman module) and not neural_surrogate/utils — see dataset.py / the
# importlib note in train_surrogate.py for the same collision.
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
sys.path.insert(0, os.path.join(_HERE, "..", "rocket_sim"))

import torch                              # noqa: E402
import schema                             # noqa: E402
from dataio import load_jsonl            # noqa: E402

from optim.diff_surrogate import DifferentiableSurrogate  # noqa: E402
from optim.diff_features import continuous_block, CONTINUOUS_NAMES, CONT  # noqa: E402

CLASS1 = {  # target name -> exact engineered column it equals
    "cg_m": "barrowman_cg_m",
    "cp_m": "barrowman_cp_m",
    "stability_margin_calibers": "barrowman_margin_cal",
}
CLASS2 = [  # 6-DOF flight-dynamics targets (need the simulator for ground truth)
    "apogee_m", "max_velocity_mps", "max_mach", "max_acceleration_mps2",
    "burnout_altitude_m", "burnout_velocity_mps", "time_to_apogee_s",
    "rail_exit_velocity_mps", "max_dynamic_pressure_pa",
]


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_designs(path, n, seed, motor_filter=None):
    recs = [r for r in load_jsonl(path)
            if r.get("output", {}).get("within_bounds") is not False]
    if motor_filter:
        recs = [r for r in recs if r["input"].get("motor_class") in motor_filter]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))[:n]
    recs = [recs[i] for i in idx]
    cont = np.array([[float(r["input"][k]) for k in schema.INPUT_CONTINUOUS]
                     for r in recs], dtype=np.float64)
    cat = np.array([[schema.ENCODING_MAPS[k][r["input"][k]]
                     for k in schema.INPUT_CATEGORICAL] for r in recs], dtype=np.int64)
    return recs, cont, cat


def _cosine(a, b, axis=-1, eps=1e-12):
    num = (a * b).sum(axis=axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    return num / np.maximum(den, eps)


def _rel_l2(a, b, axis=-1, eps=1e-12):
    return np.linalg.norm(a - b, axis=axis) / np.maximum(np.linalg.norm(b, axis=axis), eps)


def _r2(pred, true):
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# ── Class 1: analytic Barrowman ground truth ──────────────────────────────────
def eval_class1(surr, cont, cat):
    x = torch.tensor(cont, dtype=torch.float64, requires_grad=True)
    xc = torch.tensor(cat, dtype=torch.long)

    # exact (ground-truth) gradients: autograd through the exact torch Barrowman
    block = continuous_block(x, xc)
    true_grads = {}
    for tname, col in CLASS1.items():
        ci = CONTINUOUS_NAMES.index(col)
        g = torch.autograd.grad(block[:, ci].sum(), x, retain_graph=True)[0]
        true_grads[tname] = g.detach().numpy()
    true_vals = {t: block[:, CONTINUOUS_NAMES.index(c)].detach().numpy()
                 for t, c in CLASS1.items()}

    # NN gradients + values
    J = surr.jacobian(torch.tensor(cont, dtype=torch.float32),
                      torch.tensor(cat, dtype=torch.long)).detach().numpy()  # (B,12,17)
    nn_vals = surr.predict(torch.tensor(cont, dtype=torch.float32),
                           torch.tensor(cat, dtype=torch.long)).numpy()

    results = {}
    for tname in CLASS1:
        ti = surr.target_index(tname)
        g_nn = J[:, ti, :]
        g_true = true_grads[tname]
        results[tname] = {
            "grad_cosine_mean": float(np.mean(_cosine(g_nn, g_true))),
            "grad_cosine_median": float(np.median(_cosine(g_nn, g_true))),
            "grad_rel_l2_median": float(np.median(_rel_l2(g_nn, g_true))),
            "value_r2": _r2(nn_vals[:, ti], true_vals[tname]),
            "value_mae": float(np.mean(np.abs(nn_vals[:, ti] - true_vals[tname]))),
        }
    return results


# ── Class 2: finite-difference of the simulator ───────────────────────────────
_SIM = {}


def _ensure_sim_imports():
    """Import the rocket_sim modules for FD. Re-inserts rocket_sim at the front of
    sys.path so the bare ``utils`` name binds to rocket_sim/utils.py (Barrowman)
    and not the neural_surrogate ``utils`` package that diff_surrogate put first."""
    if _SIM:
        return
    sys.path.insert(0, os.path.join(_HERE, "..", "rocket_sim"))
    import utils  # noqa: F401  binds the bare name to rocket_sim/utils first
    import outputs
    import rocket_builder
    import simulator
    import validator
    _SIM.update(outputs=outputs, rocket_builder=rocket_builder,
                simulator=simulator, validator=validator)


def _raw_metrics(flight, params):
    """Unrounded flight metrics (avoids extract_output's round() quantization)."""
    _o = _SIM["outputs"]

    def _safe(g, d=0.0):
        try:
            return float(g())
        except Exception:
            return d

    try:
        tt, alt, vel = flight.z.x_array, flight.z.y_array, flight.vz.y_array
    except Exception:
        tt = alt = vel = np.array([])
    b_alt, b_vel = _o._find_burnout_state(flight, params, tt, alt, vel)
    t_apogee = float(tt[np.argmax(alt)]) if len(tt) else 0.0
    try:
        max_mach = float(flight.max_mach_number)
    except Exception:
        max_mach = _safe(lambda: flight.max_speed) / 340.0
    return {
        "apogee_m": _safe(lambda: flight.apogee),
        "max_velocity_mps": _safe(lambda: flight.max_speed),
        "max_mach": max_mach,
        "max_acceleration_mps2": _safe(lambda: flight.max_acceleration),
        "burnout_altitude_m": b_alt,
        "burnout_velocity_mps": b_vel,
        "time_to_apogee_s": t_apogee,
        "rail_exit_velocity_mps": _safe(lambda: flight.out_of_rail_velocity),
        "max_dynamic_pressure_pa": _safe(lambda: flight.max_dynamic_pressure),
    }


def _sim_once(params):
    try:
        rocket = _SIM["rocket_builder"].build_rocket(params)
    except Exception:
        return None
    # run_simulation has the per-motor-class thread-join timeout; simulate_flight
    # does NOT, and a perturbed design can integrate for minutes. For a bounded
    # diagnostic the threaded timeout (daemon-thread caveat) is the right call.
    # NB: no is_valid() filter here — the FD ground truth is the simulator's true
    # output at the perturbed point, even if that point lies outside the dataset's
    # acceptance bounds (the optimizer may query the surrogate there too).
    flight = _SIM["simulator"].run_simulation(rocket, params)
    if flight is None:
        return None
    try:
        return _raw_metrics(flight, params)
    except Exception:
        return None


def eval_class2(surr, recs, cont, cat, rel_step, abs_floor):
    _ensure_sim_imports()
    # NN gradients + values for the whole sim sample
    J = surr.jacobian(torch.tensor(cont, dtype=torch.float32),
                      torch.tensor(cat, dtype=torch.long)).detach().numpy()
    nn_vals = surr.predict(torch.tensor(cont, dtype=torch.float32),
                           torch.tensor(cat, dtype=torch.long)).numpy()

    per_design = []
    for d, rec in enumerate(recs):
        base_params = dict(rec["input"])
        base = _sim_once(base_params)
        if base is None:
            print(f"  design {d}: baseline sim failed, skipping")
            continue
        # FD over the 17 continuous inputs (central difference). A perturbation
        # that fails to simulate just leaves that input's partial as NaN — the
        # other inputs still yield a usable gradient direction.
        fd = {t: np.full(len(CONT), np.nan) for t in CLASS2}
        n_ok = 0
        for k, key in enumerate(CONT):
            x0 = float(base_params[key])
            h = max(rel_step * abs(x0), abs_floor.get(key, rel_step))
            pp = dict(base_params); pp[key] = x0 + h
            pm = dict(base_params); pm[key] = x0 - h
            mp, mm = _sim_once(pp), _sim_once(pm)
            if mp is None or mm is None:
                continue
            for t in CLASS2:
                fd[t][k] = (mp[t] - mm[t]) / (2.0 * h)
            n_ok += 1
        if n_ok < 3:
            print(f"  design {d}: only {n_ok} input partials succeeded, skipping")
            continue
        entry = {"design_index": d, "n_input_partials": n_ok, "targets": {}}
        for t in CLASS2:
            ti = surr.target_index(t)
            g_true = fd[t]
            m = np.isfinite(g_true)
            g_nn = J[d, ti, :][m]
            g_true = g_true[m]
            entry["targets"][t] = {
                "grad_cosine": float(_cosine(g_nn, g_true)),
                "grad_rel_l2": float(_rel_l2(g_nn, g_true)),
                "sign_agree_frac": float(np.mean(np.sign(g_nn) == np.sign(g_true))),
                "nn_value": float(nn_vals[d, ti]),
                "sim_value": float(base[t]),
            }
        per_design.append(entry)
        print(f"  design {d}: done, {n_ok}/17 input partials ({len(per_design)} designs ok)")

    # aggregate per target
    agg = {}
    for t in CLASS2:
        cos = [e["targets"][t]["grad_cosine"] for e in per_design]
        rl2 = [e["targets"][t]["grad_rel_l2"] for e in per_design]
        sgn = [e["targets"][t]["sign_agree_frac"] for e in per_design]
        nn = np.array([e["targets"][t]["nn_value"] for e in per_design])
        sv = np.array([e["targets"][t]["sim_value"] for e in per_design])
        agg[t] = {
            "grad_cosine_mean": float(np.mean(cos)) if cos else None,
            "grad_rel_l2_median": float(np.median(rl2)) if rl2 else None,
            "sign_agree_mean": float(np.mean(sgn)) if sgn else None,
            "value_mae": float(np.mean(np.abs(nn - sv))) if len(nn) else None,
            "n_designs": len(per_design),
        }
    return {"per_target": agg, "per_design": per_design}


def main():
    p = argparse.ArgumentParser(description="NN gradient-quality vs ground truth")
    p.add_argument("--bundle", type=str, default="../../models/neural")
    p.add_argument("--data", type=str, default="../../outputs/rocket_data_full.jsonl")
    p.add_argument("--n-class1", type=int, default=300)
    p.add_argument("--with-sim", action="store_true",
                   help="Also finite-difference the RocketPy simulator for Class-2 targets")
    p.add_argument("--n-sim", type=int, default=3)
    p.add_argument("--sim-motor-classes", type=str, default="D,E,F",
                   help="Restrict the sim FD sample to these motor classes. Small "
                        "motors solve in <1s and never hit run_simulation's timeout, "
                        "so no zombie integrator threads accumulate across the 34 "
                        "solves/design (large classes can deadlock the diagnostic).")
    p.add_argument("--rel-step", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--class1-exact", action="store_true",
                   help="Splice exact analytic Barrowman for cg/cp/stability "
                        "(perfect value+gradient; the recommended optimizer config)")
    p.add_argument("--out", type=str, default="../../outputs/gradient_eval.json")
    args = p.parse_args()

    surr = DifferentiableSurrogate(args.bundle, class1_exact=args.class1_exact)
    print(f"Loaded surrogate from {args.bundle} "
          f"(arch={surr.model.__class__.__name__}, targets={len(surr.targets)}, "
          f"class1_exact={args.class1_exact})")

    report = {"bundle": args.bundle, "data": args.data}

    print(f"\n=== Class 1 (analytic Barrowman ground truth, n={args.n_class1}) ===")
    _, cont, cat = _load_designs(args.data, args.n_class1, args.seed)
    c1 = eval_class1(surr, cont, cat)
    report["class1"] = c1
    for t, m in c1.items():
        print(f"  {t:28s} grad_cos={m['grad_cosine_mean']:.4f}  "
              f"rel_l2_med={m['grad_rel_l2_median']:.4f}  "
              f"value_R2={m['value_r2']:.4f}  value_MAE={m['value_mae']:.4g}")

    if args.with_sim:
        print(f"\n=== Class 2 (simulator finite-difference, n={args.n_sim}, "
              f"rel_step={args.rel_step}) ===")
        abs_floor = {"wind_speed_mps": 0.1, "fin_sweep_m": 0.002,
                     "elevation_m": 5.0, "wind_direction_deg": 1.0}
        motor_filter = set(c.strip() for c in args.sim_motor_classes.split(",") if c.strip())
        recs, cont2, cat2 = _load_designs(args.data, args.n_sim, args.seed + 1000,
                                          motor_filter=motor_filter)
        print(f"  (restricted to motor classes {sorted(motor_filter)}: {len(recs)} designs)")
        c2 = eval_class2(surr, recs, cont2, cat2, args.rel_step, abs_floor)
        report["class2"] = c2["per_target"]
        print("\n  per-target (averaged over designs):")
        for t, m in c2["per_target"].items():
            if m["grad_cosine_mean"] is None:
                print(f"  {t:28s} (no successful designs)")
                continue
            print(f"  {t:28s} grad_cos={m['grad_cosine_mean']:.4f}  "
                  f"sign_agree={m['sign_agree_mean']:.3f}  "
                  f"rel_l2_med={m['grad_rel_l2_median']:.3f}  "
                  f"value_MAE={m['value_mae']:.4g}  (n={m['n_designs']})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
