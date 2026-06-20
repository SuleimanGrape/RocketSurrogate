"""Gradient-based rocket-design optimizer over the differentiable surrogate.

This is the end goal the differentiable surrogate (``diff_surrogate.py``) exists
for. It maximizes a flight metric (apogee by default) over the continuous design
inputs by following the surrogate's autograd gradient, subject to the physical
validity constraints the generator enforces:

    stability_margin ∈ [0.5, 4.0] cal,  max_mach ≤ 5,  max_acceleration ≤ 2000,
    apogee ≤ 100 km,  rail-exit ≥ 10 m/s,  T/W ≥ 3,  dry_mass ≥ 1.5·propellant,
    plus the geometric prevalidation bounds (fin/nose/slenderness/capacity).

Design vs. environment split
----------------------------
The 17 continuous inputs fall into two groups:
  * 11 DESIGN variables (geometry + mass + motor) — what an engineer chooses.
  * 6 ENVIRONMENT / launch variables (wind, elevation, temperature, rail length,
    launch angle) — the launch scenario, not the rocket.

Maximizing apogee over the environment trivially exploits it (launch straight up
from a cold mountaintop), which says nothing about the *design*. So by default
only the design variables are free and the environment is held at each start's
nominal launch scenario. ``--free all`` frees all 17 (faithful to the original
plan note); ``--free`` / ``--fix`` take explicit name lists.

Box bounds = the generator's *sampling* ranges (per fixed diameter/motor class),
NOT merely the hard physical limits — this keeps the optimizer in the region the
surrogate was trained on, where its predictions (and therefore its gradients) are
trustworthy. Hard physical constraints are imposed as squared-hinge penalties
with a penalty-continuation schedule. The categoricals (diameter, nose, fin
count, motor class) are discrete and held fixed per run; ``optimize_categoricals``
sweeps them in an outer loop.

The optimized design is validated against the real RocketPy simulator
(``--validate``) and the surrogate-vs-sim gap is reported — the honest test of
whether the optimum stayed in-distribution.

Usage:
    python design_optimizer.py --diameter 54 --motor K --nose ogive --fin 4 \
        --n-restarts 16 --steps 400 --validate
    python design_optimizer.py --diameter 38 --motor F --validate   # fast sim
    python design_optimizer.py --search --diameter 54 --validate     # sweep motors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# rocket_sim FIRST so the bare ``utils``/``config`` names resolve to the Barrowman
# / generator modules (same collision dance as eval_gradients.py / dataset.py).
# This file lives in neural_surrogate/optim, so common/ and rocket_sim/ are two
# levels up under src/.
sys.path.insert(0, os.path.join(_HERE, "..", "..", "common"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "rocket_sim"))
sys.path.insert(0, os.path.join(_HERE, ".."))   # neural_surrogate, so `optim.*` resolves

import torch                                   # noqa: E402
import schema                                  # noqa: E402
import config as cfg                           # noqa: E402

from optim.diff_surrogate import DifferentiableSurrogate, load_surrogate  # noqa: E402


def _compute_body_length_range(diameter_mm: int, motor_class: str) -> Tuple[float, float]:
    """Allowed body-length range for a (diameter, motor) pair — the in-distribution
    box for length_m. Inlined from parameters.py (pure config math) to avoid that
    module's top-level ``from utils import ...`` which collides with the
    neural_surrogate ``utils`` package on sys.path. Keeps dry/propellant mass ratio
    in [1.5, 10] under the generator's dry-mass model; widens k by 20% if tight."""
    d_m = diameter_mm / 1000.0
    d2 = d_m ** 2
    if d2 <= 0:
        return cfg.BODY_LENGTH_MIN_M, cfg.BODY_LENGTH_MAX_M
    prop_min, prop_max = cfg.MOTOR_SPECS[motor_class][0], cfg.MOTOR_SPECS[motor_class][1]
    k_min, k_max = cfg.DRY_MASS_K_MIN, cfg.DRY_MASS_K_MAX
    L_min = 1.5 * prop_max / (k_max * d2)
    L_max = 10.0 * prop_min / (k_min * d2)
    if L_min > L_max:
        L_min = 1.5 * prop_max / (k_max * 1.2 * d2)
        L_max = 10.0 * prop_min / (k_min * 0.8 * d2)
    L_min = max(cfg.BODY_LENGTH_MIN_M, L_min)
    L_max = min(cfg.BODY_LENGTH_MAX_M, L_max)
    if L_min > L_max:
        return cfg.BODY_LENGTH_MIN_M, cfg.BODY_LENGTH_MAX_M
    return L_min, L_max

# ── Continuous-input layout (schema.INPUT_CONTINUOUS order) ────────────────────
CONT = list(schema.INPUT_CONTINUOUS)
_CI = {n: i for i, n in enumerate(CONT)}

DESIGN_VARS = [
    "length_m", "nose_length_m", "fin_root_chord_m", "fin_tip_chord_m",
    "fin_span_m", "fin_sweep_m", "fin_thickness_mm", "dry_mass_kg",
    "propellant_mass_kg", "burn_time_s", "avg_thrust_N",
]
ENV_VARS = [
    "wind_speed_mps", "wind_direction_deg", "elevation_m", "temperature_c",
    "rail_length_m", "launch_angle_deg",
]
assert set(DESIGN_VARS) | set(ENV_VARS) == set(CONT)

# Nominal launch scenario used when an environment variable is frozen and no
# warm-start value is available (calm, sea level, mild, vertical).
NOMINAL_ENV = {
    "wind_speed_mps": 0.0, "wind_direction_deg": 0.0, "elevation_m": 0.0,
    "temperature_c": 15.0, "rail_length_m": 4.0, "launch_angle_deg": 90.0,
}

_G = 9.81
_PROP_DENSITY = 1815.0  # validator.py propellant packing density


# ── Box bounds (the generator's sampling ranges, per fixed categoricals) ───────
def box_bounds(diameter_mm: int, motor_class: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per-input [lo, hi] in CONT order, matching parameters.py's sampling ranges.

    Staying inside these keeps the optimizer in the surrogate's training region.
    Cross-variable couplings (tip≤root, sweep≤root) are handled by projection in
    the optimization loop, so tip/sweep get generous independent boxes here.
    """
    d_m = diameter_mm / 1000.0
    mspec = cfg.MOTOR_SPECS[motor_class]
    L_lo, L_hi = _compute_body_length_range(diameter_mm, motor_class)

    lo = np.empty(len(CONT)); hi = np.empty(len(CONT))

    def put(name, a, b):
        lo[_CI[name]] = a; hi[_CI[name]] = b

    put("length_m", L_lo, L_hi)
    put("nose_length_m", cfg.NOSE_LENGTH_MIN_DIAMETERS * d_m, cfg.NOSE_LENGTH_MAX_DIAMETERS * d_m)
    put("fin_root_chord_m", cfg.FIN_ROOT_CHORD_MIN_DIAMETERS * d_m, cfg.FIN_ROOT_CHORD_MAX_DIAMETERS * d_m)
    # tip/sweep ∈ [.., root] enforced by projection; independent box covers the range.
    put("fin_tip_chord_m", cfg.FIN_TIP_CHORD_MIN_FRAC * cfg.FIN_ROOT_CHORD_MIN_DIAMETERS * d_m,
        cfg.FIN_ROOT_CHORD_MAX_DIAMETERS * d_m)
    put("fin_span_m", cfg.FIN_SPAN_MIN_DIAMETERS * d_m, cfg.FIN_SPAN_MAX_DIAMETERS * d_m)
    put("fin_sweep_m", 0.0, cfg.FIN_ROOT_CHORD_MAX_DIAMETERS * d_m)
    put("fin_thickness_mm", cfg.FIN_THICKNESS_MIN_MM, cfg.FIN_THICKNESS_MAX_MM)
    put("dry_mass_kg", cfg.DRY_MASS_MIN_KG, cfg.DRY_MASS_MAX_KG)
    put("propellant_mass_kg", mspec[0], mspec[1])
    put("burn_time_s", mspec[2], mspec[3])
    put("avg_thrust_N", mspec[4], mspec[5])
    put("wind_speed_mps", cfg.WIND_SPEED_MIN_MS, cfg.WIND_SPEED_MAX_MS)
    put("wind_direction_deg", cfg.WIND_DIRECTION_MIN_DEG, cfg.WIND_DIRECTION_MAX_DEG)
    put("elevation_m", cfg.ELEVATION_MIN_M, cfg.ELEVATION_MAX_M)
    put("temperature_c", cfg.TEMPERATURE_MIN_C, cfg.TEMPERATURE_MAX_C)
    put("rail_length_m", cfg.RAIL_LENGTH_MIN_M, cfg.RAIL_LENGTH_MAX_M)
    put("launch_angle_deg", cfg.LAUNCH_ANGLE_MIN_DEG, cfg.LAUNCH_ANGLE_MAX_DEG)
    return lo, hi


# ── Constraints ────────────────────────────────────────────────────────────────
# Each is normalized so a violation of ~1.0 means "~100% off", making the penalty
# weights commensurate across very different physical scales. A constraint maps a
# (raw-input tensor, surrogate-output tensor) pair to a per-row violation ≥ 0
# (0 = satisfied). Squared and summed in the loss; raw value used for feasibility.
def _hinge(z):  # max(0, z), differentiable
    return torch.clamp(z, min=0.0)


def constraint_violations(x: torch.Tensor, out: torch.Tensor, surr: DifferentiableSurrogate,
                          d_m: float, eps: float = 1e-9) -> Dict[str, torch.Tensor]:
    """Per-row normalized violations (≥0). x:(B,17) raw inputs, out:(B,12) metrics."""
    g = lambda name: x[:, _CI[name]]
    body_l, nose_l = g("length_m"), g("nose_length_m")
    root, tip, span, sweep = g("fin_root_chord_m"), g("fin_tip_chord_m"), g("fin_span_m"), g("fin_sweep_m")
    dry, prop = g("dry_mass_kg"), g("propellant_mass_kg")
    thrust, rail = g("avg_thrust_N"), g("rail_length_m")
    total_mass = dry + prop
    total_len = nose_l + body_l

    o = lambda name: out[:, surr.target_index(name)]
    sm, mach, accel = o("stability_margin_calibers"), o("max_mach"), o("max_acceleration_mps2")
    apogee, rail_exit = o("apogee_m"), o("rail_exit_velocity_mps")

    body_vol = math.pi * (d_m / 2.0) ** 2 * body_l
    cap = body_vol * _PROP_DENSITY * 0.8
    # analytic rail-exit estimate (validator.prevalidate gate)
    net = thrust - total_mass * _G
    v_exit_est = torch.sqrt(_hinge(2.0 * net * rail / (total_mass + eps)) + eps)

    return {
        # surrogate-predicted flight constraints
        "stability_lo": _hinge(cfg.STABILITY_MARGIN_MIN_CAL - sm),
        "stability_hi": _hinge(sm - cfg.STABILITY_MARGIN_MAX_CAL),
        "mach_cap": _hinge(mach - cfg.MAX_MACH),
        "accel_cap": _hinge((accel - 2000.0) / 2000.0),
        "apogee_cap": _hinge((apogee - cfg.MAX_APOGEE_KM * 1000.0) / (cfg.MAX_APOGEE_KM * 1000.0)),
        "rail_exit_min": _hinge((10.0 - rail_exit) / 10.0),
        # geometric / mass prevalidation constraints (analytic, from raw inputs)
        "dry_vs_prop": _hinge((1.5 * prop - dry) / (dry + eps)),
        "thrust_to_weight": _hinge((cfg.THRUST_TO_WEIGHT_MIN * total_mass * _G - thrust) / (thrust + eps)),
        "span_le_body": _hinge((span - body_l) / (body_l + eps)),
        "root_le_body": _hinge((root - body_l) / (body_l + eps)),
        "tip_le_root": _hinge((tip - root) / (root + eps)),
        "sweep_le_root": _hinge((sweep - root) / (root + eps)),
        "nose_le_body": _hinge((nose_l - 1.5 * body_l) / (body_l + eps)),
        "slenderness": _hinge((total_len / d_m - 80.0) / 80.0),
        "prop_capacity": _hinge((prop - cap) / (cap + eps)),
        "rail_vs_len": _hinge((rail - 3.0 * total_len) / (total_len + eps)),
        "v_exit_est_min": _hinge((10.0 - v_exit_est) / 10.0),
    }


def indist_violation(x: torch.Tensor, d_m: float, eps: float = 1e-9) -> torch.Tensor:
    """Soft in-distribution regularizer: dry_mass coefficient k = dry/(d²·L) was
    sampled in [200, 600]. Keeps the optimized mass realistic where dry_mass is a
    free, otherwise barely-constrained input. Low weight."""
    body_l, dry = x[:, _CI["length_m"]], x[:, _CI["dry_mass_kg"]]
    k = dry / (d_m ** 2 * body_l + eps)
    return _hinge((cfg.DRY_MASS_K_MIN - k) / cfg.DRY_MASS_K_MIN) + \
        _hinge((k - cfg.DRY_MASS_K_MAX) / cfg.DRY_MASS_K_MAX)


# ── Optimizer ──────────────────────────────────────────────────────────────────
class DesignOptimizer:
    """Projected-gradient (Adam) design optimizer over the differentiable surrogate."""

    def __init__(self, surrogate: DifferentiableSurrogate,
                 objective: str = "apogee_m", maximize: bool = True,
                 obj_scale: Optional[float] = None):
        self.surr = surrogate
        self.objective = objective
        self.maximize = maximize
        # Scale the objective to ~O(1) so penalty weights are meaningful. For
        # apogee (metres, ~1e2–1e4) default to km.
        self.obj_scale = obj_scale if obj_scale is not None else (
            1000.0 if objective == "apogee_m" else 1.0)

    # ------------------------------------------------------------------
    def optimize(self, cat_config: Dict, *, free_vars: Optional[Sequence[str]] = None,
                 inits: Optional[np.ndarray] = None, n_restarts: int = 16,
                 steps: int = 400, lr: float = 0.02, rounds: int = 3,
                 base_penalty: float = 10.0, indist_weight: float = 1.0,
                 feas_tol: float = 1e-2, seed: int = 0,
                 nominal_env: bool = False, verbose: bool = False) -> Dict:
        """Optimize a batch of restarts for one fixed categorical configuration.

        cat_config : {diameter_mm, nose_type, fin_count, motor_class}.
        free_vars  : input names to optimize (default DESIGN_VARS).
        inits      : (n_restarts, 17) warm starts; random-in-box if None/short.
        Penalty continuation: ``rounds`` rounds, penalty weight ×10 each round.
        Returns the best (highest/lowest objective) feasible design found.
        """
        surr = self.surr
        d_mm = int(cat_config["diameter_mm"]); motor = cat_config["motor_class"]
        d_m = d_mm / 1000.0
        free = list(free_vars) if free_vars is not None else list(DESIGN_VARS)
        free_idx = np.array([_CI[n] for n in free], dtype=np.int64)
        fixed_idx = np.array([i for i in range(len(CONT)) if i not in set(free_idx)], dtype=np.int64)

        lo, hi = box_bounds(d_mm, motor)

        # Optimize in box-normalized space u ∈ [0,1] (x = lo + u·(hi-lo)). This
        # rescales every input to a common magnitude so one Adam lr moves them
        # commensurately; the box constraint becomes a simple clip to [0,1]. The
        # tip≤root / sweep≤root couplings are handled by the penalty terms, not a
        # hard projection (projecting x would break the u→x mapping).
        span = hi - lo
        span_safe = np.where(span > 0, span, 1.0)
        lo_t = torch.tensor(lo, dtype=torch.float32)
        span_t = torch.tensor(span_safe, dtype=torch.float32)

        rng = np.random.default_rng(seed)
        x0 = self._make_inits(inits, n_restarts, lo, hi, cat_config, rng, nominal_env)
        B = x0.shape[0]
        codes = torch.tensor(np.tile(self._cat_codes(cat_config), (B, 1)), dtype=torch.long)
        u0 = np.clip((x0 - lo) / span_safe, 0.0, 1.0)
        u0_t = torch.tensor(u0, dtype=torch.float32)
        u = u0_t.clone().requires_grad_(True)

        sign = -1.0 if self.maximize else 1.0  # minimize sign·objective
        oi = surr.target_index(self.objective)

        def to_x(uu):
            return lo_t + uu * span_t

        for r in range(rounds):
            weight = base_penalty * (10.0 ** r)
            opt = torch.optim.Adam([u], lr=lr)
            for step in range(steps):
                opt.zero_grad()
                x = to_x(u)
                out = surr.forward(x, codes)
                obj = out[:, oi] / self.obj_scale
                viol = constraint_violations(x, out, surr, d_m)
                pen = sum((v ** 2).sum() for v in viol.values())
                reg = (indist_violation(x, d_m) ** 2).sum()
                loss = sign * obj.sum() + weight * pen + indist_weight * reg
                loss.backward()
                if fixed_idx.size:
                    u.grad[:, fixed_idx] = 0.0
                opt.step()
                with torch.no_grad():
                    u.clamp_(0.0, 1.0)
                    if fixed_idx.size:
                        u[:, fixed_idx] = u0_t[:, fixed_idx]
            if verbose:
                with torch.no_grad():
                    out = surr.forward(to_x(u), codes)
                    feas = self._feasible_mask(to_x(u), out, surr, d_m, feas_tol)
                    best = self._best_obj(out[:, oi], feas)
                    print(f"  round {r} (w={weight:g}): {int(feas.sum())}/{B} feasible, "
                          f"best {self.objective}={best:.1f}")

        # Final evaluation + pick best feasible (fallback: least-infeasible).
        with torch.no_grad():
            x = to_x(u)
            out = surr.forward(x, codes)
        viol = constraint_violations(x, out, surr, d_m)
        viol_np = {k: v.numpy() for k, v in viol.items()}
        total_viol = np.max([np.maximum(v, 0.0) for v in viol_np.values()], axis=0)
        feas = total_viol <= feas_tol
        obj_vals = out[:, oi].numpy()
        ranked = (-obj_vals if self.maximize else obj_vals)
        order = np.lexsort((ranked, ~feas))  # feasible first, then best objective
        best_i = int(order[0])

        results = []
        for i in range(B):
            results.append(self._design_record(
                x[i].detach().numpy(), out[i].numpy(), surr, cat_config,
                {k: float(v[i]) for k, v in viol_np.items()}, bool(feas[i])))
        best = results[best_i]
        best["restart_index"] = best_i
        return {
            "cat_config": cat_config, "free_vars": free, "fixed_vars": [CONT[i] for i in fixed_idx],
            "objective": self.objective, "maximize": self.maximize,
            "n_restarts": B, "n_feasible": int(feas.sum()),
            "best": best, "all_restarts": results,
        }

    # ------------------------------------------------------------------
    def _make_inits(self, inits, n, lo, hi, cat_config, rng, nominal_env):
        """Assemble (n,17) starts: supplied warm starts padded with random-in-box."""
        rows = []
        if inits is not None and len(inits):
            rows.extend(np.asarray(inits, dtype=np.float64)[:n])
        while len(rows) < n:
            u = rng.random(len(CONT))
            row = lo + u * (hi - lo)
            row[_CI["fin_tip_chord_m"]] = min(row[_CI["fin_tip_chord_m"]], row[_CI["fin_root_chord_m"]])
            row[_CI["fin_sweep_m"]] = min(row[_CI["fin_sweep_m"]], row[_CI["fin_root_chord_m"]])
            for k, val in NOMINAL_ENV.items():
                row[_CI[k]] = val
            rows.append(row)
        x0 = np.array(rows[:n], dtype=np.float64)
        if nominal_env:
            for k, val in NOMINAL_ENV.items():
                x0[:, _CI[k]] = val
        return x0

    @staticmethod
    def _cat_codes(cat_config) -> np.ndarray:
        return np.array([schema.ENCODING_MAPS[k][_coerce(k, cat_config[k])]
                         for k in schema.INPUT_CATEGORICAL], dtype=np.int64)

    def _feasible_mask(self, x, out, surr, d_m, tol):
        viol = constraint_violations(x, out, surr, d_m)
        worst = torch.stack(list(viol.values())).max(dim=0).values
        return (worst <= tol)

    def _best_obj(self, col, feas):
        vals = col.numpy()
        m = feas.numpy() if isinstance(feas, torch.Tensor) else feas
        if not m.any():
            return float(vals.max() if self.maximize else vals.min())
        v = vals[m]
        return float(v.max() if self.maximize else v.min())

    def _design_record(self, x_row, out_row, surr, cat_config, viol, feasible):
        params = {k: float(x_row[_CI[k]]) for k in CONT}
        params.update({k: _coerce(k, cat_config[k]) for k in schema.INPUT_CATEGORICAL})
        pred = {t: float(out_row[surr.target_index(t)]) for t in surr.targets}
        return {"params": params, "pred_metrics": pred,
                "feasible": feasible, "violations": viol,
                "objective_value": float(out_row[surr.target_index(self.objective)])}


def _coerce(field: str, value):
    """Cast a categorical to the type its ENCODING_MAPS keys use (int for
    diameter/fin count, str for nose/motor)."""
    if field in ("diameter_mm", "fin_count"):
        return int(value)
    return value


# ── Warm starts from the corpus ────────────────────────────────────────────────
def corpus_inits(data_path: str, cat_config: Dict, n: int, seed: int) -> np.ndarray:
    """Pull up to n real designs matching the categorical config as warm starts."""
    from dataio import load_jsonl
    want = {k: _coerce(k, cat_config[k]) for k in schema.INPUT_CATEGORICAL}
    recs = []
    for r in load_jsonl(data_path):
        if r.get("output", {}).get("within_bounds") is False:
            continue
        inp = r["input"]
        if all(_coerce(k, inp.get(k)) == want[k] for k in want):
            recs.append(inp)
    if not recs:
        return np.empty((0, len(CONT)))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))[:n]
    return np.array([[float(recs[i][k]) for k in CONT] for i in idx], dtype=np.float64)


# ── Real-simulator validation ──────────────────────────────────────────────────
_SIM = {}


def _ensure_sim():
    if _SIM:
        return
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "rocket_sim"))
    import utils  # noqa: F401  bind bare ``utils`` to rocket_sim/utils (Barrowman)
    import rocket_builder, simulator, validator, outputs
    _SIM.update(rocket_builder=rocket_builder, simulator=simulator,
                validator=validator, outputs=outputs)


def validate_design(params: Dict) -> Dict:
    """Run the optimized design through the real RocketPy pipeline.

    Returns prevalidation result, post-sim is_valid, and the true flight metrics
    (extract_output) so the caller can compare against the surrogate prediction.
    """
    _ensure_sim()
    pre_ok, pre_reason = _SIM["validator"].prevalidate(params)
    res = {"prevalidate_ok": bool(pre_ok), "prevalidate_reason": pre_reason,
           "sim_ok": False, "is_valid": False, "sim_metrics": None}
    if not pre_ok:
        return res
    try:
        rocket = _SIM["rocket_builder"].build_rocket(params)
    except Exception as e:
        res["prevalidate_reason"] = f"build_rocket failed: {e}"
        return res
    flight = _SIM["simulator"].run_simulation(rocket, params)
    if flight is None:
        return res
    res["sim_ok"] = True
    res["is_valid"] = bool(_SIM["validator"].is_valid(params, flight))
    try:
        res["sim_metrics"] = _SIM["outputs"].extract_output(params, flight)
    except Exception as e:
        res["sim_metrics"] = {"error": str(e)}
    return res


def compare_to_surrogate(pred: Dict, sim_metrics: Dict, keys=("apogee_m",
                         "stability_margin_calibers", "max_mach",
                         "max_acceleration_mps2")) -> Dict:
    """Surrogate-vs-sim gap on the headline metrics."""
    out = {}
    for k in keys:
        if k in pred and sim_metrics and k in sim_metrics:
            p, s = float(pred[k]), float(sim_metrics[k])
            out[k] = {"pred": p, "sim": s, "abs_err": abs(p - s),
                      "rel_err": abs(p - s) / max(abs(s), 1e-9)}
    return out


# ── Categorical outer loop ─────────────────────────────────────────────────────
def optimize_categoricals(opt: DesignOptimizer, diameters: Sequence[int],
                          noses: Sequence[str], fins: Sequence[int],
                          data_path: Optional[str], *, restarts: int, steps: int,
                          lr: float, rounds: int, free_vars, seed: int,
                          nominal_env: bool) -> Dict:
    """Sweep allowed (diameter, motor, nose, fin) combos; return the global best."""
    combos = []
    for d in diameters:
        for mi in cfg.ALLOWED_MOTORS_BY_DIAMETER.get(d, []):
            m = cfg.MOTOR_CLASSES[mi]
            for nose in noses:
                for fin in fins:
                    combos.append({"diameter_mm": d, "nose_type": nose,
                                   "fin_count": fin, "motor_class": m})
    print(f"Sweeping {len(combos)} categorical combos...")
    runs = []
    best = None
    for c in combos:
        inits = corpus_inits(data_path, c, restarts, seed) if data_path else None
        res = opt.optimize(c, free_vars=free_vars, inits=inits, n_restarts=restarts,
                           steps=steps, lr=lr, rounds=rounds, seed=seed,
                           nominal_env=nominal_env)
        b = res["best"]
        tag = f"d{c['diameter_mm']}/{c['motor_class']}/{c['nose_type']}/{c['fin_count']}"
        print(f"  {tag:28s} {opt.objective}={b['objective_value']:10.1f}  "
              f"feasible={b['feasible']}  ({res['n_feasible']}/{res['n_restarts']})")
        runs.append({"cat": c, "objective_value": b["objective_value"],
                     "feasible": b["feasible"], "result": res})
        if b["feasible"] and (best is None or
                              (opt.maximize and b["objective_value"] > best["best"]["objective_value"]) or
                              (not opt.maximize and b["objective_value"] < best["best"]["objective_value"])):
            best = res
    return {"best": best, "runs": [{k: r[k] for k in ("cat", "objective_value", "feasible")} for r in runs]}


# ── CLI ─────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Gradient-based rocket-design optimizer")
    p.add_argument("--bundle", default=os.path.join(_HERE, "..", "..", "..", "models", "neural_distilled"))
    p.add_argument("--no-class1-exact", action="store_true",
                   help="Use the raw NN for cg/cp/stability (default: splice exact Barrowman)")
    p.add_argument("--data", default=os.path.join(_HERE, "..", "..", "..", "outputs", "rocket_data_full.jsonl"),
                   help="Corpus for warm starts (matching categoricals)")
    p.add_argument("--no-warm-start", action="store_true", help="Random-in-box inits only")
    # categoricals
    p.add_argument("--diameter", type=int, default=54)
    p.add_argument("--motor", default="K")
    p.add_argument("--nose", default="ogive", choices=cfg.NOSE_TYPES)
    p.add_argument("--fin", type=int, default=4, choices=cfg.FIN_COUNTS)
    # objective / free vars
    p.add_argument("--objective", default="apogee_m", choices=schema.TARGETS)
    p.add_argument("--minimize", action="store_true")
    p.add_argument("--free", default="design",
                   help="'design' (default), 'all', or a comma list of input names")
    p.add_argument("--fix", default="", help="Comma list of input names to additionally fix")
    p.add_argument("--nominal-env", action="store_true",
                   help="Force one common nominal launch scenario across all restarts")
    # optimization
    p.add_argument("--n-restarts", type=int, default=16)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    # actions
    p.add_argument("--search", action="store_true", help="Sweep allowed motors for --diameter")
    p.add_argument("--validate", action="store_true", help="Validate the best design in RocketPy")
    p.add_argument("--out", default=os.path.join(_HERE, "..", "..", "..", "outputs", "design_optimization.json"))
    args = p.parse_args()

    surr = load_surrogate(args.bundle, class1_exact=not args.no_class1_exact)
    print(f"Loaded surrogate from {os.path.normpath(args.bundle)} "
          f"(class1_exact={not args.no_class1_exact}, targets={len(surr.targets)})")

    opt = DesignOptimizer(surr, objective=args.objective, maximize=not args.minimize)

    if args.free == "design":
        free = list(DESIGN_VARS)
    elif args.free == "all":
        free = list(CONT)
    else:
        free = [s.strip() for s in args.free.split(",") if s.strip()]
    fix = {s.strip() for s in args.fix.split(",") if s.strip()}
    free = [v for v in free if v not in fix]
    data_path = None if args.no_warm_start else args.data

    if args.search:
        report = optimize_categoricals(
            opt, [args.diameter], [args.nose], [args.fin], data_path,
            restarts=args.n_restarts, steps=args.steps, lr=args.lr,
            rounds=args.rounds, free_vars=free, seed=args.seed,
            nominal_env=args.nominal_env)
        result = report["best"]
        if result is None:
            print("\nNo feasible design found in the sweep.")
            _save(args.out, report); return
    else:
        cat = {"diameter_mm": args.diameter, "nose_type": args.nose,
               "fin_count": args.fin, "motor_class": args.motor}
        inits = corpus_inits(data_path, cat, args.n_restarts, args.seed) if data_path else None
        if inits is not None:
            print(f"Warm starts from corpus: {len(inits)} matching designs")
        result = opt.optimize(cat, free_vars=free, inits=inits, n_restarts=args.n_restarts,
                              steps=args.steps, lr=args.lr, rounds=args.rounds,
                              seed=args.seed, nominal_env=args.nominal_env, verbose=True)
        report = result

    _print_result(result, opt)

    if args.validate:
        print("\n=== Real-simulator validation of the best design ===")
        val = validate_design(result["best"]["params"])
        print(f"  prevalidate: {val['prevalidate_ok']}  ({val['prevalidate_reason']})")
        print(f"  sim_ok={val['sim_ok']}  is_valid={val['is_valid']}")
        if val["sim_metrics"] and "error" not in val["sim_metrics"]:
            cmp = compare_to_surrogate(result["best"]["pred_metrics"], val["sim_metrics"])
            print("  surrogate vs sim:")
            for k, m in cmp.items():
                print(f"    {k:28s} pred={m['pred']:10.2f}  sim={m['sim']:10.2f}  "
                      f"rel_err={m['rel_err']*100:5.1f}%")
            result["validation"] = {"detail": val, "comparison": cmp}
        else:
            result["validation"] = {"detail": val}

    _save(args.out, report)


def _print_result(result, opt):
    b = result["best"]
    print(f"\n=== Best design ({'feasible' if b['feasible'] else 'INFEASIBLE'}; "
          f"{result['n_feasible']}/{result['n_restarts']} restarts feasible) ===")
    print(f"  {opt.objective} = {b['objective_value']:.1f}"
          f"  ({'max' if opt.maximize else 'min'})")
    print("  design params:")
    for k in DESIGN_VARS:
        print(f"    {k:22s} {b['params'][k]:.4f}")
    print("  predicted metrics:")
    for k in ("apogee_m", "max_velocity_mps", "max_mach", "max_acceleration_mps2",
              "stability_margin_calibers", "rail_exit_velocity_mps"):
        print(f"    {k:28s} {b['pred_metrics'][k]:.2f}")
    active = {k: v for k, v in b["violations"].items() if v > 1e-3}
    if active:
        print("  active constraint violations:", {k: round(v, 4) for k, v in active.items()})


def _save(path, report):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nReport saved to {os.path.normpath(str(out))}")


if __name__ == "__main__":
    main()
