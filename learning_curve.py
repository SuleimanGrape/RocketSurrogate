#!/usr/bin/env python3
"""Sample-complexity study: how does surrogate accuracy scale with dataset size?

Trains the production XGBoost surrogate (same features, same native-categorical
setup) on increasing slices of the data we have, evaluating every slice on ONE
fixed held-out test set, and plots test R^2 vs. number of training samples. The
shape of that curve tells us how many samples we'll need before adding more data
stops paying off — which is what we want to know before generating the
not-computable (out-of-bounds) class.

Usage:
    python learning_curve.py --inputs outputs/rocket_data_all.jsonl \
                                      outputs/rocket_data_10k_s2028.jsonl \
                             --out-dir outputs/learning_curve
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "common"))
sys.path.insert(0, os.path.join(ROOT, "src", "rocket_sim"))
sys.path.insert(0, os.path.join(ROOT, "src", "gbt"))

import schema                                   # noqa: E402
from dataio import load_jsonl                   # noqa: E402
from data_loader import extract_arrays          # noqa: E402
from preprocess import add_engineered_features  # noqa: E402
import xgboost as xgb                           # noqa: E402

# Fixed XGBoost config — representative of the production models, no per-size
# tuning so the only thing that varies across the curve is the sample count.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "verbosity": 0,
    "enable_categorical": True,
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_lambda": 2.0,
}
NUM_BOOST_ROUND = 1000
EARLY_STOP = 30

# Targets highlighted individually on the plot (the rest fold into the mean).
KEY_TARGETS = ["apogee_m", "max_velocity_mps", "max_mach", "stability_margin_calibers"]

# Optional: targets trained on log1p(y) (inverted at predict) for heavy tails.
# Tested on max_acceleration_mps2 (mean 255, spikes to 2000) — it did NOT help
# (R^2 0.771 -> 0.761), so the set is empty. The plateau there is inherent
# noise/feature limitation, not a scale problem. Kept as a hook for future use.
LOG_TARGETS = set()


def _fingerprint(inp: dict) -> str:
    return hashlib.md5(
        "|".join(f"{k}={inp[k]}" for k in sorted(inp)).encode()).hexdigest()


def load_dedup(paths):
    """Load all records from the given files, dropping exact-input duplicates."""
    seen, recs = set(), []
    for p in paths:
        for r in load_jsonl(p):
            fp = _fingerprint(r["input"])
            if fp in seen:
                continue
            seen.add(fp)
            recs.append(r)
    return recs


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def size_schedule(n_pool):
    """Log-spaced training sizes up to the full pool."""
    base = [500, 1000, 2000, 4000, 8000, 16000, 24000, 32000]
    sizes = [s for s in base if s < n_pool]
    sizes.append(n_pool)
    return sizes


def train_eval(Xtr, Ytr, Xva, Yva, Xte, Yte, feat_names, target_names):
    """Train one model per target on (Xtr,Ytr); return test R^2 per target."""
    out = {}
    dtest = xgb.DMatrix(Xte, feature_names=feat_names, enable_categorical=True)
    for i, name in enumerate(target_names):
        log_t = name in LOG_TARGETS
        ytr = np.log1p(Ytr[:, i]) if log_t else Ytr[:, i]
        yva = np.log1p(Yva[:, i]) if log_t else Yva[:, i]
        dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_names,
                             enable_categorical=True)
        dval = xgb.DMatrix(Xva, label=yva, feature_names=feat_names,
                           enable_categorical=True)
        model = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True)],
            verbose_eval=False,
        )
        pred = model.predict(dtest)
        if log_t:
            pred = np.expm1(pred)        # invert to the original scale before scoring
        out[name] = r2(Yte[:, i], pred)
    return out


def main():
    ap = argparse.ArgumentParser(description="XGBoost learning curve (R^2 vs N).")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="outputs/learning_curve")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of each TRAIN slice carved off for early stopping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = load_dedup(args.inputs)
    print(f"Loaded {len(recs)} unique records from {len(args.inputs)} file(s).")

    target_names = schema.TARGETS
    X, Y, feat_names0, _ = extract_arrays(recs, schema.INPUT_FIELDS, target_names)
    X, feat_names = add_engineered_features(X, feat_names0)
    print(f"Features: {len(feat_names)}  Targets: {len(target_names)}")

    # Fixed shuffle, then a fixed held-out test set used for every slice.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(recs))
    n_test = int(len(recs) * args.test_frac)
    test_idx, pool_idx = perm[:n_test], perm[n_test:]
    Xte = X.iloc[test_idx].reset_index(drop=True)
    Yte = Y[test_idx]
    n_pool = len(pool_idx)
    print(f"Held-out test: {n_test}   Train pool: {n_pool}")

    sizes = size_schedule(n_pool)
    print(f"Training sizes: {sizes}\n")

    rows = []
    for n in sizes:
        t0 = time.time()
        sub = pool_idx[:n]
        n_val = max(1, int(n * args.val_frac))
        tr_idx, va_idx = sub[n_val:], sub[:n_val]
        Xtr = X.iloc[tr_idx].reset_index(drop=True)
        Xva = X.iloc[va_idx].reset_index(drop=True)
        per_target = train_eval(Xtr, Y[tr_idx], Xva, Y[va_idx], Xte, Yte,
                                feat_names, target_names)
        mean_r2 = float(np.mean(list(per_target.values())))
        rows.append({"n_train": int(len(tr_idx)), "mean_r2": mean_r2,
                     "per_target": per_target})
        keys = "  ".join(f"{k}={per_target[k]:.3f}" for k in KEY_TARGETS)
        print(f"  n_train={len(tr_idx):>6}  mean_R2={mean_r2:.4f}   "
              f"[{keys}]   ({time.time()-t0:.1f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "learning_curve.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"sizes": sizes, "test_n": n_test, "seed": args.seed,
                   "results": rows}, f, indent=2)

    _plot(rows, target_names, os.path.join(args.out_dir, "learning_curve.png"))
    print(f"\nWrote {json_path}")
    print(f"Wrote {os.path.join(args.out_dir, 'learning_curve.png')}")


def _plot(rows, target_names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    # Faint lines for every target.
    for t in target_names:
        ax.plot(ns, [r["per_target"][t] for r in rows],
                color="0.8", lw=1, zorder=1)
    # Highlight the key targets.
    for t in KEY_TARGETS:
        ax.plot(ns, [r["per_target"][t] for r in rows],
                marker="o", lw=1.5, label=t, zorder=2)
    # Mean across all targets, bold.
    ax.plot(ns, [r["mean_r2"] for r in rows], color="black", marker="s",
            lw=2.5, label="mean (all 13 targets)", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("test $R^2$")
    ax.set_title("XGBoost surrogate learning curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
