#!/usr/bin/env python3
"""Sample-complexity study for the NEURAL surrogate — does accuracy still scale?

The NN twin of learning_curve.py (which did this for XGBoost). Trains the MLP
surrogate on increasing slices of the data, evaluating every slice on ONE fixed
held-out test set, and plots test R^2 vs. number of training samples. A curve
that is still rising at the full corpus means more samples would help; a flat
tail means the NN is saturated like the trees.

Uses the SAME pipeline as train_surrogate.py: positives only, engineered
(Barrowman-inclusive) continuous features, standardized inputs/targets — so the
result is directly comparable to the XGBoost learning curve.

Designed to run on the ROCm/CUDA training machine (device=auto). On CPU it still
runs, just slower; drop --epochs / --sizes for a quick smoke.

Usage:
    python nn_learning_curve.py --data outputs/rocket_data_full.jsonl \
        --out-dir outputs/nn_learning_curve --model mlp --epochs 200 \
        --xgb-json outputs/learning_curve_v2/learning_curve.json   # optional overlay
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "neural_surrogate"))

from models.surrogate import (build_model, CONTINUOUS_FEATURES,  # noqa: E402
                              CATEGORICAL_FEATURES, CATEGORICAL_CARDINALITIES, TARGETS)
from models.scalers import StandardScaler  # noqa: E402
from data.dataset import RocketDataset      # noqa: E402
from training.trainer import Trainer        # noqa: E402
from utils.helpers import set_seed          # noqa: E402

# Targets highlighted on the plot (rest fold into the mean) — match learning_curve.py.
KEY_TARGETS = ["apogee_m", "max_velocity_mps", "max_mach", "stability_margin_calibers"]


def size_schedule(n_pool):
    base = [500, 1000, 2000, 4000, 8000, 16000, 24000, 32000]
    sizes = [s for s in base if s < n_pool]
    sizes.append(n_pool)
    return sizes


def make_loader(cont, cat, tgt, idx, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(cont[idx]).float(),
        torch.from_numpy(cat[idx]).long(),
        torch.from_numpy(tgt[idx]).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_eval(cont, cat, tgt_scaled, tgt_scaler, tr_idx, va_idx, te_idx,
               n_cont, args, ckpt_dir):
    """Train one MLP on tr_idx; return per-target test R^2 dict."""
    model = build_model(
        args.model,
        continuous_dim=n_cont,
        categorical_cardinalities=CATEGORICAL_CARDINALITIES,
        embedding_dim=args.embedding_dim,
        output_dim=len(TARGETS),
        dropout=args.dropout,
    )
    trainer = Trainer(model=model, device=args.device, lr=args.lr,
                      weight_decay=args.weight_decay, scheduler="cosine",
                      output_scaler=tgt_scaler)
    loaders = {
        "train": make_loader(cont, cat, tgt_scaled, tr_idx, args.batch_size, True),
        "val":   make_loader(cont, cat, tgt_scaled, va_idx, args.batch_size, False),
    }
    trainer.fit(loaders, epochs=args.epochs, patience=args.patience,
                log_every=10**9, ckpt_dir=ckpt_dir)  # silence per-epoch logs
    test_loader = make_loader(cont, cat, tgt_scaled, te_idx, args.batch_size, False)
    res = trainer.evaluate(test_loader, target_names=TARGETS)
    return {t: res[f"r2_{t}"] for t in TARGETS}


def main():
    ap = argparse.ArgumentParser(description="Neural surrogate learning curve (R^2 vs N).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="outputs/nn_learning_curve")
    ap.add_argument("--model", default="mlp", choices=["mlp", "resmlp", "transformer"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--embedding-dim", type=int, default=8)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of each TRAIN slice carved off for early stopping")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sizes", type=str, default=None,
                    help="comma-separated training sizes (overrides the default schedule)")
    ap.add_argument("--xgb-json", default=None,
                    help="optional learning_curve.json from XGBoost to overlay for comparison")
    args = ap.parse_args()

    set_seed(args.seed)

    # Same loading path as train_surrogate.py: positives + engineered features.
    ds = RocketDataset.from_jsonl(args.data, engineer_features=True)
    cont, cat, tgt = ds.continuous, ds.categorical, ds.targets
    n_cont = len(ds.continuous_names)
    print(f"Loaded {len(ds)} computable samples | continuous={n_cont} "
          f"({len(CONTINUOUS_FEATURES)} base + {n_cont - len(CONTINUOUS_FEATURES)} engineered) "
          f"| targets={len(TARGETS)}")

    # Fixed shuffle + fixed held-out test set (used for every slice).
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))
    n_test = int(len(ds) * args.test_frac)
    test_idx, pool_idx = perm[:n_test], perm[n_test:]
    n_pool = len(pool_idx)
    print(f"Held-out test: {n_test}   Train pool: {n_pool}")

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",") if int(s) <= n_pool]
        if n_pool not in sizes:
            sizes.append(n_pool)
    else:
        sizes = size_schedule(n_pool)
    print(f"Training sizes: {sizes}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "_ckpt")

    rows = []
    for n in sizes:
        t0 = time.time()
        sub = pool_idx[:n]
        n_val = max(1, int(n * args.val_frac))
        va_idx, tr_idx = sub[:n_val], sub[n_val:]

        # Scale inputs + targets on THIS slice's train rows only (no leakage).
        in_scaler = StandardScaler(); in_scaler.fit(cont[tr_idx])
        tg_scaler = StandardScaler(); tg_scaler.fit(tgt[tr_idx])
        cont_s = in_scaler.transform(cont)
        tgt_s = tg_scaler.transform(tgt)

        per_target = train_eval(cont_s, cat, tgt_s, tg_scaler, tr_idx, va_idx,
                                test_idx, n_cont, args, ckpt_dir)
        mean_r2 = float(np.mean(list(per_target.values())))
        rows.append({"n_train": int(len(tr_idx)), "mean_r2": mean_r2,
                     "per_target": per_target})
        keys = "  ".join(f"{k}={per_target[k]:.3f}" for k in KEY_TARGETS)
        print(f"  n_train={len(tr_idx):>6}  mean_R2={mean_r2:.4f}   "
              f"[{keys}]   ({time.time()-t0:.1f}s)")

    json_path = os.path.join(args.out_dir, "nn_learning_curve.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"sizes": sizes, "test_n": n_test, "seed": args.seed,
                   "model": args.model, "results": rows}, f, indent=2)
    _plot(rows, args.xgb_json, os.path.join(args.out_dir, "nn_learning_curve.png"))
    print(f"\nWrote {json_path}")
    print(f"Wrote {os.path.join(args.out_dir, 'nn_learning_curve.png')}")


def _plot(rows, xgb_json, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    for t in TARGETS:
        ax.plot(ns, [r["per_target"][t] for r in rows], color="0.85", lw=1, zorder=1)
    for t in KEY_TARGETS:
        ax.plot(ns, [r["per_target"][t] for r in rows], marker="o", lw=1.5,
                label=f"NN {t}", zorder=2)
    ax.plot(ns, [r["mean_r2"] for r in rows], color="black", marker="s", lw=2.5,
            label=f"NN mean ({len(TARGETS)} targets)", zorder=3)

    # Optional XGBoost overlay for direct comparison.
    if xgb_json and os.path.exists(xgb_json):
        with open(xgb_json, encoding="utf-8") as f:
            xgb = json.load(f)
        xns = [r["n_train"] for r in xgb["results"]]
        xmean = [r["mean_r2"] for r in xgb["results"]]
        ax.plot(xns, xmean, color="crimson", marker="^", lw=2.0, ls="--",
                label="XGBoost mean", zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("test $R^2$")
    ax.set_title("Neural surrogate learning curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
