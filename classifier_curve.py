#!/usr/bin/env python3
"""Sample-complexity study for the within_bounds (computability) classifier.

Mirrors learning_curve.py, but for the binary feasibility task: given a design,
is it computable (within_bounds=True) or not (False)? Trains XGBoost on growing
slices and plots ROC-AUC / PR-AUC vs. number of training samples, so we can size
how much labelled data the downstream LLM's feasibility gate actually needs.

Requires a dataset that contains BOTH classes (produced by run_with_monitor.py
now that it captures rejects).

Usage:
    python classifier_curve.py --inputs outputs/rocket_data_10k_sXXXX.jsonl \
                               --out-dir outputs/classifier_curve
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

XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "auc",
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


def _fingerprint(inp: dict) -> str:
    return hashlib.md5(
        "|".join(f"{k}={inp[k]}" for k in sorted(inp)).encode()).hexdigest()


def load_dedup(paths):
    seen, recs = set(), []
    for p in paths:
        for r in load_jsonl(p):
            fp = _fingerprint(r["input"])
            if fp in seen:
                continue
            seen.add(fp)
            recs.append(r)
    return recs


def roc_auc(y, p):
    """ROC-AUC via the rank (Mann-Whitney U) identity, no sklearn dependency."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = csum - (counts - 1) / 2.0
    ranks = avg[inv]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc(y, p):
    """Average precision (area under precision-recall), no sklearn dependency."""
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    total_pos = max(int(y.sum()), 1)
    recall = tp / total_pos
    ap = 0.0
    prev_r = 0.0
    for prec, rec in zip(precision, recall):
        ap += prec * (rec - prev_r)
        prev_r = rec
    return ap


def size_schedule(n_pool):
    base = [500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 48000, 64000]
    sizes = [s for s in base if s < n_pool]
    sizes.append(n_pool)
    return sizes


def main():
    ap = argparse.ArgumentParser(description="within_bounds classifier learning curve.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="outputs/classifier_curve")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = load_dedup(args.inputs)
    X, y, feat0, _ = extract_arrays(recs, schema.INPUT_FIELDS, classification=True)
    X, feat_names = add_engineered_features(X, feat0)
    pos, neg = int(y.sum()), int((y == 0).sum())
    print(f"Loaded {len(recs)} records: {pos} within_bounds / {neg} not-computable "
          f"({100*pos/len(recs):.1f}% positive)")
    if pos == 0 or neg == 0:
        print("ERROR: need both classes present to train a classifier.")
        return

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(recs))
    n_test = int(len(recs) * args.test_frac)
    test_idx, pool_idx = perm[:n_test], perm[n_test:]
    Xte = X.iloc[test_idx].reset_index(drop=True)
    yte = y[test_idx]
    dtest = xgb.DMatrix(Xte, feature_names=feat_names, enable_categorical=True)
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
        ytr = y[tr_idx]
        spw = max(1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum()))
        params = dict(XGB_PARAMS, scale_pos_weight=spw)
        dtrain = xgb.DMatrix(X.iloc[tr_idx].reset_index(drop=True), label=ytr,
                             feature_names=feat_names, enable_categorical=True)
        dval = xgb.DMatrix(X.iloc[va_idx].reset_index(drop=True), label=y[va_idx],
                           feature_names=feat_names, enable_categorical=True)
        model = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                          evals=[(dval, "val")],
                          callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True)],
                          verbose_eval=False)
        p = model.predict(dtest)
        auc, ap_ = roc_auc(yte, p), pr_auc(yte, p)
        rows.append({"n_train": int(len(tr_idx)), "roc_auc": float(auc), "pr_auc": float(ap_)})
        print(f"  n_train={len(tr_idx):>6}  ROC-AUC={auc:.4f}  PR-AUC={ap_:.4f}  ({time.time()-t0:.1f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "classifier_curve.json"), "w") as f:
        json.dump({"sizes": sizes, "test_n": n_test, "pos": pos, "neg": neg,
                   "results": rows}, f, indent=2)
    _plot(rows, os.path.join(args.out_dir, "classifier_curve.png"))
    print(f"\nWrote {args.out_dir}/classifier_curve.json + .png")


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ns, [r["roc_auc"] for r in rows], marker="o", lw=2, label="ROC-AUC")
    ax.plot(ns, [r["pr_auc"] for r in rows], marker="s", lw=2, label="PR-AUC")
    ax.set_xscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("test AUC")
    ax.set_title("within_bounds classifier learning curve")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
