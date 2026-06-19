#!/usr/bin/env python3
"""Sample-complexity study: how does model accuracy scale with dataset size?

One tool, three modes (consolidates the former learning_curve.py /
classifier_curve.py / nn_learning_curve.py):

  --mode surrogate   XGBoost regression: test R^2 vs N, one model per target
                     (log1p on heavy-tailed targets, kept in sync with the trees)
  --mode classifier  within_bounds feasibility: ROC-AUC / PR-AUC vs N
  --mode nn          neural surrogate: test R^2 vs N (same pipeline as
                     train_surrogate.py; optional --xgb-json overlay)

Every mode trains on growing slices and evaluates each slice on ONE fixed
held-out test set, so the curve shows how many samples are needed before more
data stops paying off.

Usage:
    python learning_curve.py --mode surrogate  --inputs outputs/rocket_data_full.jsonl
    python learning_curve.py --mode classifier --inputs outputs/rocket_data_full.jsonl
    python learning_curve.py --mode nn --inputs outputs/rocket_data_full.jsonl --epochs 200 \
        --xgb-json outputs/learning_curve/learning_curve.json
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in ("common", "rocket_sim", "gbt"):
    sys.path.insert(0, os.path.join(ROOT, "src", _p))

# Targets highlighted individually on R^2 plots (the rest fold into the mean).
KEY_TARGETS = ["apogee_m", "max_velocity_mps", "max_mach", "stability_margin_calibers"]


# ── Shared helpers ──────────────────────────────────────────────────────────
def _fingerprint(inp: dict) -> str:
    return hashlib.md5(
        "|".join(f"{k}={inp[k]}" for k in sorted(inp)).encode()).hexdigest()


def load_dedup(paths):
    """Load all records from the given files, dropping exact-input duplicates."""
    seen, recs = set(), []
    for p in paths:
        for r in _load_jsonl(p):
            fp = _fingerprint(r["input"])
            if fp in seen:
                continue
            seen.add(fp)
            recs.append(r)
    return recs


def size_schedule(n_pool):
    """Log-spaced training sizes up to the full pool."""
    base = [500, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 48000, 64000]
    sizes = [s for s in base if s < n_pool]
    sizes.append(n_pool)
    return sizes


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def roc_auc(y, p):
    """ROC-AUC via the rank (Mann-Whitney U) identity, no sklearn dependency."""
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
    ap, prev_r = 0.0, 0.0
    for prec, rec in zip(precision, recall):
        ap += prec * (rec - prev_r)
        prev_r = rec
    return ap


# dataio.load_jsonl is imported lazily so `--help` works even if deps are missing.
def _load_jsonl(path):
    from dataio import load_jsonl
    return load_jsonl(path)


# ── Mode: XGBoost regression surrogate ──────────────────────────────────────
_XGB_REG = {
    "objective": "reg:squarederror", "tree_method": "hist", "eval_metric": "rmse",
    "verbosity": 0, "enable_categorical": True, "max_depth": 6, "eta": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 2.0,
}


def run_surrogate(args):
    import schema
    from data_loader import extract_arrays
    from preprocess import add_engineered_features
    from model import LOG1P_TARGETS
    import xgboost as xgb

    recs = load_dedup(args.inputs)
    print(f"Loaded {len(recs)} unique records from {len(args.inputs)} file(s).")
    target_names = schema.TARGETS
    X, Y, feat0, _ = extract_arrays(recs, schema.INPUT_FIELDS, target_names)
    X, feat_names = add_engineered_features(X, feat0)
    n_rows = len(Y)
    if n_rows < len(recs):
        print(f"  ({len(recs) - n_rows} non-computable records excluded; "
              f"{n_rows} positives used for regression)")
    print(f"Features: {len(feat_names)}  Targets: {len(target_names)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_rows)
    n_test = int(n_rows * args.test_frac)
    test_idx, pool_idx = perm[:n_test], perm[n_test:]
    Xte, Yte = X.iloc[test_idx].reset_index(drop=True), Y[test_idx]
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
        Xtr = X.iloc[tr_idx].reset_index(drop=True)
        Xva = X.iloc[va_idx].reset_index(drop=True)
        per_target = {}
        for i, name in enumerate(target_names):
            log_t = name in LOG1P_TARGETS
            ytr = np.log1p(Y[tr_idx, i]) if log_t else Y[tr_idx, i]
            yva = np.log1p(Y[va_idx, i]) if log_t else Y[va_idx, i]
            dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_names, enable_categorical=True)
            dva = xgb.DMatrix(Xva, label=yva, feature_names=feat_names, enable_categorical=True)
            mdl = xgb.train(_XGB_REG, dtr, num_boost_round=1000, evals=[(dva, "val")],
                            callbacks=[xgb.callback.EarlyStopping(rounds=30, save_best=True)],
                            verbose_eval=False)
            pred = mdl.predict(dtest)
            if log_t:
                pred = np.expm1(pred)
            per_target[name] = r2(Yte[:, i], pred)
        mean_r2 = float(np.mean(list(per_target.values())))
        rows.append({"n_train": int(len(tr_idx)), "mean_r2": mean_r2, "per_target": per_target})
        keys = "  ".join(f"{k}={per_target[k]:.3f}" for k in KEY_TARGETS)
        print(f"  n_train={len(tr_idx):>6}  mean_R2={mean_r2:.4f}   [{keys}]   ({time.time()-t0:.1f}s)")

    _write_r2(args, sizes, n_test, rows, target_names,
              "XGBoost surrogate learning curve", "learning_curve")


# ── Mode: within_bounds classifier ──────────────────────────────────────────
_XGB_CLF = {
    "objective": "binary:logistic", "tree_method": "hist", "eval_metric": "auc",
    "verbosity": 0, "enable_categorical": True, "max_depth": 6, "eta": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_lambda": 2.0,
}


def run_classifier(args):
    import schema
    from data_loader import extract_arrays
    from preprocess import add_engineered_features
    import xgboost as xgb

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
    Xte, yte = X.iloc[test_idx].reset_index(drop=True), y[test_idx]
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
        params = dict(_XGB_CLF, scale_pos_weight=spw)
        dtr = xgb.DMatrix(X.iloc[tr_idx].reset_index(drop=True), label=ytr,
                          feature_names=feat_names, enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_idx].reset_index(drop=True), label=y[va_idx],
                          feature_names=feat_names, enable_categorical=True)
        mdl = xgb.train(params, dtr, num_boost_round=1000, evals=[(dva, "val")],
                        callbacks=[xgb.callback.EarlyStopping(rounds=30, save_best=True)],
                        verbose_eval=False)
        p = mdl.predict(dtest)
        auc, ap_ = roc_auc(yte, p), pr_auc(yte, p)
        rows.append({"n_train": int(len(tr_idx)), "roc_auc": float(auc), "pr_auc": float(ap_)})
        print(f"  n_train={len(tr_idx):>6}  ROC-AUC={auc:.4f}  PR-AUC={ap_:.4f}  ({time.time()-t0:.1f}s)")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "classifier_curve.json"), "w") as f:
        json.dump({"sizes": sizes, "test_n": n_test, "pos": pos, "neg": neg, "results": rows}, f, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ns, [r["roc_auc"] for r in rows], marker="o", lw=2, label="ROC-AUC")
    ax.plot(ns, [r["pr_auc"] for r in rows], marker="s", lw=2, label="PR-AUC")
    ax.set_xscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("test AUC")
    ax.set_title("within_bounds classifier learning curve")
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "classifier_curve.png"), dpi=130)
    print(f"\nWrote {args.out_dir}/classifier_curve.json + .png")


# ── Mode: neural surrogate ──────────────────────────────────────────────────
def run_nn(args):
    sys.path.insert(0, os.path.join(ROOT, "src", "neural_surrogate"))
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from models.surrogate import (build_model, CONTINUOUS_FEATURES,
                                  CATEGORICAL_CARDINALITIES, TARGETS)
    from models.scalers import StandardScaler
    from data.dataset import RocketDataset
    from training.trainer import Trainer

    def _set_seed(s):  # inline: bare `utils` collides with rocket_sim/utils.py
        import random
        random.seed(s); np.random.seed(s); torch.manual_seed(s)

    _set_seed(args.seed)
    ds = RocketDataset.from_jsonl(args.inputs[0], engineer_features=True)
    cont, cat, tgt = ds.continuous, ds.categorical, ds.targets
    n_cont = len(ds.continuous_names)
    print(f"Loaded {len(ds)} computable samples | continuous={n_cont} | targets={len(TARGETS)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ds))
    n_test = int(len(ds) * args.test_frac)
    test_idx, pool_idx = perm[:n_test], perm[n_test:]
    n_pool = len(pool_idx)
    print(f"Held-out test: {n_test}   Train pool: {n_pool}")
    sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes else size_schedule(n_pool))
    if n_pool not in sizes:
        sizes.append(n_pool)
    print(f"Training sizes: {sizes}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "_ckpt")

    def make_loader(c, k, t, idx, shuffle):
        d = TensorDataset(torch.from_numpy(c[idx]).float(),
                          torch.from_numpy(k[idx]).long(),
                          torch.from_numpy(t[idx]).float())
        return DataLoader(d, batch_size=args.batch_size, shuffle=shuffle)

    rows = []
    for n in sizes:
        t0 = time.time()
        sub = pool_idx[:n]
        n_val = max(1, int(n * args.val_frac))
        va_idx, tr_idx = sub[:n_val], sub[n_val:]
        in_s = StandardScaler(); in_s.fit(cont[tr_idx])
        tg_s = StandardScaler(); tg_s.fit(tgt[tr_idx])
        cont_s, tgt_s = in_s.transform(cont), tg_s.transform(tgt)
        model = build_model(args.model, continuous_dim=n_cont,
                            categorical_cardinalities=CATEGORICAL_CARDINALITIES,
                            embedding_dim=args.embedding_dim, output_dim=len(TARGETS),
                            dropout=args.dropout)
        trainer = Trainer(model=model, device=args.device, lr=args.lr,
                          weight_decay=args.weight_decay, scheduler="cosine", output_scaler=tg_s)
        trainer.fit({"train": make_loader(cont_s, cat, tgt_s, tr_idx, True),
                     "val": make_loader(cont_s, cat, tgt_s, va_idx, False)},
                    epochs=args.epochs, patience=args.patience, log_every=10**9, ckpt_dir=ckpt_dir)
        res = trainer.evaluate(make_loader(cont_s, cat, tgt_s, test_idx, False),
                               target_names=TARGETS, log1p_indices=ds.log1p_indices)
        per_target = {t: res[f"r2_{t}"] for t in TARGETS}
        mean_r2 = float(np.mean(list(per_target.values())))
        rows.append({"n_train": int(len(tr_idx)), "mean_r2": mean_r2, "per_target": per_target})
        keys = "  ".join(f"{k}={per_target[k]:.3f}" for k in KEY_TARGETS)
        print(f"  n_train={len(tr_idx):>6}  mean_R2={mean_r2:.4f}   [{keys}]   ({time.time()-t0:.1f}s)")

    _write_r2(args, sizes, n_test, rows, list(TARGETS),
              "Neural surrogate learning curve", "nn_learning_curve", xgb_json=args.xgb_json)


# ── Shared R^2 output (surrogate + nn) ──────────────────────────────────────
def _write_r2(args, sizes, n_test, rows, target_names, title, stem, xgb_json=None):
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"sizes": sizes, "test_n": n_test, "seed": args.seed, "results": rows}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    for t in target_names:
        ax.plot(ns, [r["per_target"][t] for r in rows], color="0.85", lw=1, zorder=1)
    for t in KEY_TARGETS:
        ax.plot(ns, [r["per_target"][t] for r in rows], marker="o", lw=1.5, label=t, zorder=2)
    ax.plot(ns, [r["mean_r2"] for r in rows], color="black", marker="s", lw=2.5,
            label=f"mean ({len(target_names)} targets)", zorder=3)
    if xgb_json and os.path.exists(xgb_json):
        with open(xgb_json, encoding="utf-8") as f:
            xj = json.load(f)
        ax.plot([r["n_train"] for r in xj["results"]], [r["mean_r2"] for r in xj["results"]],
                color="crimson", marker="^", lw=2.0, ls="--", label="XGBoost mean", zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("training samples"); ax.set_ylabel("test $R^2$")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, f"{stem}.png"), dpi=130)
    print(f"\nWrote {json_path}")
    print(f"Wrote {os.path.join(args.out_dir, f'{stem}.png')}")


def main():
    ap = argparse.ArgumentParser(description="Model accuracy vs. dataset size (3 modes).")
    ap.add_argument("--mode", choices=["surrogate", "classifier", "nn"], default="surrogate")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="JSONL file(s). nn mode uses the first only.")
    ap.add_argument("--out-dir", default=None, help="default: outputs/<mode>_curve")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    # nn-only
    ap.add_argument("--model", default="mlp", choices=["mlp", "resmlp", "transformer"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--embedding-dim", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sizes", default=None, help="comma-separated sizes (nn mode)")
    ap.add_argument("--xgb-json", default=None, help="learning_curve.json to overlay (nn mode)")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("outputs", f"{args.mode}_curve")
    {"surrogate": run_surrogate, "classifier": run_classifier, "nn": run_nn}[args.mode](args)


if __name__ == "__main__":
    main()
