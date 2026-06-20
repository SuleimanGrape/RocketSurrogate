#!/usr/bin/env python3
"""Train the production within_bounds (computability) classifier.

Given a candidate design, predict whether RocketPy can actually compute a valid
flight for it (within_bounds=True) or not (False, i.e. rejected by pre-validation
or by the sim). This is the feasibility gate the downstream LLM consults before
proposing a rocket we have no data for.

Same input features and engineered (incl. Barrowman) features as the neural
surrogate (shared from common/features), so a single feature pipeline serves
both. Trains XGBoost binary:logistic with scale_pos_weight for the class
imbalance, evaluates on a
held-out test split (ROC-AUC, PR-AUC, accuracy, confusion at a chosen
threshold), and saves the model + metadata.

Usage:
    python src/gbt/train_classifier.py --data outputs/rocket_data_full.jsonl \
        --output-dir models/classifier --plots-dir plots/classifier
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "common"))
sys.path.insert(0, os.path.join(ROOT, "src", "rocket_sim"))
sys.path.insert(0, os.path.join(ROOT, "src", "gbt"))

import schema                                   # noqa: E402
from dataio import load_jsonl                   # noqa: E402
from data_loader import extract_arrays          # noqa: E402
from features import add_engineered_features    # noqa: E402  (shared, in common/)
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
NUM_BOOST_ROUND = 2000
EARLY_STOP = 50


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
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(ys.sum()), 1)
    ap, prev_r = 0.0, 0.0
    for prec, rec in zip(precision, recall):
        ap += prec * (rec - prev_r)
        prev_r = rec
    return ap


def threshold_for_fpr(y_val, p_val, target_fpr):
    """Smallest decision threshold whose false-positive rate on the validation
    negatives is <= target_fpr (maximizes recall subject to the FP budget).

    A false positive here = predicting within_bounds=True for a design that is
    actually not computable, i.e. the score of a negative exceeds the threshold.
    So the threshold is the (1 - target_fpr) quantile of the negatives' scores.
    """
    neg_scores = p_val[y_val == 0]
    if len(neg_scores) == 0:
        return 0.5
    thr = float(np.quantile(neg_scores, 1.0 - target_fpr))
    # Nudge above the quantile so exactly-equal negative scores fall below it.
    return min(1.0, np.nextafter(thr, 1.0))


def confusion(y, p, thresh):
    pred = (p >= thresh).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"threshold": thresh, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    ap = argparse.ArgumentParser(description="Train within_bounds classifier.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", default="models/classifier")
    ap.add_argument("--plots-dir", default="plots/classifier")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="decision threshold for the reported confusion matrix")
    ap.add_argument("--target-fpr", type=float, default=0.01,
                    help="pick a deployment threshold on the validation set whose "
                         "false-positive rate (predicting computable when it is "
                         "not) is <= this; 0 disables and keeps --threshold")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    recs = load_jsonl(args.data)
    X, y, feat0, _ = extract_arrays(recs, schema.INPUT_FIELDS, classification=True)
    X, feat_names = add_engineered_features(X, feat0)
    pos, neg = int(y.sum()), int((y == 0).sum())
    print(f"Loaded {len(recs)} records: {pos} within_bounds / {neg} not-computable "
          f"({100 * pos / len(recs):.1f}% positive)")
    if pos == 0 or neg == 0:
        print("ERROR: need both classes present to train a classifier.")
        return
    print(f"Features ({len(feat_names)})")

    # Fixed train/val/test split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(recs))
    n_test = int(len(recs) * args.test_frac)
    n_val = int(len(recs) * args.val_frac)
    te_idx = perm[:n_test]
    va_idx = perm[n_test:n_test + n_val]
    tr_idx = perm[n_test + n_val:]
    print(f"Split — train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}")

    def dm(idx, label=True):
        return xgb.DMatrix(X.iloc[idx].reset_index(drop=True),
                           label=y[idx] if label else None,
                           feature_names=feat_names, enable_categorical=True)

    ytr = y[tr_idx]
    spw = max(1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum()))
    params = dict(XGB_PARAMS, scale_pos_weight=spw)
    print(f"scale_pos_weight = {spw:.3f}")

    dtrain, dval, dtest = dm(tr_idx), dm(va_idx), dm(te_idx)
    model = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND,
                      evals=[(dtrain, "train"), (dval, "val")],
                      callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOP, save_best=True)],
                      verbose_eval=False)
    print(f"Best iteration: {model.best_iteration}  ({time.time() - t0:.1f}s)")

    # ── Evaluate on the held-out test set ─────────────────────────────────
    yte = y[te_idx]
    p = model.predict(dtest)
    auc, ap_ = roc_auc(yte, p), pr_auc(yte, p)
    cm = confusion(yte, p, args.threshold)
    print(f"\nTest ROC-AUC={auc:.4f}  PR-AUC={ap_:.4f}")
    print(f"At threshold={args.threshold}:  acc={cm['accuracy']:.4f}  "
          f"precision={cm['precision']:.4f}  recall={cm['recall']:.4f}  f1={cm['f1']:.4f}")
    print(f"  TP={cm['tp']}  TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}")

    # ── Pick a low-FP deployment threshold on validation, report on test ───
    deploy_thr, cm_deploy = args.threshold, cm
    if args.target_fpr and args.target_fpr > 0:
        p_val = model.predict(dval)
        deploy_thr = threshold_for_fpr(y[va_idx], p_val, args.target_fpr)
        cm_deploy = confusion(yte, p, deploy_thr)
        test_fpr = cm_deploy["fp"] / max(cm_deploy["fp"] + cm_deploy["tn"], 1)
        print(f"\nDeployment threshold for <= {args.target_fpr:.1%} FPR "
              f"(tuned on val): {deploy_thr:.4f}")
        print(f"At threshold={deploy_thr:.4f}:  acc={cm_deploy['accuracy']:.4f}  "
              f"precision={cm_deploy['precision']:.4f}  recall={cm_deploy['recall']:.4f}  "
              f"f1={cm_deploy['f1']:.4f}")
        print(f"  TP={cm_deploy['tp']}  TN={cm_deploy['tn']}  "
              f"FP={cm_deploy['fp']}  FN={cm_deploy['fn']}  (test FPR={test_fpr:.3%})")

    # ── Save model + metadata ─────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "xgb_within_bounds.json")
    model.save_model(model_path)
    meta = {
        "task": "within_bounds_classifier",
        "data_path": args.data,
        "model_format": "xgboost_json",
        "feature_names": feat_names,
        "categorical_cols": list(schema.XGB_CATEGORICAL_COLS),
        "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
        "pos_total": pos, "neg_total": neg,
        "scale_pos_weight": float(spw),
        "best_iteration": int(model.best_iteration),
        "test_roc_auc": float(auc), "test_pr_auc": float(ap_),
        "test_confusion": cm,
        "target_fpr": float(args.target_fpr),
        "deploy_threshold": float(deploy_thr),
        "test_confusion_deploy": cm_deploy,
        "params": params,
    }
    meta_path = os.path.join(args.output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model    -> {model_path}")
    print(f"Saved metadata -> {meta_path}")

    _plot_diagnostics(yte, p, model, feat_names, args.plots_dir)


def _plot_diagnostics(y, p, model, feat_names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # ROC + PR curves.
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    n_pos, n_neg = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)
    tpr, fpr = tp / n_pos, fp / n_neg
    precision = tp / np.maximum(tp + fp, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(fpr, tpr, lw=2)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("false positive rate")
    ax1.set_ylabel("true positive rate")
    ax1.set_title(f"ROC (AUC={roc_auc(y, p):.4f})")
    ax1.grid(True, alpha=0.3)

    ax2.plot(tp / n_pos, precision, lw=2)
    ax2.axhline(n_pos / (n_pos + n_neg), color="k", ls="--", lw=1, alpha=0.5,
                label="baseline (prevalence)")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title(f"Precision-Recall (AP={pr_auc(y, p):.4f})")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "classifier_roc_pr.png"), dpi=140)
    plt.close(fig)

    # Feature importance (gain).
    imp = model.get_score(importance_type="gain")
    if imp:
        items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:15]
        names = []
        for k, _ in items:
            if k.startswith("f") and k[1:].isdigit() and int(k[1:]) < len(feat_names):
                names.append(feat_names[int(k[1:])])
            else:
                names.append(k)
        vals = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(8, 6))
        ypos = np.arange(len(names))
        ax.barh(ypos, vals, color="indianred", alpha=0.85)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("gain")
        ax.set_title("within_bounds — feature importance")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "classifier_importance.png"), dpi=140)
        plt.close(fig)
    print(f"Saved plots    -> {out_dir}")


if __name__ == "__main__":
    main()
