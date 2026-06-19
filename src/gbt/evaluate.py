#!/usr/bin/env python3
"""Model evaluation: metrics, plots, and feature importance analysis."""

import os
import numpy as np
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from model import LOG1P_TARGETS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics for a single target."""
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / (y_true + 1e-12))) * 100)
    return {"r2": round(r2, 6), "rmse": round(rmse, 6), "mae": round(mae, 6), "mape": round(mape, 4)}


def predict_all(
    models: List[xgb.Booster],
    X,
    feature_names: Optional[List[str]] = None,
    target_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Run prediction for all target models. Returns (n_samples, n_targets).

    Targets in LOG1P_TARGETS were fit on log1p(y); pass `target_names` so their
    predictions are inverted (expm1) back to the original units here, the single
    point every consumer routes through.
    """
    dmatrix = xgb.DMatrix(X, feature_names=feature_names, enable_categorical=True)
    preds = np.column_stack([m.predict(dmatrix) for m in models])
    if target_names is not None:
        for i, name in enumerate(target_names):
            if name in LOG1P_TARGETS:
                preds[:, i] = np.expm1(preds[:, i])
    return preds


def evaluate_all(
    models: List[xgb.Booster],
    X,
    Y: np.ndarray,
    target_names: List[str],
    split_name: str = "test",
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """Evaluate all target models on a dataset. Returns per-target metrics."""
    preds = predict_all(models, X, feature_names, target_names)
    results = {}
    print(f"\n{'='*60}")
    print(f"Evaluation on {split_name} set  ({X.shape[0]} samples)")
    print(f"{'='*60}")
    print(f"{'Target':<25} {'R²':>8} {'RMSE':>12} {'MAE':>12} {'MAPE%':>8}")
    print("-" * 70)
    for i, name in enumerate(target_names):
        m = compute_metrics(Y[:, i], preds[:, i])
        results[name] = m
        print(f"{name:<25} {m['r2']:>8.4f} {m['rmse']:>12.4f} {m['mae']:>12.4f} {m['mape']:>8.2f}")
    return results


def plot_predictions(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_names: List[str],
    output_dir: str,
    split_name: str = "test",
):
    """Scatter plots: predicted vs actual for each target."""
    os.makedirs(output_dir, exist_ok=True)
    n_targets = len(target_names)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, name in enumerate(target_names):
        ax = axes[i]
        yt = Y_true[:, i]
        yp = Y_pred[:, i]
        ax.scatter(yt, yp, alpha=0.3, s=8, c="steelblue")
        # Perfect prediction line
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1.5, label="Perfect")
        r2 = 1.0 - np.sum((yt - yp) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}  (R²={r2:.4f})")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Predicted vs Actual — {split_name} set", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"predictions_{split_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved prediction plots -> {path}")


def plot_residuals(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_names: List[str],
    output_dir: str,
    split_name: str = "test",
):
    """Residual distribution plots for each target."""
    os.makedirs(output_dir, exist_ok=True)
    n_targets = len(target_names)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, name in enumerate(target_names):
        ax = axes[i]
        residuals = Y_true[:, i] - Y_pred[:, i]
        ax.hist(residuals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}  (μ={residuals.mean():.3f}, σ={residuals.std():.3f})")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Residual Distributions — {split_name} set", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"residuals_{split_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved residual plots ->{path}")


def plot_feature_importance(
    models: List[xgb.Booster],
    target_names: List[str],
    feature_names: List[str],
    output_dir: str,
    top_n: int = 15,
):
    """Feature importance (gain) plots for each target model."""
    os.makedirs(output_dir, exist_ok=True)

    for model, tname in zip(models, target_names):
        importance = model.get_score(importance_type="gain")
        if not importance:
            continue

        # XGBoost may use f0,f1,... or real feature names depending on how DMatrix was built
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        names = []
        for k, _ in sorted_imp:
            if k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                names.append(feature_names[idx] if idx < len(feature_names) else k)
            else:
                names.append(k)
        values = [v for _, v in sorted_imp]

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color="steelblue", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Gain")
        ax.set_title(f"Feature Importance — {tname}")
        ax.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        safe_name = tname.replace("/", "_").replace(" ", "_")
        path = os.path.join(output_dir, f"importance_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved feature importance plots ->{output_dir}")


def plot_training_curves(
    training_infos: List[Dict],
    output_dir: str,
):
    """Plot random search trial results for each target."""
    os.makedirs(output_dir, exist_ok=True)

    for info in training_infos:
        if "all_trials" not in info:
            continue
        trials = info["all_trials"]
        target = info["target"]

        fig, ax = plt.subplots(figsize=(10, 4))
        rmses = [t["val_rmse"] for t in trials]
        best_so_far = np.minimum.accumulate(rmses)
        ax.plot(range(1, len(rmses) + 1), rmses, "bo-", alpha=0.5, label="Trial RMSE")
        ax.plot(range(1, len(rmses) + 1), best_so_far, "r-", lw=2, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Validation RMSE")
        ax.set_title(f"Hyperparameter Search — {target}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        safe_name = target.replace("/", "_").replace(" ", "_")
        path = os.path.join(output_dir, f"tuning_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved tuning curves ->{output_dir}")
