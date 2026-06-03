#!/usr/bin/env python3
"""XGBoost model training, hyperparameter tuning, and persistence."""

import json
import os
import time
import numpy as np
import xgboost as xgb
from typing import Dict, List, Optional, Tuple


# ── Default hyperparameter search space ──────────────────────────────────
DEFAULT_PARAM_GRID = {
    "n_estimators":  [200, 500, 1000],
    "max_depth":     [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample":     [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha":     [0, 0.1, 1.0],
    "reg_lambda":    [1.0, 2.0, 5.0],
}

# Base params always used
# enable_categorical=True allows XGBoost to handle categorical features natively
BASE_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "verbosity": 0,
    "enable_categorical": True,
}


def train_single(
    X_train,
    Y_train: np.ndarray,
    X_val,
    Y_val: np.ndarray,
    target_idx: int,
    target_name: str,
    params: Dict,
    feature_names: Optional[List[str]] = None,
) -> Tuple[xgb.Booster, Dict]:
    """Train a single XGBoost model for one target column."""
    dtrain = xgb.DMatrix(X_train, label=Y_train[:, target_idx], feature_names=feature_names, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=Y_val[:, target_idx], feature_names=feature_names, enable_categorical=True)

    evals = [(dtrain, "train"), (dval, "val")]
    callbacks = [xgb.callback.EarlyStopping(rounds=50, save_best=True)]

    t0 = time.time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.pop("num_boost_round", 10000),
        evals=evals,
        callbacks=callbacks,
        verbose_eval=False,
    )
    train_time = time.time() - t0

    # Best validation score
    best_val_rmse = model.best_score if hasattr(model, 'best_score') else float('inf')
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else 0

    info = {
        "target": target_name,
        "train_time_s": round(train_time, 2),
        "best_iteration": best_iteration,
        "best_val_rmse": round(best_val_rmse, 6),
        "params": params.copy(),
    }

    return model, info


def random_search(
    X_train,
    Y_train: np.ndarray,
    X_val,
    Y_val: np.ndarray,
    target_idx: int,
    target_name: str,
    param_grid: Dict,
    n_trials: int = 20,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> Tuple[xgb.Booster, Dict]:
    """
    Random search over the hyperparameter grid.
    Returns the best model and its info dict.
    """
    rng = np.random.default_rng(seed)
    best_model = None
    best_score = float("inf")
    best_params = {}
    best_info = {}
    trial_results = []

    print(f"\n  Tuning '{target_name}' with {n_trials} random search trials...")

    for trial in range(n_trials):
        # Sample params
        params = {k: rng.choice(v) for k, v in param_grid.items()}
        params_full = {**BASE_PARAMS, **params, "num_boost_round": 5000}

        dtrain = xgb.DMatrix(X_train, label=Y_train[:, target_idx], feature_names=feature_names, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=Y_val[:, target_idx], feature_names=feature_names, enable_categorical=True)

        model = xgb.train(
            params_full,
            dtrain,
            num_boost_round=5000,
            evals=[(dtrain, "train"), (dval, "val")],
            callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)],
            verbose_eval=False,
        )

        val_rmse = model.best_score if hasattr(model, 'best_score') else float('inf')
        trial_results.append({**params, "val_rmse": round(val_rmse, 6)})

        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model
            best_params = params.copy()
            best_info = {
                "target": target_name,
                "best_val_rmse": round(val_rmse, 6),
                "best_iteration": model.best_iteration,
                "best_params": params,
                "n_trials": n_trials,
            }
            print(f"    Trial {trial+1}/{n_trials}: NEW BEST val_rmse={val_rmse:.6f}  params={params}")
        else:
            print(f"    Trial {trial+1}/{n_trials}: val_rmse={val_rmse:.6f}")

    best_info["all_trials"] = trial_results
    return best_model, best_info


def train_multi_target(
    X_train,
    Y_train: np.ndarray,
    X_val,
    Y_val: np.ndarray,
    target_names: List[str],
    params: Optional[Dict] = None,
    tune: bool = True,
    n_trials: int = 20,
    param_grid: Optional[Dict] = None,
    seed: int = 42,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Train one XGBoost model per target output.

    Args:
        X_train, Y_train: training data
        X_val, Y_val: validation data
        target_names: names of output columns
        params: fixed params (used if tune=False)
        tune: whether to run random search
        n_trials: number of random search trials per target
        param_grid: custom param grid for tuning
        seed: random seed
        feature_names: names for XGBoost feature importances

    Returns:
        Dict with "models" (list of Booster), "infos" (list of dicts), "target_names"
    """
    models = []
    infos = []

    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    for i, name in enumerate(target_names):
        print(f"\n{'='*60}")
        print(f"Target {i+1}/{len(target_names)}: {name}")
        print(f"{'='*60}")

        if tune:
            model, info = random_search(
                X_train, Y_train, X_val, Y_val,
                target_idx=i, target_name=name,
                param_grid=grid, n_trials=n_trials, seed=seed + i,
                feature_names=feature_names,
            )
        else:
            p = {**BASE_PARAMS, **(params or {}), "num_boost_round": 10000}
            model, info = train_single(
                X_train, Y_train, X_val, Y_val,
                target_idx=i, target_name=name,
                params=p.copy(),
                feature_names=feature_names,
            )
            print(f"  Trained '{name}': val_rmse={info['best_val_rmse']:.6f}  time={info['train_time_s']}s")

        models.append(model)
        infos.append(info)

    return {"models": models, "infos": infos, "target_names": target_names}


def save_models(
    models: List[xgb.Booster],
    target_names: List[str],
    output_dir: str,
    infos: Optional[List[Dict]] = None,
    feature_names: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
):
    """Save trained models and metadata to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save each model in XGBoost native format
    for model, name in zip(models, target_names):
        path = os.path.join(output_dir, f"xgb_{name}.json")
        model.save_model(path)
        print(f"  Saved model for '{name}' ->{path}")

    # Save metadata
    meta = {
        "target_names": target_names,
        "feature_names": feature_names,
        "n_targets": len(target_names),
        "model_format": "xgboost_json",
    }
    if infos:
        meta["training_infos"] = infos
    if metadata:
        meta.update(metadata)

    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        # numpy types aren't JSON serializable
        json.dump(_sanitize_json(meta), f, indent=2)
    print(f"  Saved metadata ->{meta_path}")


def load_models(model_dir: str) -> Dict:
    """Load trained models and metadata from disk."""
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    models = []
    for name in meta["target_names"]:
        path = os.path.join(model_dir, f"xgb_{name}.json")
        model = xgb.Booster()
        model.load_model(path)
        models.append(model)

    return {"models": models, **meta}


def _sanitize_json(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    elif isinstance(obj, (_np.integer,)):
        return int(obj)
    elif isinstance(obj, (_np.floating,)):
        return float(obj)
    elif isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    return obj
