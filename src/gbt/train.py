#!/usr/bin/env python3
"""
Main entry point: generate data (or load real data), preprocess,
train XGBoost models, evaluate, and save everything.

Usage:
    # Train with synthetic data (no real data needed):
    python train.py

    # Train with real data:
    python train.py --data path/to/rocket_data.jsonl

    # Skip hyperparameter tuning for faster results:
    python train.py --no-tune
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from synthetic_data import INPUT_FEATURES, TARGET_FEATURES, generate as generate_synthetic
from data_loader import load_jsonl, extract_arrays, train_val_test_split
from preprocess import preprocess
from model import train_multi_target, save_models, BASE_PARAMS
from evaluate import (
    evaluate_all,
    predict_all,
    plot_predictions,
    plot_residuals,
    plot_feature_importance,
)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost models on rocket data.")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to JSONL data file. If not given, generates synthetic data.")
    parser.add_argument("--generate", type=int, default=5000,
                        help="Number of synthetic samples to generate (default: 5000)")
    parser.add_argument("--targets", type=str, nargs="+", default=None,
                        help="Target output columns to predict (default: all)")
    parser.add_argument("--input-features", type=str, nargs="+", default=None,
                        help="Input feature columns to use (default: all)")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save trained models (default: models/)")
    parser.add_argument("--plots-dir", type=str, default="plots",
                        help="Directory to save evaluation plots (default: plots/)")
    parser.add_argument("--tune", action="store_true", default=True,
                        help="Enable hyperparameter tuning (default: on)")
    parser.add_argument("--no-tune", action="store_false", dest="tune",
                        help="Disable hyperparameter tuning for faster training")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Random search trials per target (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--scale-features", action="store_true", default=False,
                        help="Apply z-score scaling to features")
    parser.add_argument("--engineer-features", action="store_true", default=True,
                        help="Add engineered features (default: on)")
    args = parser.parse_args()

    t_start = time.time()

    # ── Step 1: Data ───────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Data")
    print("=" * 60)
    if args.data is None:
        print("No --data provided. Generating synthetic rocket data...\n")
        data_path = generate_synthetic(count=args.generate, seed=args.seed)
    else:
        data_path = args.data
        print(f"Loading real data from {data_path}")

    records = load_jsonl(data_path)
    print(f"Loaded {len(records)} records")

    # Use all input features (XGBoost handles categorical natively via enable_categorical=True)
    input_features = args.input_features if args.input_features is not None else INPUT_FEATURES
    output_targets = args.targets if args.targets is not None else TARGET_FEATURES

    X, Y, feature_names, target_names = extract_arrays(
        records,
        input_features=input_features,
        output_targets=output_targets,
    )
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Targets  ({len(target_names)}): {target_names}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # ── Step 2: Split ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2: Train/Val/Test Split")
    print(f"{'='*60}")
    splits = train_val_test_split(
        X, Y,
        train_frac=1.0 - args.val_frac - args.test_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    for name, (x, y) in splits.items():
        print(f"  {name:5s}: {x.shape[0]} samples")

    # ── Step 3: Preprocess ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: Preprocessing")
    print(f"{'='*60}")
    proc = preprocess(
        splits,
        feature_names=feature_names,
        scale_features=args.scale_features,
        scale_targets=False,
        engineer_features=args.engineer_features,
    )
    X_train = proc["X_train"]
    Y_train = proc["Y_train"]
    X_val = proc["X_val"]
    Y_val = proc["Y_val"]
    X_test = proc["X_test"]
    Y_test = proc["Y_test"]
    feature_names_used = proc["feature_names"]
    print(f"  Feature scaling : {args.scale_features}")
    print(f"  Engineered feats: {args.engineer_features}")
    print(f"  Features used   : {len(feature_names_used)}")
    print(f"  X_train shape   : {X_train.shape}")
    print(f"  X_val shape     : {X_val.shape}")
    print(f"  X_test shape    : {X_test.shape}")

    # ── Step 4: Train ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4: Training XGBoost Models")
    print(f"{'='*60}")
    train_result = train_multi_target(
        X_train, Y_train,
        X_val, Y_val,
        target_names=target_names,
        tune=args.tune,
        n_trials=args.n_trials,
        seed=args.seed,
        feature_names=feature_names_used,
    )
    models = train_result["models"]
    infos = train_result["infos"]

    # ── Step 5: Evaluate ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 5: Evaluation")
    print(f"{'='*60}")

    # Evaluate on all splits
    for split_name, X_split, Y_split in [("train", X_train, Y_train), ("val", X_val, Y_val), ("test", X_test, Y_test)]:
        eval_results = evaluate_all(
            models, X_split, Y_split,
            target_names=target_names,
            split_name=split_name,
            feature_names=feature_names_used,
        )

        # Generate plots only on test set
        if split_name == "test":
            preds = predict_all(models, X_split, feature_names_used)
            plot_predictions(Y_split, preds, target_names, args.plots_dir, split_name)
            plot_residuals(Y_split, preds, target_names, args.plots_dir, split_name)
            plot_feature_importance(models, target_names, feature_names_used, args.plots_dir)

    # ── Step 6: Save ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 6: Save Models & Metadata")
    print(f"{'='*60}")
    save_models(
        models, target_names,
        output_dir=args.output_dir,
        infos=infos,
        feature_names=feature_names_used,
        metadata={
            "data_path": data_path,
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
            "n_test": X_test.shape[0],
            "feature_names": feature_names_used,
            "scale_features": args.scale_features,
            "engineer_features": args.engineer_features,
        },
    )

    # ── Summary ────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE — Total time: {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"Models saved to : {args.output_dir}/")
    print(f"Plots saved to  : {args.plots_dir}/")


if __name__ == "__main__":
    main()
