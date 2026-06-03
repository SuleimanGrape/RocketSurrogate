#!/usr/bin/env python3
"""Run 10 rocket simulations with random parameters, then train XGBoost.

This script:
  1. Runs 10 rocket simulations with random parameters
  2. Saves results as JSONL (compatible with the XGBoost pipeline)
  3. Trains XGBoost on the existing 100-record dataset
  4. Evaluates the trained model on the 10 new simulations

Usage:
    python run_sim_and_train.py [--seed 42] [--sim-output outputs/new_10.jsonl]
"""

import argparse
import json
import os
import sys
import time

# ── Rocket simulation imports ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rocket_sim'))

import parameters as params_mod
import rocket_builder
import simulator
import validator
import outputs as outputs_mod
from utils import compute_cp_barrowman, stability_margin_calibers

# ── XGBoost pipeline imports ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'gbt'))

from data_loader import load_jsonl, extract_arrays, train_val_test_split
from preprocess import preprocess, add_engineered_features
from model import train_multi_target, save_models, BASE_PARAMS
from evaluate import evaluate_all, predict_all
from synthetic_data import INPUT_FEATURES, TARGET_FEATURES


def run_simulations(n=10, seed=42):
    """Run n rocket simulations with random parameters.

    Returns:
        records: list of {"input": {...}, "output": {...}} dicts for successful sims
        all_results: list of all result dicts (including failures) for diagnostics
    """
    print("=" * 60)
    print(f"Running {n} Rocket Simulations (seed={seed})")
    print("=" * 60)

    # Sample extra to account for rejections
    all_params = params_mod.balanced_sample(30, seed=seed)
    params = []
    for p in all_params:
        ok, _ = validator.prevalidate(p)
        if ok:
            params.append(p)
        if len(params) >= n:
            break

    print(f"  Pre-validated: {len(params)}/{len(all_params)} passed\n")

    records = []
    all_results = []
    for i, p in enumerate(params):
        record = {"rocket_id": i + 1, "input": {}, "output": {}, "status": "pending"}
        try:
            rocket = rocket_builder.build_rocket(p)
            flight = simulator.run_simulation(rocket, p)

            if flight is None:
                record["status"] = "simulation_failed_or_timeout"
                all_results.append(record)
                print(f"  [{i+1:>2}/{n}] FAILED/TIMEOUT")
                continue

            if not validator.is_valid(p, flight):
                record["status"] = "post_validation_failed"
                all_results.append(record)
                print(f"  [{i+1:>2}/{n}] POST-VALIDATION FAILED")
                continue

            out = outputs_mod.extract_output(p, flight)
            inp = outputs_mod.extract_input(p)
            cg, cp = compute_cp_barrowman(p)
            sm = stability_margin_calibers(cg, cp, p["diameter_mm"])
            out["cg_m"] = round(cg, 4)
            out["cp_m"] = round(cp, 4)
            out["stability_margin_calibers"] = round(sm, 2)

            record["input"] = inp
            record["output"] = out
            record["status"] = "success"
            records.append({"input": inp, "output": out})
            all_results.append(record)
            print(f"  [{i+1:>2}/{n}] OK  apogee={out['apogee_m']:.0f}m  "
                  f"mach={out['max_mach']:.2f}  sm={sm:.2f}cal")

        except Exception as e:
            record["status"] = f"error: {e}"
            all_results.append(record)
            print(f"  [{i+1:>2}/{n}] ERROR: {e}")

    print(f"\n  Successful: {len(records)}/{len(params)}")
    return records, all_results


def save_jsonl(records, path):
    """Save records as JSONL."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(records)} records to {path}")


def train_xgboost(data_path, output_dir="models", plots_dir="plots", seed=42):
    """Train XGBoost models on the given JSONL data."""
    print(f"\n{'=' * 60}")
    print("Training XGBoost Models")
    print(f"{'=' * 60}")

    # Load data
    records = load_jsonl(data_path)
    print(f"  Loaded {len(records)} records from {data_path}")

    # Extract arrays (XGBoost handles categorical features natively)
    X, Y, feature_names, target_names = extract_arrays(
        records,
        input_features=INPUT_FEATURES,
        output_targets=TARGET_FEATURES,
    )
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Targets  ({len(target_names)}): {target_names}")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")

    # Split
    splits = train_val_test_split(
        X, Y,
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=seed,
    )
    for name, (x, y) in splits.items():
        print(f"  {name:5s}: {x.shape[0]} samples")

    # Preprocess
    proc = preprocess(
        splits,
        feature_names=feature_names,
        scale_features=False,
        scale_targets=False,
        engineer_features=True,
    )
    X_train = proc["X_train"]
    Y_train = proc["Y_train"]
    X_val = proc["X_val"]
    Y_val = proc["Y_val"]
    X_test = proc["X_test"]
    Y_test = proc["Y_test"]
    feature_names_used = proc["feature_names"]
    print(f"  Engineered features: {len(feature_names_used)} total")

    # Train (no tuning for speed — use sensible defaults)
    train_result = train_multi_target(
        X_train, Y_train,
        X_val, Y_val,
        target_names=target_names,
        tune=False,
        seed=seed,
        feature_names=feature_names_used,
    )
    models = train_result["models"]
    infos = train_result["infos"]

    # Evaluate on all splits
    for split_name, X_split, Y_split in [
        ("train", X_train, Y_train),
        ("val", X_val, Y_val),
        ("test", X_test, Y_test),
    ]:
        eval_results = evaluate_all(
            models, X_split, Y_split,
            target_names=target_names,
            split_name=split_name,
            feature_names=feature_names_used,
        )

    # Save models
    save_models(
        models, target_names,
        output_dir=output_dir,
        infos=infos,
        feature_names=feature_names_used,
        metadata={
            "data_path": data_path,
            "n_train": X_train.shape[0],
            "n_val": X_val.shape[0],
            "n_test": X_test.shape[0],
            "feature_names": feature_names_used,
        },
    )

    return models, feature_names_used, target_names


def predict_on_new(models, feature_names, target_names, new_records):
    """Run trained models on new simulation records and show predictions vs actual."""
    if not new_records:
        print("  No new records to predict on.")
        return

    X_new, Y_new, _, _ = extract_arrays(
        new_records,
        input_features=INPUT_FEATURES,
        output_targets=TARGET_FEATURES,
    )

    # Apply same preprocessing (feature engineering only, no scaling)
    X_new, _ = add_engineered_features(X_new, INPUT_FEATURES)

    preds = predict_all(models, X_new, feature_names)

    print(f"\n{'=' * 60}")
    print("Predictions on New Simulations")
    print(f"{'=' * 60}")
    print(f"{'Target':<25} {'Actual':>12} {'Predicted':>12} {'Error%':>10}")
    print("-" * 65)
    for i, name in enumerate(target_names):
        for j in range(Y_new.shape[0]):
            actual = Y_new[j, i]
            predicted = preds[j, i]
            err_pct = abs(actual - predicted) / (abs(actual) + 1e-9) * 100
            tag = f"  [#{j+1}]" if len(target_names) > 1 else ""
            print(f"{name:<25} {actual:>12.2f} {predicted:>12.2f} {err_pct:>9.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run 10 rocket simulations and train XGBoost."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sim-output", type=str, default="outputs/new_10_simulations.jsonl",
                        help="Path to save new simulation JSONL")
    parser.add_argument("--train-data", type=str, default="outputs/rocket_data.jsonl",
                        help="Path to training data JSONL")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save trained models")
    args = parser.parse_args()

    t_start = time.time()

    # ── Step 1: Run 10 simulations ─────────────────────────────────────
    records, all_results = run_simulations(n=10, seed=args.seed)

    if not records:
        print("\nERROR: No simulations succeeded. Cannot proceed.")
        sys.exit(1)

    save_jsonl(records, args.sim_output)

    # ── Step 2: Train XGBoost on existing dataset ────────────────────
    if not os.path.exists(args.train_data):
        print(f"\nWARNING: Training data not found at {args.train_data}")
        print("Skipping XGBoost training.")
        sys.exit(0)

    models, feature_names, target_names = train_xgboost(
        args.train_data,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # ── Step 3: Predict on new simulations ─────────────────────────────
    predict_on_new(models, feature_names, target_names, records)

    # ── Summary ────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE — Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")
    print(f"  New simulations : {args.sim_output}")
    print(f"  Training data   : {args.train_data}")
    print(f"  Models saved to : {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
