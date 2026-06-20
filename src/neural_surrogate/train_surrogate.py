"""Main entry point: train a surrogate model on rocket simulation data.

Usage:
    python train_surrogate.py --data rocket_data.jsonl --model mlp --epochs 200
    python train_surrogate.py --data rocket_data.jsonl --model resmlp --epochs 300 --lr 5e-4
    python train_surrogate.py --data rocket_data.jsonl --model transformer --epochs 200
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.surrogate import (
    build_model,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_CARDINALITIES,
    TARGETS,
)
from data.dataset import RocketDataset
from training.trainer import Trainer

# `from utils.helpers import set_seed` is unsafe here: importing data.dataset above
# puts rocket_sim/ on sys.path, whose utils.py then shadows the neural_surrogate
# `utils` package (and importing ours first would poison the bare `utils` name for
# preprocess's `from utils import compute_cp_barrowman`). Load helpers by file path
# under a unique module name to sidestep the collision entirely.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "ns_helpers", str(Path(__file__).resolve().parent / "utils" / "helpers.py"))
_ns = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ns)
set_seed = _ns.set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a rocket surrogate model")
    p.add_argument("--data", type=str, required=True, help="Path to JSONL of simulation results")
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "resmlp", "transformer"])
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"],
                   help="Regression loss. Huber (delta=1.0) is robust to the heavy "
                        "max_acceleration tail; use it for the canonical model.")
    p.add_argument("--huber-delta", type=float, default=1.0,
                   help="Transition point for Huber loss (in scaled-target units).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-batch-norm", action="store_true",
                   help="Disable BatchNorm in the MLP/ResMLP. At eval BatchNorm is a "
                        "fixed affine, but dropping it gives cleaner second-order "
                        "behaviour for the downstream gradient-based optimizer.")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--save-dir", type=str, default=None,
                   help="If set, write a canonical, self-contained model bundle here "
                        "(model.pt + model_config.json + feature_config.json + scalers "
                        "+ model_metadata.json), mirroring models/surrogate/ for the trees.")
    p.add_argument("--hidden-dims", type=str, default="256,512,512,256,128",
                   help="Comma-separated hidden layer sizes (MLP only)")
    p.add_argument("--num-blocks", type=int, default=6,
                   help="Number of residual blocks (ResMLP only)")
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--use-amp", action="store_true",
                   help="Enable mixed precision (recommended for ROCm GPU training)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (set > 0 when training on GPU)")
    p.add_argument("--no-engineer", action="store_true",
                   help="Disable engineered features (Barrowman etc.); use raw inputs only")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Loading data from {args.data} ...")
    dataset = RocketDataset.from_jsonl(args.data, engineer_features=not args.no_engineer)
    n_base = len(CONTINUOUS_FEATURES)
    n_cont = len(dataset.continuous_names)
    print(f"  {len(dataset)} computable samples loaded (negatives dropped)")
    print(f"  Continuous features : {n_cont} ({n_base} base + {n_cont - n_base} engineered)")
    print(f"  Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"  Targets             : {len(TARGETS)}")
    if dataset.log1p_indices:
        print(f"  log1p targets       : {[TARGETS[i] for i in dataset.log1p_indices]}")

    # ── Split + scale + DataLoaders ───────────────────────────────────
    loaders = RocketDataset.make_loaders(
        dataset,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        batch_size=args.batch_size,
        scale_inputs=True,
        scale_targets=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(f"  Train / Val / Test : {len(loaders['train'].dataset)} / "
          f"{len(loaders['val'].dataset)} / {len(loaders['test'].dataset)}")

    # ── Build model ───────────────────────────────────────────────────
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    model_kwargs = {
        "continuous_dim": len(dataset.continuous_names),
        "categorical_cardinalities": CATEGORICAL_CARDINALITIES,
        "embedding_dim": args.embedding_dim,
        "output_dim": len(TARGETS),
        "dropout": args.dropout,
    }

    if args.model == "mlp":
        model_kwargs["hidden_dims"] = hidden_dims
        model_kwargs["use_batch_norm"] = not args.no_batch_norm
    elif args.model == "resmlp":
        model_kwargs["hidden_dim"] = hidden_dims[0] if hidden_dims else 256
        model_kwargs["num_blocks"] = args.num_blocks
        model_kwargs["use_batch_norm"] = not args.no_batch_norm
    elif args.model == "transformer":
        model_kwargs["embedding_dim"] = args.embedding_dim

    model = build_model(args.model, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | params: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────
    loss_fn = nn.HuberLoss(delta=args.huber_delta) if args.loss == "huber" else nn.MSELoss()
    print(f"Loss: {args.loss}" + (f" (delta={args.huber_delta})" if args.loss == "huber" else ""))
    trainer = Trainer(
        model=model,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler if args.scheduler != "none" else None,
        loss_fn=loss_fn,
        output_scaler=loaders["target_scaler"],
        use_amp=args.use_amp,
    )

    metrics = trainer.fit(
        loaders,
        epochs=args.epochs,
        patience=args.patience,
        ckpt_dir=args.ckpt_dir,
    )

    # ── Final test evaluation ─────────────────────────────────────────
    print("\n=== Test Set Evaluation ===")
    results = trainer.evaluate(loaders["test"], target_names=TARGETS, log1p_indices=dataset.log1p_indices)
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    # Save results + scalers + feature/target layout (needed to reproduce the
    # engineered continuous columns at inference time, in order).
    ckpt_path = Path(args.ckpt_dir)
    with open(ckpt_path / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    feature_config = {
        "continuous_features": dataset.continuous_names,
        "base_continuous_features": list(CONTINUOUS_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "targets": list(TARGETS),
        "engineered": not args.no_engineer,
        "embedding_dim": args.embedding_dim,
        "log1p_indices": dataset.log1p_indices,
        "log1p_targets": [TARGETS[i] for i in dataset.log1p_indices],
        "model": args.model,
    }
    with open(ckpt_path / "feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)
    loaders["input_scaler"].save(str(ckpt_path / "input_scaler.joblib"))
    loaders["target_scaler"].save(str(ckpt_path / "target_scaler.joblib"))

    print(f"\nDone. Checkpoints saved to {args.ckpt_dir}/")

    # ── Canonical, self-contained model bundle ────────────────────────
    # Mirrors models/surrogate/ for the trees: everything an inference / gradient
    # client needs to rebuild the exact model and reproduce the engineered columns,
    # in one directory, with no dependence on the transient checkpoints/ dir.
    if args.save_dir:
        save_canonical_bundle(
            save_dir=Path(args.save_dir),
            model=model,
            arch=args.model,
            model_kwargs=model_kwargs,
            feature_config=feature_config,
            input_scaler=loaders["input_scaler"],
            target_scaler=loaders["target_scaler"],
            results=results,
            args=args,
            n_params=n_params,
            split_sizes=(
                len(loaders["train"].dataset),
                len(loaders["val"].dataset),
                len(loaders["test"].dataset),
            ),
        )


def save_canonical_bundle(save_dir, model, arch, model_kwargs, feature_config,
                          input_scaler, target_scaler, results, args, n_params,
                          split_sizes):
    """Write a portable model bundle to ``save_dir`` (mirrors models/surrogate/).

    Contents:
      - model.pt              best weights (state_dict)
      - model_config.json     arch name + exact kwargs to rebuild via build_model
      - feature_config.json   continuous (base+engineered) / categorical / targets,
                              log1p layout, embedding dim — to rebuild input columns
      - input_scaler.joblib / target_scaler.joblib   fitted StandardScalers
      - model_metadata.json   per-target R^2/MAE/MAPE + training provenance
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "model.pt")

    with open(save_dir / "model_config.json", "w") as f:
        json.dump({"arch": arch, "kwargs": model_kwargs}, f, indent=2)

    with open(save_dir / "feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)

    input_scaler.save(str(save_dir / "input_scaler.joblib"))
    target_scaler.save(str(save_dir / "target_scaler.joblib"))

    n_train, n_val, n_test = split_sizes
    per_target = {}
    for name in TARGETS:
        per_target[name] = {
            "r2": results.get(f"r2_{name}"),
            "mae": results.get(f"mae_{name}"),
            "mape": results.get(f"mape_{name}"),
        }
    metadata = {
        "target_names": list(TARGETS),
        "feature_names": list(feature_config["continuous_features"])
                         + list(CATEGORICAL_FEATURES),
        "n_targets": len(TARGETS),
        "model_format": "torch_state_dict",
        "arch": arch,
        "n_params": int(n_params),
        "loss": args.loss,
        "use_batch_norm": not args.no_batch_norm,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "scheduler": args.scheduler,
        "data_path": args.data,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "scale_features": True,
        "scale_targets": True,
        "engineer_features": not args.no_engineer,
        "log1p_targets": feature_config["log1p_targets"],
        "overall_mae": results.get("overall_mae"),
        "mean_r2": float(np.mean([v["r2"] for v in per_target.values()
                                  if v["r2"] is not None])),
        "per_target_metrics": per_target,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(save_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Canonical bundle saved to {save_dir}/ "
          f"(mean R^2 {metadata['mean_r2']:.4f}, overall MAE {metadata['overall_mae']:.3f})")


if __name__ == "__main__":
    main()
