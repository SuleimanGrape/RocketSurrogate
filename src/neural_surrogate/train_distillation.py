"""Train a neural network via knowledge distillation from an XGBoost teacher.

Pipeline:
1. Load the XGBoost model bundle (trained in the other session).
2. Generate soft labels on the training set (and optionally augmented data).
3. Train the NN with combined hard + soft loss.

Usage:
    python train_distillation.py --data rocket_data.jsonl \\
        --teacher checkpoints/xgboost/xgboost_bundle.joblib \\
        --model mlp --epochs 200 --alpha 0.3

ROCm:
    On a cloud AMD GPU machine, add --device cuda (or auto-detect).
    The DistillationTrainer inherits ROCm-compatible device handling from Trainer.
"""

from __future__ import annotations

import argparse
import json
import joblib
import numpy as np
from pathlib import Path

from models.surrogate import (
    build_model,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_CARDINALITIES,
    TARGETS,
    ENCODING_MAPS,
)
from data.dataset import RocketDataset, make_splits
from training.distillation_trainer import DistillationTrainer, DistillationDataset
from utils.helpers import set_seed
from utils.data_augmentation import augment_data, chunked_generator
from models.scalers import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description="Train NN via XGBoost distillation")
    p.add_argument("--data", type=str, required=True, help="Path to JSONL")
    p.add_argument("--teacher", type=str, required=True, help="Path to XGBoost bundle (.joblib)")
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "resmlp", "transformer"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.3,
                   help="Weight for hard label loss (1-alpha for soft)")
    p.add_argument("--augment", type=int, default=0,
                   help="Number of augmented copies per sample (0 = no augmentation)")
    p.add_argument("--noise-std", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/distilled")
    p.add_argument("--use-amp", action="store_true",
                   help="Enable mixed precision (recommended for ROCm GPU training)")
    p.add_argument("--hidden-dims", type=str, default="256,512,512,256,128")
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (set > 0 when training on GPU)")
    return p.parse_args()


def load_teacher(path: str) -> dict:
    """Load XGBoost bundle: {target_name: XGBRegressor, 'scaler_in', 'scaler_tgt'}."""
    bundle = joblib.load(path)
    print(f"  Loaded teacher bundle with targets: {sorted(k for k in bundle if k not in ('scaler_in', 'scaler_tgt'))}")
    return bundle


def generate_soft_labels(
    teacher: dict,
    continuous: np.ndarray,
    categorical: np.ndarray,
    target_keys: list,
    scaler_in: StandardScaler = None,
) -> np.ndarray:
    """Run XGBoost teacher inference to produce soft labels.

    The teacher was trained on UN-scaled features (XGBoost needs no scaling),
    but if a scaler is provided we still transform for consistency.
    Returns (N, D_tgt) array of teacher predictions.
    """
    n = len(continuous)
    soft = np.zeros((n, len(target_keys)), dtype=np.float32)

    # Build feature matrix: continuous (unscaled for XGBoost) + categorical as codes
    X = np.concatenate([continuous, categorical.astype(np.float32)], axis=1)

    for i, tgt in enumerate(target_keys):
        model = teacher.get(tgt)
        if model is None:
            print(f"  WARNING: teacher has no model for target '{tgt}', using zeros")
            continue
        # XGBoost predict handles DataFrames or ndarrays
        soft[:, i] = model.predict(X)

    return soft


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[{'='*60}")
    print(f"  Neural Network Knowledge Distillation Trainer")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data from {args.data} ...")
    dataset = RocketDataset.from_jsonl(args.data)
    print(f"  {len(dataset)} samples")

    # ── Split ──────────────────────────────────────────────────────────
    print(f"\n[2/5] Splitting data ...")
    train_idx, val_idx, test_idx = make_splits(len(dataset), seed=args.seed)
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Fit scalers on training data
    scaler_in = StandardScaler().fit(dataset.continuous[train_idx])
    scaler_tgt = StandardScaler().fit(dataset.targets[train_idx])

    # Scale the full dataset
    cont_scaled = scaler_in.transform(dataset.continuous)
    cat = dataset.categorical  # already int codes
    tgt_scaled = scaler_tgt.transform(dataset.targets)

    # ── Load teacher ───────────────────────────────────────────────────
    print(f"\n[3/5] Loading XGBoost teacher from {args.teacher} ...")
    teacher = load_teacher(args.teacher)

    # ── Generate soft labels ──────────────────────────────────────────
    print(f"\n[4/5] Generating soft labels from teacher ...")

    # For distillation, we only need soft labels on the training set
    train_cont = cont_scaled[train_idx]
    train_cat = cat[train_idx]
    train_tgt = tgt_scaled[train_idx]

    # Optional augmentation
    if args.augment > 0:
        print(f"  Augmenting with {args.augment} copies per sample (noise_std={args.noise_std}) ...")
        train_cont, train_cat, train_tgt = augment_data(
            train_cont, train_cat, train_tgt,
            n_augmented=args.augment, noise_std=args.noise_std, seed=args.seed,
        )

    # Generate teacher predictions
    # XGBoost was trained with raw features + categorical codes as float
    # We use the SAME format: continuous (scaled back to raw for teacher) + cat codes
    teacher_cont = scaler_in.inverse_transform(train_cont)  # revert to raw for XGBoost
    soft_labels = generate_soft_labels(teacher, teacher_cont, train_cat, TARGETS, scaler_in)
    soft_labels_scaled = scaler_tgt.transform(soft_labels)

    # Build distillation datasets
    # Train: has both hard and soft labels
    train_ds = DistillationDataset(train_cont, train_cat, train_tgt, soft_labels_scaled)

    # Val: only needs hard labels for monitoring; use zeros for soft (ignored in val)
    val_cont = cont_scaled[val_idx]
    val_cat = cat[val_idx]
    val_tgt = tgt_scaled[val_idx]
    val_soft = np.zeros_like(val_tgt)  # placeholder, val loss uses hard labels only
    val_ds = DistillationDataset(val_cont, val_cat, val_tgt, val_soft)

    # Test: same as val
    test_cont = cont_scaled[test_idx]
    test_cat = cat[test_idx]
    test_tgt = tgt_scaled[test_idx]
    test_soft = np.zeros_like(test_tgt)
    test_ds = DistillationDataset(test_cont, test_cat, test_tgt, test_soft)

    pin = True if args.device == "cuda" or (args.device == "auto" and __import__("torch").cuda.is_available()) else False
    train_loader = __import__("torch").utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
    )
    val_loader = __import__("torch").utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )
    test_loader = __import__("torch").utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "input_scaler": scaler_in,
        "target_scaler": scaler_tgt,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }

    # ──────────────────────────────────────────────────────────────────
    # ── Build model ───────────────────────────────────────────────────
    print(f"\n[5/5] Building and training model ({args.model}) ...")
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    model_kwargs = {
        "continuous_dim": len(CONTINUOUS_FEATURES),
        "categorical_cardinalities": CATEGORICAL_CARDINALITIES,
        "embedding_dim": args.embedding_dim,
        "output_dim": len(TARGETS),
        "dropout": args.dropout,
    }
    if args.model == "mlp":
        model_kwargs["hidden_dims"] = hidden_dims
    elif args.model == "resmlp":
        model_kwargs["hidden_dim"] = hidden_dims[0] if hidden_dims else 256
        model_kwargs["num_blocks"] = args.num_blocks

    model = build_model(args.model, **model_kwargs)

    trainer = DistillationTrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler if args.scheduler != "none" else None,
        output_scaler=scaler_tgt,
        use_amp=args.use_amp,
        alpha=args.alpha,
    )

    metrics = trainer.fit(loaders, epochs=args.epochs, patience=args.patience, ckpt_dir=args.ckpt_dir)

    # ── Final test evaluation ─────────────────────────────────────────
    print("\n=== Test Set Evaluation ===")
    results = trainer.evaluate(loaders["test"], target_names=TARGETS)
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")

    ckpt_path = Path(args.ckpt_dir)
    with open(ckpt_path / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    scaler_in.save(str(ckpt_path / "input_scaler.joblib"))
    scaler_tgt.save(str(ckpt_path / "target_scaler.joblib"))

    print(f"\nDone. Checkpoints saved to {args.ckpt_dir}/")


if __name__ == "__main__":
    main()
