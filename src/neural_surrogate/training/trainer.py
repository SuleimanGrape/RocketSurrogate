"""Training loop with early stopping, LR scheduling, metric logging.

ROCm compatibility:
- Device auto-detection via torch.cuda.is_available() (works on ROCm — AMD GPUs
  expose the same cuda API in PyTorch ROCm builds).
- All tensors use .to(self.device), never hardcoded .cuda().
- Mixed precision via torch.amp.autocast('cuda') — works on both CUDA and ROCm.
- pin_memory=True in DataLoaders works on ROCm.
"""

from __future__ import annotations

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models.scalers import StandardScaler


def _auto_device(requested: str) -> torch.device:
    """Resolve device string to torch.device.

    ROCm note: PyTorch ROCm builds expose AMD GPUs through the standard
    ``torch.cuda`` API, so ``torch.cuda.is_available()`` returns True and
    ``torch.device('cuda')`` targets the AMD GPU. No special handling needed.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS for Apple Silicon laptops (local dev fallback)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class MetricsTracker:
    """Collect per-epoch metrics and provide summary stats."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.history.setdefault(k, []).append(float(v))

    def summary(self, last_n: int = 5) -> Dict[str, float]:
        return {k: np.mean(v[-last_n:]) for k, v in self.history.items()}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


class EarlyStopping:
    """Stop training when val loss hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """High-level training orchestrator.

    Usage:
        trainer = Trainer(model, device="auto")
        trainer.fit(loaders, epochs=200)

    ROCm notes:
        - Pass device="cuda" or device="auto" on an ROCm machine to use the AMD GPU.
        - Mixed precision (use_amp=True) uses torch.amp.autocast('cuda') which is
          supported on both NVIDIA CUDA and AMD ROCm.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        scheduler_T0: int = 50,
        loss_fn: Optional[nn.Module] = None,
        grad_clip: float = 1.0,
        output_scaler: Optional[StandardScaler] = None,
        use_amp: bool = False,
    ):
        self.device = _auto_device(device)
        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = loss_fn or nn.MSELoss()
        self.grad_clip = grad_clip
        self.output_scaler = output_scaler
        self.use_amp = use_amp and self.device.type == "cuda"

        if scheduler == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T0=scheduler_T0, T_mult=2)
        elif scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        else:
            self.scheduler = None

        self.metrics = MetricsTracker()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def fit(
        self,
        loaders: Dict[str, DataLoader],
        epochs: int = 200,
        patience: int = 30,
        log_every: int = 10,
        ckpt_dir: str = "checkpoints",
    ) -> MetricsTracker:
        """Train on loaders["train"], validate on loaders["val"]."""
        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        early_stop = EarlyStopping(patience=patience)

        print(f"Training on {self.device} | {epochs} epochs | patience={patience}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.use_amp:
            print("  Mixed precision: ENABLED (torch.amp.autocast)")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_mae = self._run_epoch(loaders["train"], train=True)
            val_loss, val_mae = self._run_epoch(loaders["val"], train=False)

            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step()
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            elapsed = time.time() - t0
            self.metrics.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_mae=train_mae,
                val_mae=val_mae,
                lr=self.optimizer.param_groups[0]["lr"],
                time_s=elapsed,
            )

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:>4d}/{epochs}  "
                    f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                    f"val_mae={val_mae:.4f}  lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                    f"({elapsed:.1f}s)"
                )

            if val_loss <= early_stop.best:
                self.save_checkpoint(ckpt_path / "best.pt")

            if early_stop(val_loss):
                print(f"  Early stopping at epoch {epoch} (best val_loss={early_stop.best:.6f})")
                break

        self.load_checkpoint(ckpt_path / "best.pt")
        self.metrics.save(str(ckpt_path / "metrics.json"))
        return self.metrics

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> Tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_mae = 0.0
        n = 0

        # ROCm note: torch.amp.autocast('cuda') works on both CUDA and ROCm; it is
        # applied directly in each branch below (eval also wraps in torch.no_grad()).
        if train:
            for cont, cat, tgt in loader:
                cont = cont.to(self.device, non_blocking=True)
                cat = cat.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pred = self.model(cont, cat)
                    loss = self.loss_fn(pred, tgt)

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                batch = cont.shape[0]
                total_loss += loss.item() * batch
                total_mae += torch.abs(pred - tgt).sum().item()
                n += batch
        else:
            with torch.no_grad():
                for cont, cat, tgt in loader:
                    cont = cont.to(self.device, non_blocking=True)
                    cat = cat.to(self.device, non_blocking=True)
                    tgt = tgt.to(self.device, non_blocking=True)

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        pred = self.model(cont, cat)
                        loss = self.loss_fn(pred, tgt)

                    batch = cont.shape[0]
                    total_loss += loss.item() * batch
                    total_mae += torch.abs(pred - tgt).sum().item()
                    n += batch

        avg_loss = total_loss / max(n, 1)
        avg_mae = total_mae / max(n, 1)
        return avg_loss, avg_mae

    # ------------------------------------------------------------------
    # Evaluation on test set
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Return per-target MAE and overall metrics."""
        self.model.eval()
        all_preds, all_targets = [], []

        for cont, cat, tgt in loader:
            cont = cont.to(self.device, non_blocking=True)
            cat = cat.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred = self.model(cont, cat)

            all_preds.append(pred.float().cpu().numpy())
            all_targets.append(tgt.numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        if self.output_scaler:
            preds = self.output_scaler.inverse_transform(preds)
            targets = self.output_scaler.inverse_transform(targets)

        mae_per_target = np.mean(np.abs(preds - targets), axis=0)
        results = {"overall_mae": float(np.mean(mae_per_target))}

        if target_names:
            for i, name in enumerate(target_names):
                results[f"mae_{name}"] = float(mae_per_target[i])

        ss_res = np.sum((targets - preds) ** 2, axis=0)
        ss_tot = np.sum((targets - targets.mean(axis=0)) ** 2, axis=0)
        r2 = 1 - ss_res / np.maximum(ss_tot, 1e-8)
        for i, name in enumerate(target_names or range(len(r2))):
            results[f"r2_{name}"] = float(r2[i])

        return results

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics_history": self.metrics.history,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
