"""Knowledge distillation trainer: soft labels from XGBoost teacher → neural network student.

Loss = α * MSE(pred, y_hard) + (1 - α) * MSE(pred, y_soft)

where y_hard are the true simulator outputs and y_soft are XGBoost predictions
(on original and optionally augmented data).

ROCm compatibility: inherits from Trainer which handles ROCm device selection,
mixed precision (AMP), and non_blocking transfers. This module is device-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from training.trainer import Trainer, _auto_device
from models.scalers import StandardScaler


class DistillationDataset(Dataset):
    """Dataset that provides (continuous, categorical, y_hard, y_soft) tuples."""

    def __init__(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
        y_hard: np.ndarray,
        y_soft: np.ndarray,
    ):
        assert len(continuous) == len(categorical) == len(y_hard) == len(y_soft)
        self.continuous = continuous.astype(np.float32)
        self.categorical = categorical.astype(np.int64)
        self.y_hard = y_hard.astype(np.float32)
        self.y_soft = y_soft.astype(np.float32)

    def __len__(self):
        return len(self.y_hard)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.continuous[idx]),
            torch.from_numpy(self.categorical[idx]),
            torch.from_numpy(self.y_hard[idx]),
            torch.from_numpy(self.y_soft[idx]),
        )


class DistillationTrainer(Trainer):
    """Extends Trainer with knowledge distillation loss.

    Parameters
    ----------
    alpha : float
        Weight for the hard label loss. (1 - alpha) is the weight for soft labels.
        Default 0.3 means 30% hard, 70% soft.
    temperature : float
        Not used for regression (only relevant for classification softmax).
        Kept for API consistency, ignored.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        loss_fn: Optional[nn.Module] = None,
        grad_clip: float = 1.0,
        output_scaler: Optional[StandardScaler] = None,
        use_amp: bool = False,
        alpha: float = 0.3,
    ):
        super().__init__(
            model=model,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            scheduler=scheduler,
            loss_fn=loss_fn or nn.MSELoss(),
            grad_clip=grad_clip,
            output_scaler=output_scaler,
            use_amp=use_amp,
        )
        self.alpha = alpha

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> Tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_mae = 0.0
        n = 0

        if train:
            for cont, cat, y_hard, y_soft in loader:
                cont = cont.to(self.device, non_blocking=True)
                cat = cat.to(self.device, non_blocking=True)
                y_hard = y_hard.to(self.device, non_blocking=True)
                y_soft = y_soft.to(self.device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    pred = self.model(cont, cat)
                    loss_hard = self.loss_fn(pred, y_hard)
                    loss_soft = self.loss_fn(pred, y_soft)
                    loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                batch = cont.shape[0]
                total_loss += loss.item() * batch
                total_mae += torch.abs(pred - y_hard).sum().item()
                n += batch
        else:
            with torch.no_grad():
                for cont, cat, y_hard, y_soft in loader:
                    cont = cont.to(self.device, non_blocking=True)
                    cat = cat.to(self.device, non_blocking=True)
                    y_hard = y_hard.to(self.device, non_blocking=True)
                    y_soft = y_soft.to(self.device, non_blocking=True)

                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        pred = self.model(cont, cat)
                        loss_hard = self.loss_fn(pred, y_hard)
                        loss_soft = self.loss_fn(pred, y_soft)
                        loss = self.alpha * loss_hard + (1.0 - self.alpha) * loss_soft

                    batch = cont.shape[0]
                    total_loss += loss.item() * batch
                    total_mae += torch.abs(pred - y_hard).sum().item()
                    n += batch

        avg_loss = total_loss / max(n, 1)
        avg_mae = total_mae / max(n, 1)
        return avg_loss, avg_mae

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate against hard (ground truth) labels."""
        self.model.eval()
        all_preds, all_targets = [], []

        for cont, cat, y_hard, _y_soft in loader:
            cont = cont.to(self.device, non_blocking=True)
            cat = cat.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred = self.model(cont, cat)

            all_preds.append(pred.float().cpu().numpy())
            all_targets.append(y_hard.numpy())

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
