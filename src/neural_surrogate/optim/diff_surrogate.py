"""End-to-end differentiable surrogate for gradient-based design optimization.

Loads a canonical model bundle (written by train_surrogate.py --save-dir) and
exposes the full inference pipeline as a single differentiable torch module:

    raw design params ─▶ engineered features (torch) ─▶ input scaling
        ─▶ network ─▶ target un-scaling ─▶ expm1(log1p targets)
        ─▶ flight metrics in natural units

Because every stage is torch, ``d(metric)/d(raw continuous input)`` flows by
autograd — which is exactly what a gradient design optimizer consumes. The
categorical inputs (diameter / nose / fin count / motor) are discrete and held
fixed per design; gradients are w.r.t. the 17 continuous raw inputs only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))            # neural_surrogate (models.*)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "common"))

from models.surrogate import build_model               # noqa: E402
from scalers import StandardScaler                      # noqa: E402
from optim.diff_features import continuous_block, CONTINUOUS_NAMES, CONT, CAT  # noqa: E402

# Closed-form (Barrowman) targets and the exact engineered column each equals.
# With class1_exact=True these outputs are spliced from the analytic feature
# columns, giving perfect value AND gradient — the production configuration for
# the constraint targets, since approximating a known closed form with a NN only
# loses accuracy (cg_m's gradient in particular is hard for a shared backbone).
CLASS1_EXACT = {
    "cg_m": "barrowman_cg_m",
    "cp_m": "barrowman_cp_m",
    "stability_margin_calibers": "barrowman_margin_cal",
}


class DifferentiableSurrogate(nn.Module):
    """Differentiable raw-inputs → natural-unit flight metrics surrogate."""

    def __init__(self, bundle_dir: str, device: str = "cpu", class1_exact: bool = False):
        super().__init__()
        bundle = Path(bundle_dir)
        self.device = torch.device(device)
        self.class1_exact = class1_exact

        with open(bundle / "model_config.json") as f:
            cfg = json.load(f)
        with open(bundle / "feature_config.json") as f:
            self.feature_config = json.load(f)

        self.model = build_model(cfg["arch"], **cfg["kwargs"])
        state = torch.load(bundle / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

        in_scaler = StandardScaler().load(str(bundle / "input_scaler.joblib"))
        tgt_scaler = StandardScaler().load(str(bundle / "target_scaler.joblib"))
        # Buffers so .to(device) / state are handled uniformly. float32 matches
        # the network weights; feature engineering runs in float64 internally.
        self.register_buffer("in_mean", torch.tensor(in_scaler.mean_, dtype=torch.float32))
        self.register_buffer("in_std", torch.tensor(in_scaler.std_, dtype=torch.float32))
        self.register_buffer("tgt_mean", torch.tensor(tgt_scaler.mean_, dtype=torch.float32))
        self.register_buffer("tgt_std", torch.tensor(tgt_scaler.std_, dtype=torch.float32))

        self.targets: List[str] = list(self.feature_config["targets"])
        self.log1p_indices: List[int] = list(self.feature_config.get("log1p_indices", []))
        self.continuous_inputs = list(CONT)   # 17 raw continuous, the grad variables
        self.categorical_inputs = list(CAT)
        # (output index, engineered-column index) pairs for the exact-splice mode.
        self._class1_splice = [(self.targets.index(t), CONTINUOUS_NAMES.index(c))
                               for t, c in CLASS1_EXACT.items()]
        self.to(self.device)

    # ------------------------------------------------------------------
    def forward(self, x_cont_raw: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """(B,17) raw continuous + (B,4) codes → (B,12) metrics in natural units."""
        cont = continuous_block(x_cont_raw.to(torch.float32), x_cat)     # (B,31)
        scaled = (cont - self.in_mean) / self.in_std
        out = self.model(scaled, x_cat)                                  # scaled targets
        out = out * self.tgt_std + self.tgt_mean                         # natural (log space for log1p targets)
        if self.log1p_indices or self.class1_exact:
            cols = list(out.unbind(dim=1))
            for i in self.log1p_indices:
                cols[i] = torch.expm1(cols[i])
            if self.class1_exact:
                # Replace the closed-form targets with their exact (and exactly
                # differentiable) analytic feature columns.
                for out_i, col_i in self._class1_splice:
                    cols[out_i] = cont[:, col_i]
            out = torch.stack(cols, dim=1)
        return out

    # ------------------------------------------------------------------
    def jacobian(self, x_cont_raw: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """d(metric)/d(raw continuous input) — shape (B, n_targets, 17).

        Uses the sum-over-batch trick: outputs of row b depend only on inputs of
        row b, so grad of sum_b out[b,t] w.r.t. the input batch is row-wise exact.
        """
        x = x_cont_raw.detach().clone().to(torch.float32).requires_grad_(True)
        out = self.forward(x, x_cat)
        rows = []
        for t in range(out.shape[1]):
            g = torch.autograd.grad(out[:, t].sum(), x, retain_graph=True)[0]
            rows.append(g)
        return torch.stack(rows, dim=1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x_cont_raw: torch.Tensor, x_cat: torch.Tensor):
        """Convenience no-grad forward returning a detached tensor."""
        return self.forward(x_cont_raw, x_cat)

    def target_index(self, name: str) -> int:
        return self.targets.index(name)


def load_surrogate(bundle_dir: Optional[str] = None, device: str = "cpu",
                   class1_exact: bool = False) -> DifferentiableSurrogate:
    """Load the canonical bundle (defaults to models/neural/).

    Set class1_exact=True for the recommended optimizer configuration: the NN
    serves the 9 flight-dynamics targets while cg/cp/stability are taken from the
    exact analytic Barrowman (perfect value and gradient).
    """
    if bundle_dir is None:
        bundle_dir = os.path.join(_HERE, "..", "..", "..", "models", "neural")
    return DifferentiableSurrogate(bundle_dir, device=device, class1_exact=class1_exact)
