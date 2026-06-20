#!/usr/bin/env python3
"""The torch differentiable feature block must reproduce the numpy/pandas
engineered features (gbt/preprocess.add_engineered_features) exactly, and be
differentiable w.r.t. the continuous raw inputs.

Run: python tests/test_diff_features.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "common"))
sys.path.insert(0, os.path.join(ROOT, "src", "rocket_sim"))
sys.path.insert(0, os.path.join(ROOT, "src", "gbt"))
sys.path.insert(0, os.path.join(ROOT, "src", "neural_surrogate"))

import torch                                    # noqa: E402
import pandas as pd                             # noqa: E402
import schema                                   # noqa: E402
from dataio import load_jsonl                   # noqa: E402
from preprocess import add_engineered_features  # noqa: E402
from optim.diff_features import (               # noqa: E402
    continuous_block, ENGINEERED_NAMES, CONTINUOUS_NAMES,
)

DATA = os.path.join(ROOT, "outputs", "rocket_data_2k_v2.jsonl")
N = 500


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  OK: {msg}")


def main():
    recs = [r for r in load_jsonl(DATA)
            if r.get("output", {}).get("within_bounds") is not False][:N]
    print(f"Loaded {len(recs)} records from {os.path.basename(DATA)}")

    cont = np.array([[float(r["input"][k]) for k in schema.INPUT_CONTINUOUS]
                     for r in recs], dtype=np.float64)
    cat = np.array([[schema.ENCODING_MAPS[k][r["input"][k]]
                     for k in schema.INPUT_CATEGORICAL] for r in recs], dtype=np.int64)

    x_cont = torch.tensor(cont, dtype=torch.float64, requires_grad=True)
    x_cat = torch.tensor(cat, dtype=torch.long)

    block = continuous_block(x_cont, x_cat)            # (B, 31) torch float64
    block_np = block.detach().numpy()

    # numpy ground truth (float64 from pandas, before any float32 cast)
    raw = pd.DataFrame([r["input"] for r in recs])[schema.INPUT_FIELDS]
    full, _ = add_engineered_features(raw, list(schema.INPUT_FIELDS))
    eng_np = full[ENGINEERED_NAMES].to_numpy(dtype=np.float64)

    # 1) base 17 continuous columns are passed through untouched
    base_err = np.max(np.abs(block_np[:, :17] - cont))
    check(base_err < 1e-9, f"base continuous columns identical (max abs err {base_err:.2e})")

    # 2) each engineered column matches numpy to ~machine precision
    worst = 0.0
    for j, name in enumerate(ENGINEERED_NAMES):
        a = block_np[:, 17 + j]
        b = eng_np[:, j]
        denom = np.maximum(np.abs(b), 1e-8)
        rel = np.max(np.abs(a - b) / denom)
        worst = max(worst, rel)
        check(rel < 1e-6, f"{name}: max rel err {rel:.2e}")
    print(f"  worst engineered rel err across all columns: {worst:.2e}")

    # 3) differentiable: autograd grad of barrowman margin matches central FD
    idx = CONTINUOUS_NAMES.index("barrowman_margin_cal")
    xi = x_cont[:1].detach().clone().requires_grad_(True)
    ci = x_cat[:1]
    m = continuous_block(xi, ci)[0, idx]
    m.backward()
    g_auto = xi.grad[0].detach().numpy()

    eps = 1e-6
    g_fd = np.zeros(17)
    base = xi.detach().clone()
    for k in range(17):
        xp = base.clone(); xp[0, k] += eps
        xm = base.clone(); xm[0, k] -= eps
        fp = continuous_block(xp, ci)[0, idx].item()
        fm = continuous_block(xm, ci)[0, idx].item()
        g_fd[k] = (fp - fm) / (2 * eps)

    scale = max(np.max(np.abs(g_fd)), 1e-6)
    grad_err = np.max(np.abs(g_auto - g_fd)) / scale
    check(grad_err < 1e-4,
          f"autograd grad of stability margin matches finite-diff (rel err {grad_err:.2e})")

    print("\nAll differentiable-feature checks passed.")


if __name__ == "__main__":
    main()
