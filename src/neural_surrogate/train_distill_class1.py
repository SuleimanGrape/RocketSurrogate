"""Formula distillation with Sobolev (derivative) matching for the Class-1 targets.

Step 2's gradient-quality eval showed the catch with a plain value-trained
surrogate: cg_m's *value* is near-perfect (R²≈1.0) while its *gradient* is wrong
(cosine≈0.67). A gradient-based optimizer follows the gradient, so that is the
metric that must be fixed.

cg_m, cp_m and stability_margin_calibers are exact closed-form functions of the
design (Barrowman), so we can synthesise unlimited training data with *both* the
exact value and the exact derivative — no simulator needed. This script
fine-tunes the canonical model with three losses:

  1. Real-corpus replay (Huber on all 12 targets, scaled space) — keeps the 9
     flight-dynamics targets from drifting (they have no closed form to distill).
  2. Synthetic Class-1 value loss — MSE(pred, exact) for cg/cp/margin.
  3. Synthetic Class-1 Sobolev loss — MSE(d pred/d input, d exact/d input), the
     term that actually repairs the gradient field. Requires double backprop
     (create_graph=True).

The synthetic generator (parameters.balanced_sample + exact torch Barrowman) is
CPU-cheap and scales to millions of rows; --synth / --epochs / --device let the
same script run a much larger pass on the $100 AMD GPU.

Usage:
    python train_distill_class1.py --bundle ../../models/neural \
        --data ../../outputs/rocket_data_full.jsonl --save-dir ../../models/neural_distilled
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# rocket_sim before neural_surrogate so the bare `utils` import inside
# parameters.py binds to rocket_sim/utils.py (Barrowman), not neural's utils pkg.
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
sys.path.insert(0, os.path.join(_HERE, "..", "rocket_sim"))

import schema                       # noqa: E402
import parameters                   # noqa: E402  (binds `utils` -> rocket_sim/utils)
from dataio import load_jsonl       # noqa: E402

import torch                        # noqa: E402
import torch.nn as nn               # noqa: E402

from optim.diff_surrogate import DifferentiableSurrogate            # noqa: E402
from optim.diff_features import continuous_block, CONTINUOUS_NAMES, CONT  # noqa: E402

# Class-1 targets and the exact engineered column each equals.
CLASS1 = {
    "stability_margin_calibers": "barrowman_margin_cal",
    "cg_m": "barrowman_cg_m",
    "cp_m": "barrowman_cp_m",
}
CLASS2 = [t for t in schema.TARGETS if t not in CLASS1]


# ── data helpers ──────────────────────────────────────────────────────────────
def make_splits(n, train=0.70, val=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nt, nv = int(n * train), int(n * val)
    return idx[:nt], idx[nt:nt + nv], idx[nt + nv:]


def load_corpus(path):
    recs = [r for r in load_jsonl(path)
            if r.get("output", {}).get("within_bounds") is not False]
    cont = np.array([[float(r["input"][k]) for k in schema.INPUT_CONTINUOUS]
                     for r in recs], dtype=np.float32)
    cat = np.array([[schema.ENCODING_MAPS[k][r["input"][k]]
                     for k in schema.INPUT_CATEGORICAL] for r in recs], dtype=np.int64)
    tgt = np.array([[float(r["output"][k]) for k in schema.TARGETS]
                    for r in recs], dtype=np.float32)
    return cont, cat, tgt


def gen_synth(n, seed):
    """n realistic in-distribution designs → (cont_raw (n,17), cat (n,4))."""
    params = parameters.balanced_sample(n, seed)
    cont = np.array([[float(p[k]) for k in schema.INPUT_CONTINUOUS]
                     for p in params], dtype=np.float32)
    cat = np.array([[schema.ENCODING_MAPS[k][p[k]]
                     for k in schema.INPUT_CATEGORICAL] for p in params], dtype=np.int64)
    return cont, cat


def exact_class1(cont, cat, target_names, device):
    """Exact Barrowman values (n,T) and gradients (n,T,17) for target_names."""
    x = torch.tensor(cont, dtype=torch.float32, device=device, requires_grad=True)
    xc = torch.tensor(cat, dtype=torch.long, device=device)
    block = continuous_block(x, xc)
    cols = [CONTINUOUS_NAMES.index(CLASS1[t]) for t in target_names]
    vals = block[:, cols].detach().clone()
    grads = []
    for ci in cols:
        g = torch.autograd.grad(block[:, ci].sum(), x, retain_graph=True)[0]
        grads.append(g.detach())
    return vals, torch.stack(grads, dim=1)   # (n,T), (n,T,17)


# ── metrics ───────────────────────────────────────────────────────────────────
def _r2(pred, true):
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def evaluate(surr, test_raw, test_cat, test_tgt_nat, synth_eval, device):
    """Return dict: Class-1 grad cosine / value MAE (synth) + Class-2 R² (test)."""
    se_cont, se_cat, se_vals, se_grads, c1_names = synth_eval
    out = {}

    # Class-1 gradient + value quality vs exact Barrowman
    J = surr.jacobian(torch.tensor(se_cont, device=device),
                      torch.tensor(se_cat, dtype=torch.long, device=device))
    preds = surr.predict(torch.tensor(se_cont, device=device),
                         torch.tensor(se_cat, dtype=torch.long, device=device))
    for k, t in enumerate(c1_names):
        ti = surr.target_index(t)
        g_nn = J[:, ti, :]
        g_true = se_grads[:, k, :]
        cos = torch.nn.functional.cosine_similarity(g_nn, g_true, dim=1)
        out[f"{t}_grad_cos"] = float(cos.mean())
        out[f"{t}_val_mae"] = float((preds[:, ti] - se_vals[:, k]).abs().mean())

    # Class-2 retention: R² on the held-out test split (natural units)
    with torch.no_grad():
        pred_nat = surr.predict(torch.tensor(test_raw, device=device),
                                torch.tensor(test_cat, dtype=torch.long, device=device)).cpu().numpy()
    for t in CLASS2:
        ti = schema.TARGETS.index(t)
        out[f"{t}_r2"] = _r2(pred_nat[:, ti], test_tgt_nat[:, ti])
    out["class2_mean_r2"] = float(np.mean([out[f"{t}_r2"] for t in CLASS2]))
    return out


def _print_eval(tag, m, c1_names):
    print(f"[{tag}] Class-1 (synthetic, exact):")
    for t in c1_names:
        print(f"    {t:28s} grad_cos={m[f'{t}_grad_cos']:.4f}  val_mae={m[f'{t}_val_mae']:.4g}")
    print(f"[{tag}] Class-2 retention: mean R²={m['class2_mean_r2']:.5f}  "
          f"(min {min(m[f'{t}_r2'] for t in CLASS2):.5f})")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Sobolev formula distillation (Class-1)")
    p.add_argument("--bundle", type=str, default="../../models/neural")
    p.add_argument("--data", type=str, default="../../outputs/rocket_data_full.jsonl")
    p.add_argument("--save-dir", type=str, default="../../models/neural_distilled")
    p.add_argument("--synth", type=int, default=150_000, help="synthetic Class-1 pool size")
    p.add_argument("--synth-eval", type=int, default=4_000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-val", type=float, default=1.0)
    p.add_argument("--lambda-grad", type=float, default=1.0)
    p.add_argument("--lambda-replay", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    c1_names = list(CLASS1.keys())
    t0 = time.time()

    # ── load canonical model (the thing we fine-tune) ─────────────────────────
    surr = DifferentiableSurrogate(args.bundle, device=str(device))
    model = surr.model
    print(f"Loaded canonical bundle from {args.bundle} "
          f"({sum(p_.numel() for p_ in model.parameters()):,} params)")

    # Fixed-affine BatchNorm + no dropout: train the *inference* function so the
    # gradients we match are the gradients the optimizer will actually see.
    model.eval()

    # ── real corpus → scaled replay tensors + natural test set ────────────────
    cont, cat, tgt = load_corpus(args.data)
    tr_idx, _, te_idx = make_splits(len(cont), seed=42)   # same split as training
    print(f"Corpus: {len(cont)} computable  (replay train={len(tr_idx)}, test={len(te_idx)})")

    with torch.no_grad():
        cont31 = continuous_block(torch.tensor(cont, device=device),
                                  torch.tensor(cat, dtype=torch.long, device=device))
        scaled_cont = ((cont31 - surr.in_mean) / surr.in_std).cpu()
    tgt_log = tgt.copy()
    for i in surr.log1p_indices:
        tgt_log[:, i] = np.log1p(tgt_log[:, i])
    scaled_tgt = ((torch.tensor(tgt_log) - surr.tgt_mean.cpu()) / surr.tgt_std.cpu())

    replay_cont = scaled_cont[tr_idx].to(device)
    replay_cat = torch.tensor(cat[tr_idx], dtype=torch.long, device=device)
    replay_tgt = scaled_tgt[tr_idx].to(device)
    n_replay = replay_cont.shape[0]

    test_raw = cont[te_idx]
    test_cat = cat[te_idx]
    test_tgt_nat = tgt[te_idx]    # natural units (un-transformed)

    # ── synthetic Class-1 data (value + exact gradient) ───────────────────────
    print(f"Generating {args.synth:,} synthetic designs + exact Barrowman grads ...")
    s_cont, s_cat = gen_synth(args.synth, args.seed)
    s_vals, s_grads = exact_class1(s_cont, s_cat, c1_names, device)   # (N,3),(N,3,17)
    s_cont_t = torch.tensor(s_cont, device=device)
    s_cat_t = torch.tensor(s_cat, dtype=torch.long, device=device)

    # held-out synthetic eval batch (never trained on)
    se_cont, se_cat = gen_synth(args.synth_eval, args.seed + 9999)
    se_vals, se_grads = exact_class1(se_cont, se_cat, c1_names, device)
    synth_eval = (se_cont, se_cat, se_vals, se_grads, c1_names)

    # per-target normalisers so the 3 targets contribute comparably
    c1_out_idx = torch.tensor([surr.target_index(t) for t in c1_names], device=device)
    val_scale = surr.tgt_std[c1_out_idx]                              # (3,)
    grad_scale = torch.sqrt((s_grads ** 2).mean(dim=(0, 2)) + 1e-12)  # (3,)

    # ── baseline metrics ──────────────────────────────────────────────────────
    print()
    base = evaluate(surr, test_raw, test_cat, test_tgt_nat, synth_eval, device)
    _print_eval("before", base, c1_names)

    # ── fine-tune ─────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    huber = nn.HuberLoss(delta=1.0)
    n_synth = s_cont_t.shape[0]
    steps = n_synth // args.batch_size

    def batch_losses(sb):
        """Raw (value, gradient, replay) losses for one synthetic minibatch."""
        xb = s_cont_t[sb].detach().requires_grad_(True)
        cb = s_cat_t[sb]
        out = surr.forward(xb, cb)                       # natural units (B,12)
        pred_c1 = out[:, c1_out_idx]
        val_l = (((pred_c1 - s_vals[sb]) / val_scale) ** 2).mean()
        grad_l = 0.0
        for k in range(len(c1_names)):                   # Sobolev term (double backprop)
            ti = int(c1_out_idx[k])
            g = torch.autograd.grad(out[:, ti].sum(), xb,
                                    create_graph=True, retain_graph=True)[0]
            grad_l = grad_l + (((g - s_grads[sb, k, :]) / grad_scale[k]) ** 2).mean()
        grad_l = grad_l / len(c1_names)
        rb = torch.randint(0, n_replay, (sb.shape[0],), device=device)
        replay_l = huber(model(replay_cont[rb], replay_cat[rb]), replay_tgt[rb])
        return val_l, grad_l, replay_l

    # The three raw losses span ~100x in magnitude (grad≈100, val≈0.01,
    # replay≈0.002), so a plain weighted sum lets the gradient term swamp the
    # others (Class-2 then drifts). Normalise each by its initial scale so all
    # start at O(1) and the lambdas express true relative priority.
    vn = gn = rn = 0.0
    for _ in range(8):
        sb = torch.randint(0, n_synth, (args.batch_size,), device=device)
        vl, gl, rl = batch_losses(sb)
        vn += float(vl.detach()); gn += float(gl.detach()); rn += float(rl.detach())
    val_norm, grad_norm, replay_norm = max(vn / 8, 1e-8), max(gn / 8, 1e-8), max(rn / 8, 1e-8)
    print(f"\nLoss scales (initial): val={val_norm:.4e} grad={grad_norm:.4e} replay={replay_norm:.4e}")
    print(f"Fine-tuning: {args.epochs} epochs × {steps} steps "
          f"(synth batch {args.batch_size}) on {device}  "
          f"[lambda v/g/r = {args.lambda_val}/{args.lambda_grad}/{args.lambda_replay}]")

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_synth, device=device)
        ep_v = ep_g = ep_r = 0.0
        te = time.time()
        for b in range(steps):
            sb = perm[b * args.batch_size:(b + 1) * args.batch_size]
            val_loss, grad_loss, replay_loss = batch_losses(sb)
            loss = (args.lambda_val * val_loss / val_norm
                    + args.lambda_grad * grad_loss / grad_norm
                    + args.lambda_replay * replay_loss / replay_norm)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_v += float(val_loss.detach()); ep_g += float(grad_loss.detach())
            ep_r += float(replay_loss.detach())

        print(f"  epoch {epoch}/{args.epochs}  val={ep_v/steps:.4e}  "
              f"grad={ep_g/steps:.4e}  replay={ep_r/steps:.4e}  ({time.time()-te:.0f}s)")

    # ── final metrics ─────────────────────────────────────────────────────────
    print()
    final = evaluate(surr, test_raw, test_cat, test_tgt_nat, synth_eval, device)
    _print_eval("after", final, c1_names)

    # ── save distilled bundle (copy bundle then overwrite weights/metadata) ───
    src = Path(args.bundle)
    dst = Path(args.save_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("model_config.json", "feature_config.json",
               "input_scaler.joblib", "target_scaler.joblib"):
        shutil.copyfile(src / fn, dst / fn)
    torch.save(model.state_dict(), dst / "model.pt")

    with open(src / "model_metadata.json") as f:
        meta = json.load(f)
    meta["distillation"] = {
        "method": "sobolev_class1",
        "synth_pool": args.synth,
        "epochs": args.epochs,
        "lambda_val": args.lambda_val,
        "lambda_grad": args.lambda_grad,
        "lambda_replay": args.lambda_replay,
        "lr": args.lr,
        "class1_targets": c1_names,
        "before": {k: base[k] for k in base if "grad_cos" in k or k == "class2_mean_r2"},
        "after": {k: final[k] for k in final if "grad_cos" in k or k == "class2_mean_r2"},
        "wall_time_s": round(time.time() - t0, 1),
    }
    with open(dst / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDistilled bundle saved to {dst}/  (total {time.time()-t0:.0f}s)")
    for t in c1_names:
        print(f"  {t:28s} grad_cos {base[f'{t}_grad_cos']:.4f} -> {final[f'{t}_grad_cos']:.4f}")
    print(f"  Class-2 mean R² {base['class2_mean_r2']:.5f} -> {final['class2_mean_r2']:.5f}")


if __name__ == "__main__":
    main()
