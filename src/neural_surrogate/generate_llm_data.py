"""Generate synthetic text-pair data for LLM fine-tuning using the NN oracle.

Outputs a JSONL file where each line is:
    {"instruction": "...", "response": "..."}

Usage:
    python generate_llm_data.py \\
        --nn-checkpoint checkpoints/distilled/best.pt \\
        --count 50000 \\
        --output llm_sft_data.jsonl \\
        --seed 42 \\
        --batch-size 4096

ROCm:
    On cloud AMD GPU: --device cuda (or auto-detect) for fast batched inference.
    The NNOracle.predict_batch uses torch.amp.autocast for GPU acceleration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.llm_oracle import NNOracle
from utils.helpers import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Generate LLM training data via NN oracle")
    p.add_argument("--nn-checkpoint", type=str, required=True, help="NN checkpoint (.pt)")
    p.add_argument("--scaler-in", type=str, default=None, help="Input scaler (.joblib), auto-detected if omitted")
    p.add_argument("--scaler-tgt", type=str, default=None, help="Target scaler (.joblib), auto-detected if omitted")
    p.add_argument("--model-type", type=str, default="mlp", choices=["mlp", "resmlp", "transformer"])
    p.add_argument("--count", type=int, default=10000, help="Number of text pairs to generate")
    p.add_argument("--output", type=str, default="llm_sft_data.jsonl", help="Output JSONL path")
    p.add_argument("--batch-size", type=int, default=1024, help="Inference batch size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="Device for NN inference (auto/cuda/cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Loading NN oracle checkpoint: {args.nn_checkpoint}")
    oracle = NNOracle.from_checkpoint(
        ckpt_path=args.nn_checkpoint,
        scaler_in_path=args.scaler_in,
        scaler_tgt_path=args.scaler_tgt,
        model_type=args.model_type,
        device=args.device,
    )
    print(f"  Device: {oracle.device}")
    print(f"  Generating {args.count} text pairs ...")

    text_pairs = oracle.generate_text_pairs(
        n_samples=args.count,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    # Save as JSONL
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in text_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"  Saved {len(text_pairs)} text pairs to {out_path}")

    # Print a sample
    print(f"\n--- Sample pair ---")
    print(f"INSTRUCTION:\n{text_pairs[0]['instruction'][:300]}...")
    print(f"\nRESPONSE:\n{text_pairs[0]['response'][:300]}...")
    print(f"--- End sample ---\n")


if __name__ == "__main__":
    main()
