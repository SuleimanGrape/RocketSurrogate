"""LLM fine-tuning entry point: SFT on NN-generated rocket text pairs.

Usage:
    python train_llm.py \\
        --data llm_sft_data.jsonl \\
        --base-model meta-llama/Llama-3.1-8B \\
        --output-dir checkpoints/llm_sft \\
        --epochs 3

ROCm cloud usage:
    - Install PyTorch ROCm: pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
    - bitsandbytes 0.43+ has native ROCm support
    - device_map='auto' automatically places model layers on AMD GPU(s)
    - For multi-GPU: accelerate config → MULTI_GPU, backend=nccl
"""

from __future__ import annotations

import argparse
from training.llm_finetuner import LLMFinetuner
from utils.helpers import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune an LLM on rocket text pairs")
    p.add_argument("--data", type=str, required=True, help="Path to JSONL with instruction/response pairs")
    p.add_argument("--base-model", type=str, default="meta-llama/Llama-3.1-8B",
                   help="HF model ID or local path")
    p.add_argument("--output-dir", type=str, default="checkpoints/llm_sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    p.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention")
    p.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    finetuner = LLMFinetuner(
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
        use_flash_attn=not args.no_flash_attn,
        gradient_checkpointing=not args.no_grad_checkpoint,
    )

    finetuner.train(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
