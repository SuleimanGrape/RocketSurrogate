"""LLM fine-tuning via supervised fine-tuning (SFT) on NN-generated text pairs.

Uses Hugging Face transformers + trl (SFTTrainer) + peft (LoRA).

ROCm compatibility:
    - PyTorch ROCm exposes AMD GPUs through the standard `torch.cuda` API.
    - `accelerate` handles multi-GPU on ROCm with `backend: nccl`.
    - `bitsandbytes` 0.43+ has native ROCm support (no fork needed for recent versions).
    - `device_map='auto'` in Hugging Face automatically places model layers on
      the available GPU (NVIDIA CUDA or AMD ROCm).
    - Mixed precision: `torch.amp.autocast('cuda')` works on both.

Usage:
    finetuner = LLMFinetuner(
        base_model="meta-llama/Llama-3.1-8B",
        lora_r=64,
        lora_alpha=128,
    )
    finetuner.train(
        data_path="llm_sft_data.jsonl",
        output_dir="checkpoints/llm_sft",
        epochs=3,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch


def _rocm_check():
    """Warn if running on ROCm with incompatible settings."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        if "amd" in device_name or "radeon" in device_name:
            print("[ROCm] AMD GPU detected. Using ROCm-compatible settings.")
            return True
        else:
            print(f"[CUDA] NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            return False
    return False


class LLMFinetuner:
    """Supervised fine-tuning of an LLM on instruction-response text pairs.

    Parameters
    ----------
    base_model : str
        Hugging Face model ID or local path (e.g. "meta-llama/Llama-3.1-8B").
    lora_r : int
        LoRA rank (higher = more capacity, more memory). Default 64.
    lora_alpha : int
        LoRA alpha (scaling factor). Default 128.
    lora_dropout : float
        Dropout for LoRA layers. Default 0.05.
    max_seq_length : int
        Max token sequence length. Default 2048.
    load_in_4bit : bool
        Use 4-bit quantization (QLoRA). Requires bitsandbytes. Default True.
    bnb_4bit_compute_dtype : str
        Compute dtype for 4-bit layers. "float16" or "bfloat16". Default "bfloat16".
    use_flash_attn : bool
        Use Flash Attention 2 if available. Default True.
    gradient_checkpointing : bool
        Trade compute for memory. Recommended for large models. Default True.
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B",
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        use_flash_attn: bool = True,
        gradient_checkpointing: bool = True,
    ):
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.use_flash_attn = use_flash_attn
        self.gradient_checkpointing = gradient_checkpointing

        self.model = None
        self.tokenizer = None
        self._is_rocm = False

    def _load_model_and_tokenizer(self):
        """Lazy-load model + tokenizer with ROCm-compatible settings."""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self._is_rocm = _rocm_check()

        # Quantization config
        bnb_config = None
        if self.load_in_4bit:
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            print(f"  QLoRA: 4-bit quantization ({self.bnb_4bit_compute_dtype})")

        # Attention implementation
        attn_impl = "flash_attention_2" if self.use_flash_attn else "eager"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",          # Automatically distributes across GPUs
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
            )
        except Exception:
            # Fallback if flash attention not available on ROCm
            print("  Flash Attention not available, falling back to eager attention")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Enable gradient checkpointing to save VRAM
        if self.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("  Gradient checkpointing: ENABLED")

        print(f"  Model loaded: {self.base_model}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _apply_lora(self):
        """Wrap the model with LoRA adapters."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _load_data(self, data_path: str):
        """Load JSONL instruction-response pairs."""
        from datasets import load_dataset

        data = load_dataset("json", data_files=data_path, split="train")
        print(f"  Loaded {len(data)} training examples")
        return data

    def train(
        self,
        data_path: str,
        output_dir: str = "checkpoints/llm_sft",
        epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        save_steps: int = 100,
        lr_scheduler_type: str = "cosine",
        optim: str = "paged_adamw_8bit",
        seed: int = 42,
    ):
        """Run SFT training.

        Parameters
        ----------
        data_path : str
            Path to JSONL with {"instruction": ..., "response": ...} entries.
        output_dir : str
            Directory to save checkpoints and final model.
        epochs : int
            Number of training epochs.
        batch_size : int
            Per-device batch size.
        gradient_accumulation_steps : int
            Effective batch = batch_size * grad_accum * num_gpus.
        learning_rate : float
            Peak learning rate. Default 2e-4 (standard for LoRA).
        warmup_ratio : float
            Fraction of training for LR warmup.
        """
        from trl import SFTConfig, SFTTrainer

        print(f"\n{'='*60}")
        print(f"  LLM Fine-tuning (SFT)")
        print(f"{'='*60}")

        self._load_model_and_tokenizer()
        self._apply_lora()
        dataset = self._load_data(data_path)

        # Format function: combine instruction + response for causal LM
        def format_func(example):
            return {
                "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
            }

        dataset = dataset.map(format_func)

        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            optim=optim,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            fp16=False,                     # Use bf16 (better for ROCm)
            bf16=True,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
            packing=True,                   # Pack multiple short examples per sequence
            seed=seed,
            report_to="none",              # Disable wandb/tensorboard by default
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        print(f"\n  Training config:")
        print(f"    Epochs: {epochs}")
        print(f"    Batch size (per device): {batch_size}")
        print(f"    Gradient accumulation: {gradient_accumulation_steps}")
        print(f"    Effective batch size: {batch_size * gradient_accumulation_steps * max(1, torch.cuda.device_count())}")
        print(f"    Learning rate: {learning_rate}")
        print(f"    GPUs: {torch.cuda.device_count()}")

        trainer.train()

        # Save LoRA adapter
        final_path = Path(output_dir) / "final_lora"
        trainer.save_model(str(final_path))
        print(f"\nLoRA adapter saved to {final_path}/")

        return trainer
