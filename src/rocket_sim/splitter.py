"""Train / validation / test dataset splitting."""

import json
import os
from typing import List


def split_dataset(
    records: List[dict],
    output_dir: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """Shuffle and split records into train/val/test JSONL files."""
    import random
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    os.makedirs(output_dir, exist_ok=True)
    for name, data in splits.items():
        with open(os.path.join(output_dir, f"{name}.jsonl"), "w") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")

    counts = {k: len(v) for k, v in splits.items()}
    print(f"  Splits: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    return counts
