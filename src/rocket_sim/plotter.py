"""Summary distribution plots for the dataset."""

import json
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import schema

INPUT_KEYS = schema.PLOT_INPUT_KEYS
OUTPUT_KEYS = schema.PLOT_OUTPUT_KEYS


def _extract_field(records: List[dict], field: str, sub: str = "input") -> list:
    vals = []
    for r in records:
        try:
            vals.append(r[sub][field])
        except (KeyError, TypeError):
            pass
    return vals


def generate_plots(records: List[dict], output_dir: str) -> None:
    """Generate input/output distribution plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    for i, key in enumerate(INPUT_KEYS):
        ax = axes[i]
        vals = _extract_field(records, key, "input")
        if vals:
            if key in ("motor_class", "fin_count"):
                unique, counts = np.unique(vals, return_counts=True)
                ax.bar([str(u) for u in unique], counts)
            else:
                ax.hist(vals, bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    for j in range(len(INPUT_KEYS), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Input Parameter Distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "input_distributions.png"), dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    for i, key in enumerate(OUTPUT_KEYS):
        ax = axes[i]
        vals = _extract_field(records, key, "output")
        if vals:
            ax.hist(vals, bins=30, edgecolor="black", alpha=0.7)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    for j in range(len(OUTPUT_KEYS), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Output Metric Distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "output_distributions.png"), dpi=100)
    plt.close(fig)

    print(f"  Plots saved to {output_dir}/")
