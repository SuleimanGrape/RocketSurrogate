"""Summary distribution plots for the dataset."""

import json
import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INPUT_KEYS = [
    "diameter_mm", "length_m", "nose_length_m", "fin_count",
    "fin_root_chord_m", "fin_span_m", "dry_mass_kg", "motor_class",
    "wind_speed_mps", "elevation_m",
]

OUTPUT_KEYS = [
    "apogee_m", "max_velocity_mps", "max_mach", "max_acceleration_mps2",
    "burnout_altitude_m", "flight_time_s", "landing_velocity_mps",
    "stability_margin_calibers", "rail_exit_velocity_mps",
    "max_dynamic_pressure_pa",
]


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
