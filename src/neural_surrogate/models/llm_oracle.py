"""LLM-assisted analysis: generate synthetic text pairs for LLM fine-tuning.

Uses a trained NN surrogate to predict flight metrics for random rocket designs,
then formats each (design, prediction) as an instruction-response text pair for
supervised fine-tuning (SFT). Inference runs via the standard PyTorch pipeline
and is ROCm-compatible (device='cuda' targets AMD GPUs).
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from models.surrogate import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    TARGETS,
    ENCODING_MAPS,
)
from models.scalers import StandardScaler

# Discrete design choices — single source of truth in rocket_sim/config.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rocket_sim"))
import config as cfg  # noqa: E402

BODY_DIAMETERS_MM = cfg.BODY_DIAMETERS_MM
NOSE_TYPES = cfg.NOSE_TYPES
FIN_COUNTS = cfg.FIN_COUNTS
MOTOR_CLASSES = cfg.MOTOR_CLASSES

# NOTE: MOTOR_SPECS / SAMPLING_RANGES below are intentionally looser than
# config.MOTOR_SPECS — they widen the design space for LLM-oracle exploration,
# so they are kept local on purpose (not a drift bug).
# Sampling bounds for continuous parameters
SAMPLING_RANGES = {
    "length_m":               (0.5, 6.0),
    "nose_length_m":          None,     # derived from diameter
    "fin_root_chord_m":       None,     # derived from diameter
    "fin_tip_chord_m":        None,     # derived from root chord
    "fin_span_m":             None,     # derived from diameter
    "fin_sweep_m":            None,     # derived from root chord
    "fin_thickness_mm":       (2.0, 12.0),
    "dry_mass_kg":            (0.3, 120.0),
    "propellant_mass_kg":     None,     # derived from motor class
    "burn_time_s":            None,     # derived from motor class
    "avg_thrust_N":           None,     # derived from motor class
    "wind_speed_mps":         (0.0, 15.0),
    "wind_direction_deg":     (0.0, 360.0),
    "elevation_m":            (0.0, 3000.0),
    "temperature_c":          (-10.0, 40.0),
    "rail_length_m":          (1.0, 8.0),
    "launch_angle_deg":       (85.0, 90.0),
}

MOTOR_SPECS = {
    "D":  (0.015, 0.030, 0.8,  1.5,  10,   30),
    "E":  (0.030, 0.060, 1.0,  2.0,  30,   60),
    "F":  (0.060, 0.120, 1.2,  2.5,  60,   120),
    "G":  (0.120, 0.250, 1.5,  3.0,  120,  250),
    "H":  (0.250, 0.500, 1.8,  3.5,  250,  500),
    "I":  (0.500, 1.000, 2.0,  4.0,  500,  1000),
    "J":  (1.000, 2.000, 2.5,  5.0,  1000, 2000),
    "K":  (2.000, 4.000, 3.0,  6.0,  2000, 4000),
    "L":  (4.000, 8.000, 3.5,  7.0,  4000, 8000),
    "M":  (8.000, 16.00, 4.0,  8.0,  8000, 16000),
}


def sample_random_designs(n: int, seed: int = 42) -> List[Dict]:
    """Sample n random rocket parameter dicts (same schema as the generator's extract_input)."""
    rng = np.random.default_rng(seed)
    designs = []

    for _ in range(n):
        diam = int(rng.choice(BODY_DIAMETERS_MM))
        nose = str(rng.choice(NOSE_TYPES))
        n_fins = int(rng.choice(FIN_COUNTS))
        motor = str(rng.choice(MOTOR_CLASSES))
        diam_m = diam / 1000.0

        lo, hi = SAMPLING_RANGES["length_m"]
        length = float(rng.uniform(lo, hi))

        nose_len_diam = float(rng.uniform(0.5, 5.0))
        nose_len = nose_len_diam * diam_m

        root_diam = float(rng.uniform(1.0, 8.0))
        root_chord = root_diam * diam_m
        tip_chord = float(rng.uniform(0.2 * root_chord, root_chord))

        span_diam = float(rng.uniform(0.5, 3.0))
        span = span_diam * diam_m

        sweep = float(rng.uniform(0.0, 1.0 * root_chord))

        lo, hi = SAMPLING_RANGES["fin_thickness_mm"]
        thick = float(rng.uniform(lo, hi))

        lo, hi = SAMPLING_RANGES["dry_mass_kg"]
        dry_mass = float(rng.uniform(lo, hi))

        ms = MOTOR_SPECS[motor]
        prop_mass = float(rng.uniform(ms[0], ms[1]))
        burn = float(rng.uniform(ms[2], ms[3]))
        thrust = float(rng.uniform(ms[4], ms[5]))

        lo, hi = SAMPLING_RANGES["wind_speed_mps"]
        wind_spd = float(rng.uniform(lo, hi))
        lo, hi = SAMPLING_RANGES["wind_direction_deg"]
        wind_dir = float(rng.uniform(lo, hi))
        lo, hi = SAMPLING_RANGES["elevation_m"]
        elev = float(rng.uniform(lo, hi))
        lo, hi = SAMPLING_RANGES["temperature_c"]
        temp = float(rng.uniform(lo, hi))
        lo, hi = SAMPLING_RANGES["rail_length_m"]
        rail = float(rng.uniform(lo, hi))
        lo, hi = SAMPLING_RANGES["launch_angle_deg"]
        angle = float(rng.uniform(lo, hi))

        designs.append({
            "diameter_mm": diam,
            "length_m": round(length, 3),
            "nose_type": nose,
            "nose_length_m": round(nose_len, 3),
            "fin_count": n_fins,
            "fin_root_chord_m": round(root_chord, 4),
            "fin_tip_chord_m": round(tip_chord, 4),
            "fin_span_m": round(span, 4),
            "fin_sweep_m": round(sweep, 4),
            "fin_thickness_mm": round(thick, 1),
            "dry_mass_kg": round(dry_mass, 3),
            "motor_class": motor,
            "propellant_mass_kg": round(prop_mass, 4),
            "burn_time_s": round(burn, 2),
            "avg_thrust_N": round(thrust, 1),
            "wind_speed_mps": round(wind_spd, 1),
            "wind_direction_deg": round(wind_dir, 1),
            "elevation_m": round(elev, 0),
            "temperature_c": round(temp, 1),
            "rail_length_m": round(rail, 1),
            "launch_angle_deg": round(angle, 1),
        })

    return designs


def _design_to_arrays(
    designs: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of design dicts to (continuous, categorical) numpy arrays."""
    n = len(designs)
    cont = np.zeros((n, len(CONTINUOUS_FEATURES)), dtype=np.float32)
    cat = np.zeros((n, len(CATEGORICAL_FEATURES)), dtype=np.int64)

    for i, d in enumerate(designs):
        for j, k in enumerate(CONTINUOUS_FEATURES):
            cont[i, j] = float(d[k])
        for j, k in enumerate(CATEGORICAL_FEATURES):
            cat[i, j] = int(ENCODING_MAPS[k][d[k]])

    return cont, cat


def _arrays_to_predictions(
    pred_scaled: np.ndarray,
    scaler: StandardScaler,
) -> List[Dict]:
    """Convert (N, D_tgt) scaled predictions back to dicts with real units."""
    pred_real = scaler.inverse_transform(pred_scaled)
    results = []
    for row in pred_real:
        d = {}
        for j, k in enumerate(TARGETS):
            d[k] = round(float(row[j]), 4)
        results.append(d)
    return results


# Text templates
INPUT_TEMPLATE = (
    "Rocket design parameters:\n"
    "- Body diameter: {diameter_mm} mm, length: {length_m} m\n"
    "- Nose: {nose_type}, length: {nose_length_m} m\n"
    "- Fins: {fin_count}, root chord: {fin_root_chord_m} m, tip chord: {fin_tip_chord_m} m, "
    "span: {fin_span_m} m, sweep: {fin_sweep_m} m, thickness: {fin_thickness_mm} mm\n"
    "- Dry mass: {dry_mass_kg} kg\n"
    "- Motor: {motor_class}, propellant: {propellant_mass_kg} kg, burn time: {burn_time_s} s, "
    "avg thrust: {avg_thrust_N} N\n"
    "- Environment: elevation {elevation_m} m, temperature {temperature_c} C, "
    "wind {wind_speed_mps} m/s at {wind_direction_deg} deg\n"
    "- Launch: rail {rail_length_m} m, angle {launch_angle_deg} deg from vertical\n"
    "\nPredict the flight performance metrics for this rocket design."
)

TARGET_TEMPLATE = (
    "Predicted flight performance:\n"
    "- Apogee: {apogee_m} m\n"
    "- Max velocity: {max_velocity_mps} m/s (Mach {max_mach})\n"
    "- Max acceleration: {max_acceleration_mps2} m/s^2\n"
    "- Burnout altitude: {burnout_altitude_m} m\n"
    "- Burnout velocity: {burnout_velocity_mps} m/s\n"
    "- Flight time: {flight_time_s} s\n"
    "- Landing velocity: {landing_velocity_mps} m/s\n"
    "- Stability margin: {stability_margin_calibers} calibers\n"
    "- Rail exit velocity: {rail_exit_velocity_mps} m/s\n"
    "- Max dynamic pressure: {max_dynamic_pressure_pa} Pa\n"
    "- CG position: {cg_m} m from nose tip\n"
    "- CP position: {cp_m} m from nose tip\n"
    "- Motor class: {motor_class}"
)


def format_text_pair(design: Dict, prediction: Dict) -> Dict:
    """Format a design + prediction as an SFT text pair."""
    instruction = INPUT_TEMPLATE.format(**design)
    response = TARGET_TEMPLATE.format(**prediction, motor_class=design["motor_class"])
    return {
        "instruction": instruction,
        "response": response,
    }


class NNOracle:
    """Wraps a trained NN surrogate + scalers to generate text-pair data at scale.

    Usage:
        oracle = NNOracle.from_checkpoint("checkpoints/distilled/best.pt")
        text_pairs = oracle.generate_text_pairs(n_samples=10000, seed=123)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_scaler: StandardScaler,
        target_scaler: StandardScaler,
        device: str = "auto",
    ):
        import torch  # local import to avoid hard dependency if only sampling
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        scaler_in_path: Optional[str] = None,
        scaler_tgt_path: Optional[str] = None,
        model_type: str = "mlp",
        device: str = "auto",
        **model_kwargs,
    ) -> "NNOracle":
        """Load a trained NN + scalers from disk.

        Looks for scaler files alongside the checkpoint if paths not provided:
            ckpt_path = "checkpoints/distilled/best.pt"
            → scaler_in  = "checkpoints/distilled/input_scaler.joblib"
            → scaler_tgt = "checkpoints/distilled/target_scaler.joblib"
        """
        from models.surrogate import build_model, CONTINUOUS_FEATURES, CATEGORICAL_CARDINALITIES, TARGETS

        # Find scaler paths
        ckpt_dir = str(Path(ckpt_path).parent)
        if scaler_in_path is None:
            scaler_in_path = str(Path(ckpt_dir) / "input_scaler.joblib")
        if scaler_tgt_path is None:
            scaler_tgt_path = str(Path(ckpt_dir) / "target_scaler.joblib")

        # Build model architecture
        kwargs = {
            "continuous_dim": len(CONTINUOUS_FEATURES),
            "categorical_cardinalities": CATEGORICAL_CARDINALITIES,
            "output_dim": len(TARGETS),
            **model_kwargs,
        }
        model = build_model(model_type, **kwargs)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

        scaler_in = StandardScaler().load(scaler_in_path)
        scaler_tgt = StandardScaler().load(scaler_tgt_path)

        return cls(model, scaler_in, scaler_tgt, device=device)

    @torch.no_grad()
    def predict_batch(self, designs: List[Dict]) -> List[Dict]:
        """Run NN inference on a batch of design dicts, return prediction dicts."""
        cont, cat = _design_to_arrays(designs)
        cont_scaled = self.input_scaler.transform(cont)

        cont_t = torch.from_numpy(cont_scaled).to(self.device)
        cat_t = torch.from_numpy(cat).to(self.device)

        # Use AMP for faster inference on GPU (works on ROCm)
        with torch.amp.autocast('cuda', enabled=(self.device.type == "cuda")):
            pred = self.model(cont_t, cat_t)

        pred_np = pred.float().cpu().numpy()
        return _arrays_to_predictions(pred_np, self.target_scaler)

    def generate_text_pairs(
        self,
        n_samples: int,
        seed: int = 42,
        batch_size: int = 1024,
    ) -> List[Dict]:
        """Generate n_samples text pairs: random rocket designs + NN predictions."""
        designs = sample_random_designs(n_samples, seed=seed)
        all_text_pairs = []

        for start in range(0, len(designs), batch_size):
            batch = designs[start:start + batch_size]
            predictions = self.predict_batch(batch)
            for d, p in zip(batch, predictions):
                all_text_pairs.append(format_text_pair(d, p))

        return all_text_pairs
