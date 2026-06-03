"""Surrogate neural network model definitions.

Supports multiple architectures that can be swapped in/out as we learn
what works best for the rocket simulation data.

Feature lists match the JSONL keys produced by the Rocket generator's
extract_input() and extract_output() in Rocket/outputs.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Feature definitions — MUST match extract_input / extract_output keys
# ---------------------------------------------------------------------------

# Continuous scalar inputs from the JSONL "input" object
CONTINUOUS_FEATURES: List[str] = [
    "length_m",
    "nose_length_m",
    "fin_root_chord_m",
    "fin_tip_chord_m",
    "fin_span_m",
    "fin_sweep_m",
    "fin_thickness_mm",
    "dry_mass_kg",
    "propellant_mass_kg",
    "burn_time_s",
    "avg_thrust_N",
    "wind_speed_mps",
    "wind_direction_deg",
    "elevation_m",
    "temperature_c",
    "rail_length_m",
    "launch_angle_deg",
]

# Categorical inputs (string labels → integer codes via ENCODING_MAPS)
CATEGORICAL_FEATURES: List[str] = [
    "diameter_mm",   # 6  classes: 24, 29, 38, 54, 75, 98
    "nose_type",     # 4  classes: conical, ogive, von_karman, elliptical
    "fin_count",     # 2  classes: 3, 4
    "motor_class",   # 10 classes: D, E, F, G, H, I, J, K, L, M
]

# String → integer encoding for each categorical feature
ENCODING_MAPS = {
    "diameter_mm": {24: 0, 29: 1, 38: 2, 54: 3, 75: 4, 98: 5},
    "nose_type":  {"conical": 0, "ogive": 1, "von_karman": 2, "elliptical": 3},
    "fin_count":  {3: 0, 4: 1},
    "motor_class": {
        "D": 0, "E": 1, "F": 2, "G": 3, "H": 4,
        "I": 5, "J": 6, "K": 7, "L": 8, "M": 9,
    },
}

# Number of classes per categorical feature
CATEGORICAL_CARDINALITIES = {k: len(v) for k, v in ENCODING_MAPS.items()}

# Regression targets from the JSONL "output" object
TARGETS: List[str] = [
    "apogee_m",
    "max_velocity_mps",
    "max_mach",
    "max_acceleration_mps2",
    "burnout_altitude_m",
    "burnout_velocity_mps",
    "flight_time_s",
    "landing_velocity_mps",
    "stability_margin_calibers",
    "rail_exit_velocity_mps",
    "max_dynamic_pressure_pa",
    "cg_m",
    "cp_m",
]


# ---------------------------------------------------------------------------
# Categorical embedding module
# ---------------------------------------------------------------------------

class CategoricalEmbeddings(nn.Module):
    """Learnable embeddings for each categorical feature, concatenated."""

    def __init__(self, cardinalities: dict[str, int], embedding_dim: int = 8):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_classes, embedding_dim)
            for name, num_classes in cardinalities.items()
        })
        self.embedding_dim = embedding_dim

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        embedded = []
        for i, (name, emb) in enumerate(self.embeddings.items()):
            embedded.append(emb(x_cat[:, i]))
        return torch.cat(embedded, dim=-1)

    @property
    def output_dim(self) -> int:
        return len(self.embeddings) * self.embedding_dim


# ---------------------------------------------------------------------------
# Architecture 1: MLP baseline
# ---------------------------------------------------------------------------

class MLPSurrogate(nn.Module):
    """Feed-forward network with categorical embeddings."""

    def __init__(
        self,
        continuous_dim: int = len(CONTINUOUS_FEATURES),
        categorical_cardinalities: dict[str, int] = CATEGORICAL_CARDINALITIES,
        embedding_dim: int = 8,
        hidden_dims: List[int] = None,
        output_dim: int = len(TARGETS),
        dropout: float = 0.1,
        activation: str = "silu",
        use_batch_norm: bool = True,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 512, 512, 256, 128]
        super().__init__()

        self.cat_emb = CategoricalEmbeddings(categorical_cardinalities, embedding_dim)
        input_dim = continuous_dim + self.cat_emb.output_dim

        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        act = act_map[activation]

        layers: list = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cat_embed = self.cat_emb(x_cat)
        x = torch.cat([x_cont, cat_embed], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Architecture 2: Residual MLP
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float, activation: str, use_batch_norm: bool):
        super().__init__()
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}
        layers = []
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(act_map[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim, dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(dim))
        self.block = nn.Sequential(*layers)
        self.act = act_map[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualMLPSurrogate(nn.Module):
    """MLP with residual skip connections — trains better at depth."""

    def __init__(
        self,
        continuous_dim: int = len(CONTINUOUS_FEATURES),
        categorical_cardinalities: dict[str, int] = CATEGORICAL_CARDINALITIES,
        embedding_dim: int = 8,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        output_dim: int = len(TARGETS),
        dropout: float = 0.1,
        activation: str = "silu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.cat_emb = CategoricalEmbeddings(categorical_cardinalities, embedding_dim)
        input_dim = continuous_dim + self.cat_emb.output_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout, activation, use_batch_norm)
            for _ in range(num_blocks)
        ])
        self.output_head = nn.Linear(hidden_dim, output_dim)
        self.apply(MLPSurrogate._init_weights)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_cont, self.cat_emb(x_cat)], dim=-1)
        return self.output_head(self.blocks(self.input_proj(x)))


# ---------------------------------------------------------------------------
# Architecture 3: Feature-token Transformer
# ---------------------------------------------------------------------------

class FeatureTransformerSurrogate(nn.Module):
    """Treats each input feature as a token, processes with a Transformer encoder."""

    def __init__(
        self,
        continuous_dim: int = len(CONTINUOUS_FEATURES),
        categorical_cardinalities: dict[str, int] = CATEGORICAL_CARDINALITIES,
        embedding_dim: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        output_dim: int = len(TARGETS),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_emb = CategoricalEmbeddings(categorical_cardinalities, embedding_dim)
        self.cont_proj = nn.Linear(1, d_model)
        self.cat_proj = nn.Linear(embedding_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cont_tokens = self.cont_proj(x_cont.unsqueeze(-1))
        cat_embeds = torch.stack([
            emb(x_cat[:, i]) for i, (_, emb) in enumerate(self.cat_emb.embeddings.items())
        ], dim=1)
        cat_tokens = self.cat_proj(cat_embeds)
        tokens = torch.cat([cont_tokens, cat_tokens], dim=1)
        return self.output_head(self.transformer(tokens).mean(dim=1))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str = "mlp", **kwargs) -> nn.Module:
    registry = {
        "mlp": MLPSurrogate,
        "resmlp": ResidualMLPSurrogate,
        "transformer": FeatureTransformerSurrogate,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry)}")
    return registry[name](**kwargs)
