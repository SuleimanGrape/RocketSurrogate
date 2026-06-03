"""Data augmentation via Gaussian perturbation of continuous features.

Used to expand the training set for knowledge distillation: XGBoost predicts
soft labels on the augmented data, and the neural network learns from both
the hard labels (original data) and soft labels (augmented data).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from models.surrogate import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, ENCODING_MAPS


def augment_data(
    continuous: np.ndarray,
    categorical: np.ndarray,
    targets: np.ndarray,
    n_augmented: int = 5,
    noise_std: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create augmented copies of the dataset by adding Gaussian noise to continuous features.

    Parameters
    ----------
    continuous : (N, D_cont) float array — original continuous features (scaled).
    categorical : (N, D_cat) int array — original categorical features (unchanged).
    targets : (N, D_tgt) float array — original targets.
    n_augmented : int — number of augmented copies per original sample.
    noise_std : float — std of Gaussian noise as a fraction of each feature's
                 training-set standard deviation. Default 2%.
    seed : int — random seed.

    Returns
    -------
    (continuous_aug, categorical_aug, targets_aug) — concatenated original + augmented.
        Shape: (N * (1 + n_augmented), ...)
    """
    rng = np.random.default_rng(seed)
    n = len(continuous)

    # Compute per-feature std from the data for proportional noise
    feature_std = continuous.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-8] = 1.0   # avoid zero noise for constant feats

    aug_cont_list = [continuous]
    aug_cat_list = [categorical]
    aug_tgt_list = [targets]

    for _ in range(n_augmented):
        noise = rng.normal(0, noise_std * feature_std, size=continuous.shape).astype(np.float32)
        aug_cont_list.append(continuous + noise)
        aug_cat_list.append(categorical.copy())
        aug_tgt_list.append(targets.copy())

    return (
        np.concatenate(aug_cont_list, axis=0),
        np.concatenate(aug_cat_list, axis=0),
        np.concatenate(aug_tgt_list, axis=0),
    )


def chunked_generator(
    x: np.ndarray,
    cat: np.ndarray,
    chunk_size: int = 10000,
):
    """Yield (x_chunk, cat_chunk) in chunks to avoid memory blowup during XGBoost inference."""
    n = len(x)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield x[start:end], cat[start:end]
