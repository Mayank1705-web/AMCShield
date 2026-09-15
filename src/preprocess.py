"""
preprocess.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 1 (Week 1-2)

Responsibility: normalize signals and produce a stratified train/val/test
split, saved to data/processed/.

Stratification MUST be by (modulation, SNR) jointly, not by class alone --
see design.md Section 2. An unstratified split will silently corrupt the
accuracy-vs-SNR curve and the generalization-gap heatmap downstream.
"""

import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Per-sample amplitude (unit-energy) normalization.
    TODO (Person A): implement per design.md Section 2.
    """
    raise NotImplementedError


def stratified_split(
    X: np.ndarray, y: np.ndarray, snr: np.ndarray,
    train_size: float = 0.70, val_size: float = 0.15, test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Stratify by (modulation, SNR) jointly.

    TODO (Person A): build a combined stratification key from y and snr,
    then split. Return a dict with 'train', 'val', 'test' keys, each
    containing {'X':..., 'y':..., 'snr':...}.
    """
    raise NotImplementedError


def save_processed(splits: dict, out_dir: str = "data/processed") -> None:
    """Save split arrays to disk (.npy). TODO (Person A)."""
    raise NotImplementedError


if __name__ == "__main__":
    # TODO: wire together load_radioml -> normalize -> stratified_split -> save_processed
    pass
