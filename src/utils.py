"""
utils.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 1, used throughout

Responsibility: seed control, plotting helpers, checkpoint I/O.
Keep this dependency-light -- it's imported by nearly every other file.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across random, numpy, and torch for reproducibility.
    Call this at the start of any training or attack script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, path: str, epoch: int = None) -> None:
    """TODO (Person C): save model state_dict, optionally with epoch metadata."""
    raise NotImplementedError


def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """TODO (Person C): load model state_dict from path."""
    raise NotImplementedError


def plot_snr_curve(results: dict, title: str, save_path: str) -> None:
    """TODO (Person C): plot accuracy/success-rate vs SNR, matplotlib."""
    raise NotImplementedError


def plot_confusion_matrix(cm, class_names: list, save_path: str) -> None:
    """TODO (Person C): render confusion matrix as a heatmap, seaborn."""
    raise NotImplementedError
