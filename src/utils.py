"""
utils.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 1, used throughout

Responsibility:
    Seed control, checkpoint I/O, and plotting helpers.
"""

import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42) -> None:
    """
    Set random seed across random, numpy, and torch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    epoch: int = None,
) -> None:
    """
    Save model weights.

    Args:
        model: PyTorch model
        path: checkpoint destination
        epoch: optional epoch number
    """

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
) -> torch.nn.Module:
    """
    Load model weights from checkpoint.
    """

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    # Support our checkpoint format
    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    # Also support a raw state_dict checkpoint
    else:

        model.load_state_dict(
            checkpoint
        )

    return model


def plot_snr_curve(
    results: dict,
    title: str,
    save_path: str,
) -> None:
    """
    Plot accuracy/success rate against SNR.
    """

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        results.keys(),
        results.values(),
        marker="o",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy / Success Rate")
    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()


def plot_confusion_matrix(
    cm,
    class_names: list,
    save_path: str,
) -> None:
    """
    Render and save a confusion matrix.
    """

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(12, 10)
    )

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
    )

    plt.close()