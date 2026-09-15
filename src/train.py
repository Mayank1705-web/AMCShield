"""
train.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 1-2 (Week 1-2), used throughout

Responsibility: the ONE shared generic training loop, used by both
baseline (plain supervised) and robust (PGD-adversarial) training runs.

CONSTRAINT: must not be duplicated per model type. Supports adversarial
training via an adversarial_fn hook (None = plain supervised training).
See architecture.md Section 4, integration point #1 -- this interface
must be agreed on by Person A and Person B before both build against it
(Week 1 sync point, see phases.md).
"""

import torch
import yaml
from typing import Optional, Callable
from src import utils


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: dict,
    adversarial_fn: Optional[Callable] = None,
    checkpoint_path: str = "checkpoints/model.pt",
) -> torch.nn.Module:
    """
    Generic training loop.

    Args:
        model: the model to train (BaselineCNN or RobustCNN)
        train_loader, val_loader: DataLoaders
        config: loaded from configs/config.yaml
        adversarial_fn: if provided, called on each batch to generate
            adversarial examples (e.g. PGD) that replace a fraction
            (config['adv_train_ratio']) of the batch. If None, plain
            supervised training.
        checkpoint_path: where to save model weights

    IMPORTANT: checkpoint every few epochs, not only at the end --
    Colab free-tier GPU disconnects mid-run are expected, not exceptional
    (see ROLLBACK.md and CONSTRAINTS.md).

    TODO (Person C): implement per design.md Section 3-4 and
    architecture.md Section 2.
    """
    utils.set_seed(config.get("seed", 42))
    raise NotImplementedError


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # TODO: argparse for --model baseline|robust, wire up data loaders,
    # call train_model with or without adversarial_fn accordingly.
    pass
