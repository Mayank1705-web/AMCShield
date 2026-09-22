"""
train_baseline.py
AMCShield - Baseline CNN Training Runner

Purpose:
    Wire the RadioML 2018.01A DataLoaders, BaselineCNN, and generic
    train_model() function into a baseline training experiment.

The model is trained with standard supervised learning.
No adversarial training is used here.

Input:
    RadioML 2018.01A
    X shape: (2, 1024)
    Classes: 24

Output:
    checkpoints/baseline_best.pt
"""

import torch

from src.data_loader import create_dataloaders
from src.model_baseline import BaselineCNN
from src.train import train_model, load_config


def main():
    print("=" * 70)
    print("AMCShield Baseline CNN Training")
    print("=" * 70)

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    config = load_config()

    batch_size = config.get("batch_size", 128)
    num_classes = config["dataset"].get("num_classes", 24)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dataset      : {config['dataset']['name']}")
    print(f"Classes      : {num_classes}")
    print(f"Batch size   : {batch_size}")
    print(f"Epochs       : {config.get('epochs', 30)}")
    print(f"Learning rate: {config.get('learning_rate', 0.001)}")
    print(f"Device       : {device}")

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------
    print("\nCreating DataLoaders...")

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=0,
    )

    print(f"Train batches : {len(train_loader):,}")
    print(f"Val batches   : {len(val_loader):,}")
    print(f"Test batches  : {len(test_loader):,}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    print("\nCreating BaselineCNN...")

    model = BaselineCNN(num_classes=num_classes)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    checkpoint_path = "checkpoints/baseline_best.pt"

    print("\nStarting baseline training...")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 70)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        adversarial_fn=None,
        checkpoint_path=checkpoint_path,
    )

    # ------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()