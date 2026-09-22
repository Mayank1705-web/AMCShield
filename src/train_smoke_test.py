"""
AMCShield Training Smoke Test

Tests the real train_model() function using only a few batches.
"""

from src.data_loader import create_dataloaders
from src.model_baseline import BaselineCNN
from src.train import train_model, load_config
from src.utils import load_checkpoint


def main():

    print("=" * 70)
    print("AMCShield - train_model() Smoke Test")
    print("=" * 70)

    config = load_config()

    # Small batch only for testing
    train_loader, val_loader, _ = create_dataloaders(
        batch_size=4,
        num_workers=0,
    )

    model = BaselineCNN(
        num_classes=24
    )

    print("\nStarting training...")

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_path="checkpoints/smoke_test_train_model.pt",
        max_train_batches=2,
        max_val_batches=1,
    )

    print("\nLoading checkpoint...")

    model = load_checkpoint(
        model,
        "checkpoints/smoke_test_train_model.pt",
    )

    print("Checkpoint loaded successfully.")

    print("\n" + "=" * 70)
    print("train_model() SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()