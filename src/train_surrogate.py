"""
train_surrogate.py

AMCShield - Surrogate Training

Training procedure:

    Training signals
          |
          v
    BaselineCNN query
          |
          v
    Baseline predicted labels
          |
          v
    Surrogate training

Ground-truth labels are NOT used for surrogate training.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.data_loader import create_dataloaders
from src.model_baseline import BaselineCNN
from src.model_surrogate import (
    SurrogateModel,
    collect_query_pairs,
)
from src.train import load_config


# ============================================================
# Configuration
# ============================================================

BASELINE_CHECKPOINT = (
    "checkpoints/baseline_best.pt"
)

SURROGATE_CHECKPOINT = (
    "checkpoints/surrogate_best.pt"
)

QUERY_BATCH_SIZE = 128

SURROGATE_EPOCHS = 20

LEARNING_RATE = 0.001

VAL_RATIO = 0.20


# ============================================================
# Device
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# Collect query signals
# ============================================================

def collect_training_signals(
    train_loader,
    query_budget,
):
    """
    Collect signals from the training split.

    Ground-truth labels and SNR values are ignored.
    """

    signals = []

    collected = 0

    print()
    print(
        "Collecting query signals..."
    )

    for batch_signals, _labels, _snr in train_loader:

        remaining = (
            query_budget - collected
        )

        if remaining <= 0:
            break

        take = min(
            remaining,
            batch_signals.size(0),
        )

        signals.append(
            batch_signals[
                :take
            ].cpu()
        )

        collected += take

        if collected >= query_budget:
            break

    if not signals:

        raise RuntimeError(
            "No query signals were collected."
        )

    X_query = torch.cat(
        signals,
        dim=0,
    )

    print(
        f"✓ Query signals collected: "
        f"{len(X_query):,}"
    )

    return X_query


# ============================================================
# Train
# ============================================================

def main():

    print("=" * 70)
    print("AMCShield - Black-Box Surrogate Training")
    print("=" * 70)

    config = load_config()

    device = get_device()

    num_classes = int(
        config["dataset"].get(
            "num_classes",
            24,
        )
    )

    query_budget = int(
        config.get(
            "surrogate_query_budget",
            5000,
        )
    )

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    print()
    print(
        f"Dataset          : "
        f"{config['dataset']['name']}"
    )

    print(
        f"Classes          : "
        f"{num_classes}"
    )

    print(
        f"Query budget     : "
        f"{query_budget:,}"
    )

    print(
        f"Device           : "
        f"{device}"
    )

    # --------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------

    print()
    print(
        "Loading trained baseline..."
    )

    baseline = BaselineCNN(
        num_classes=num_classes
    )

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location=device,
    )

    baseline.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    baseline.to(device)
    baseline.eval()

    print(
        "✓ Baseline checkpoint loaded"
    )

    if "epoch" in checkpoint:

        print(
            f"  Epoch           : "
            f"{checkpoint['epoch']}"
        )

    if "val_accuracy" in checkpoint:

        print(
            f"  Validation Acc. : "
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print()
    print(
        "Creating training DataLoader..."
    )

    train_loader, _, _ = create_dataloaders(
        batch_size=QUERY_BATCH_SIZE,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Collect X_query
    # --------------------------------------------------------

    X_query = collect_training_signals(
        train_loader=train_loader,
        query_budget=query_budget,
    )

    # --------------------------------------------------------
    # Query baseline
    # --------------------------------------------------------

    print()
    print(
        "Querying baseline model..."
    )

    query_X, query_labels = (
        collect_query_pairs(
            baseline_model=baseline,
            X_query=X_query,
            query_budget=query_budget,
        )
    )

    print(
        f"✓ Baseline queries completed: "
        f"{len(query_X):,}"
    )

    print(
        "✓ Only baseline-predicted labels "
        "are used for surrogate training."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    query_dataset = TensorDataset(
        query_X,
        query_labels,
    )

    validation_size = int(
        len(query_dataset)
        * VAL_RATIO
    )

    training_size = (
        len(query_dataset)
        - validation_size
    )

    generator = torch.Generator().manual_seed(
        seed
    )

    train_dataset, val_dataset = (
        random_split(
            query_dataset,
            [
                training_size,
                validation_size,
            ],
            generator=generator,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=QUERY_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=QUERY_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print(
        f"Surrogate train samples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Surrogate val samples  : "
        f"{len(val_dataset):,}"
    )

    # --------------------------------------------------------
    # Create surrogate
    # --------------------------------------------------------

    surrogate = SurrogateModel(
        num_classes=num_classes
    ).to(device)

    total_params = sum(
        p.numel()
        for p in surrogate.parameters()
    )

    print()
    print(
        f"Surrogate parameters: "
        f"{total_params:,}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        surrogate.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = -1.0

    Path(
        SURROGATE_CHECKPOINT
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("STARTING SURROGATE TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        SURROGATE_EPOCHS + 1,
    ):

        surrogate.train()

        train_correct = 0
        train_total = 0
        train_loss_total = 0.0

        for signals, labels in train_loader:

            signals = signals.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = surrogate(
                signals
            )

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            train_loss_total += (
                loss.item()
                * signals.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += (
                labels.size(0)
            )

        train_loss = (
            train_loss_total
            / train_total
        )

        train_accuracy = (
            train_correct
            / train_total
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        surrogate.eval()

        val_correct = 0
        val_total = 0
        val_loss_total = 0.0

        with torch.no_grad():

            for signals, labels in val_loader:

                signals = signals.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                logits = surrogate(
                    signals
                )

                loss = criterion(
                    logits,
                    labels,
                )

                val_loss_total += (
                    loss.item()
                    * signals.size(0)
                )

                predictions = torch.argmax(
                    logits,
                    dim=1,
                )

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += (
                    labels.size(0)
                )

        val_loss = (
            val_loss_total
            / val_total
        )

        val_accuracy = (
            val_correct
            / val_total
        )

        print(
            f"Epoch [{epoch:02d}/{SURROGATE_EPOCHS:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": (
                        surrogate.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "val_accuracy": (
                        val_accuracy
                    ),
                    "val_loss": val_loss,
                    "query_budget": (
                        query_budget
                    ),
                    "config": config,
                },
                SURROGATE_CHECKPOINT,
            )

            print(
                f"  ✓ Best surrogate checkpoint saved: "
                f"{SURROGATE_CHECKPOINT}"
            )

    print()
    print("=" * 70)
    print("SURROGATE TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Checkpoint: "
        f"{SURROGATE_CHECKPOINT}"
    )


if __name__ == "__main__":
    main()