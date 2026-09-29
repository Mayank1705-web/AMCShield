"""
train.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 2

Responsibility:
    Shared generic training loop for baseline and robust models.

Supports:
    1. Normal supervised training
    2. Adversarial training through adversarial_fn

Input:
    DataLoader returning:
        signals: (B, 2, 1024)
        labels : (B,)
        snr    : (B,)

Model output:
    logits: (B, 24)

Progress:
    - Overall training progress across all epochs
    - Per-epoch batch progress
    - Loss and accuracy shown in tqdm
"""

from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src import utils


# ============================================================
# Validation
# ============================================================

def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    criterion,
    device: torch.device,
    max_batches: Optional[int] = None,
):
    """
    Evaluate model on a validation/test DataLoader.

    Args:
        model: PyTorch model.
        data_loader: Validation or test DataLoader.
        criterion: Loss function.
        device: CPU or CUDA device.
        max_batches: Optional limit for validation batches.

    Returns:
        average_loss: Average loss over evaluated samples.
        accuracy: Accuracy as a decimal between 0 and 1.
    """

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_index, (signals, labels, _snr) in enumerate(data_loader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            signals = signals.to(device)
            labels = labels.to(device)

            logits = model(signals)

            loss = criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item()
                * signals.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / total
        if total > 0
        else 0.0
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return average_loss, accuracy


# ============================================================
# Training
# ============================================================

def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: dict,
    adversarial_fn: Optional[Callable] = None,
    checkpoint_path: str = "checkpoints/model.pt",
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
) -> torch.nn.Module:
    """
    Generic training loop.

    If adversarial_fn is None:
        Normal supervised training.

    If adversarial_fn is provided:
        Adversarial training through the supplied hook.

    The adversarial function receives:

        model
        signals
        labels

    and returns adversarial signals.

    Progress bars:
        - Overall progress across all epochs
        - Batch progress inside each epoch

    Checkpointing:
        Best model is saved whenever validation accuracy improves.
    """

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    utils.set_seed(
        config.get("seed", 42)
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("AMCShield Training")
    print("=" * 70)

    print()
    print(f"Device : {device}")

    # --------------------------------------------------------
    # Move model
    # --------------------------------------------------------

    model = model.to(device)

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    batch_size = config.get(
        "batch_size",
        128,
    )

    epochs = config.get(
        "epochs",
        30,
    )

    learning_rate = config.get(
        "learning_rate",
        0.001,
    )

    # --------------------------------------------------------
    # Loss
    #
    # label_smoothing regularizes against over-confident
    # predictions, which tends to generalize better on the
    # ambiguous low-SNR / easily-confused modulation pairs
    # (e.g. QAM16 vs QAM64, AM-DSB vs AM-SSB) in RadioML.
    # --------------------------------------------------------

    label_smoothing = config.get(
        "label_smoothing",
        0.1,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )

    # --------------------------------------------------------
    # Optimizer
    #
    # AdamW (decoupled weight decay) instead of plain Adam.
    # --------------------------------------------------------

    weight_decay = config.get(
        "weight_decay",
        1e-4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # LR scheduler
    #
    # Reduces the learning rate once validation accuracy
    # plateaus, instead of training at a single fixed LR for
    # all epochs.
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopping_patience = config.get(
        "early_stopping_patience",
        7,
    )

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    checkpoint_file = Path(
        checkpoint_path
    )

    checkpoint_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Best validation accuracy
    # --------------------------------------------------------

    best_val_accuracy = -1.0

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # ========================================================
    # Overall training progress
    # ========================================================

    print()
    print("Training progress:")
    print()

    overall_progress = tqdm(
        total=epochs,
        desc="Overall Training",
        unit="epoch",
        dynamic_ncols=True,
        position=0,
    )

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # ----------------------------------------------------
        # Determine number of batches for progress bar
        # ----------------------------------------------------

        if max_train_batches is not None:
            epoch_total_batches = min(
                len(train_loader),
                max_train_batches,
            )
        else:
            epoch_total_batches = len(train_loader)

        # ----------------------------------------------------
        # Batch progress bar
        # ----------------------------------------------------

        batch_progress = tqdm(
            enumerate(train_loader),
            total=epoch_total_batches,
            desc=f"Epoch {epoch:02d}/{epochs:02d}",
            unit="batch",
            dynamic_ncols=True,
            position=1,
            leave=True,
        )

        # ----------------------------------------------------
        # Batch loop
        # ----------------------------------------------------

        for batch_index, (
            signals,
            labels,
            _snr,
        ) in batch_progress:

            if (
                max_train_batches is not None
                and batch_index >= max_train_batches
            ):
                break

            signals = signals.to(device)
            labels = labels.to(device)

            # ------------------------------------------------
            # Generate adversarial examples if requested.
            #
            # adv_train_ratio controls what fraction of each
            # batch is adversarially perturbed; the rest stays
            # clean. This mixing (rather than attacking 100%
            # of every batch) is what lets the robust model
            # keep clean accuracy close to the baseline while
            # still training against PGD examples.
            #
            # adv_train_ratio=1.0 (or omitted) reproduces the
            # previous behavior of attacking every batch.
            # ------------------------------------------------

            if adversarial_fn is not None:

                adv_train_ratio = float(
                    config.get("adv_train_ratio", 1.0)
                )

                batch_size_actual = signals.size(0)

                num_adv = int(
                    round(
                        adv_train_ratio
                        * batch_size_actual
                    )
                )

                if num_adv <= 0:

                    pass

                elif num_adv >= batch_size_actual:

                    signals = adversarial_fn(
                        model,
                        signals,
                        labels,
                    )

                else:

                    adv_signals = adversarial_fn(
                        model,
                        signals[:num_adv],
                        labels[:num_adv],
                    )

                    signals = torch.cat(
                        [adv_signals, signals[num_adv:]],
                        dim=0,
                    )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(
                signals
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels,
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping (helps stability, especially
            # on batches containing adversarial examples).
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            # ------------------------------------------------
            # Update weights
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            running_loss += (
                loss.item()
                * signals.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            batch_correct = (
                predictions == labels
            ).sum().item()

            correct += batch_correct

            total += labels.size(0)

            # ------------------------------------------------
            # Current batch accuracy
            # ------------------------------------------------

            batch_accuracy = (
                batch_correct / labels.size(0)
            ) * 100.0

            # ------------------------------------------------
            # Running training metrics
            # ------------------------------------------------

            running_accuracy = (
                correct / total
            ) * 100.0 if total > 0 else 0.0

            running_loss_average = (
                running_loss / total
            ) if total > 0 else 0.0

            # ------------------------------------------------
            # Update batch progress bar
            # ------------------------------------------------

            batch_progress.set_postfix(
                loss=f"{running_loss_average:.4f}",
                acc=f"{running_accuracy:.2f}%",
                batch_acc=f"{batch_accuracy:.1f}%",
            )

        # ----------------------------------------------------
        # Close batch progress bar
        # ----------------------------------------------------

        batch_progress.close()

        # ----------------------------------------------------
        # Epoch training metrics
        # ----------------------------------------------------

        train_loss = (
            running_loss / total
            if total > 0
            else 0.0
        )

        train_accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        # ----------------------------------------------------
        # Print epoch result
        # ----------------------------------------------------

        print(
            f"\nEpoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_accuracy * 100:.2f}% "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # LR scheduler step (driven by validation accuracy)
        # ----------------------------------------------------

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_file,
            )

            print(
                f"  ✓ Best checkpoint saved: "
                f"{checkpoint_file}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  No improvement for "
                f"{epochs_without_improvement} epoch(s) "
                f"(patience={early_stopping_patience}) "
                f"| LR: {current_lr:.6f}"
            )

        # ----------------------------------------------------
        # Update overall progress
        # ----------------------------------------------------

        overall_progress.update(1)

        overall_progress.set_postfix(
            epoch=f"{epoch}/{epochs}",
            train_acc=f"{train_accuracy * 100:.2f}%",
            val_acc=f"{val_accuracy * 100:.2f}%",
            lr=f"{current_lr:.6f}",
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= early_stopping_patience:

            print()
            print(
                f"Early stopping triggered after {epoch} "
                f"epochs (no val improvement for "
                f"{early_stopping_patience} epochs)."
            )

            break

    # ========================================================
    # Close overall progress
    # ========================================================

    overall_progress.close()

    # ========================================================
    # Training completed
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best checkpoint: "
        f"{checkpoint_file}"
    )

    return model


# ============================================================
# Configuration
# ============================================================

def load_config(
    path: str = "configs/config.yaml",
) -> dict:
    """
    Load YAML configuration.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


# ============================================================
# Manual sanity test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AMCShield Training Module")
    print("=" * 70)

    config = load_config()

    print()
    print(
        "Configuration loaded successfully."
    )

    print(
        f"Batch size     : "
        f"{config.get('batch_size')}"
    )

    print(
        f"Epochs         : "
        f"{config.get('epochs')}"
    )

    print(
        f"Learning rate  : "
        f"{config.get('learning_rate')}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device         : {device}"
    )

    print()
    print(
        "Training module syntax and "
        "configuration test passed."
    )