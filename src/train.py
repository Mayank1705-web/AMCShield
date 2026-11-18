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

import os
import random
import sys

# Windows redirects (>> training.log) use the cp1252 codepage, which
# crashes on characters like a check mark. Never let a print() kill
# a multi-day training run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils


# ============================================================
# Crash-safe checkpointing / resume helpers
#
# Problem solved:
#     If the laptop dies (battery, power cut, crash), training
#     used to restart from epoch 1. Now the FULL training state
#     is saved regularly and training continues from the exact
#     epoch AND batch where it stopped.
# ============================================================

def _atomic_save(obj, path: Path) -> None:
    """
    Write a checkpoint so that a power cut can never leave a
    half-written (corrupt) file behind.

    1. write to  <name>.tmp   (and flush to disk)
    2. keep the previous good file as <name>.prev
    3. rename .tmp -> final   (atomic rename)
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    prev = path.with_name(path.name + ".prev")

    torch.save(obj, tmp)

    try:
        with open(tmp, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass

    if path.exists():
        os.replace(path, prev)

    os.replace(tmp, path)


def _safe_load(path: Path):
    """
    Load a checkpoint; if the main file is missing/corrupt,
    fall back to the previous good copy (.prev).
    Returns (checkpoint_dict_or_None, path_used_or_None).
    """
    path = Path(path)
    candidates = [path, path.with_name(path.name + ".prev")]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            ckpt = torch.load(
                candidate,
                map_location="cpu",
                weights_only=False,
            )
            return ckpt, candidate
        except Exception as error:
            print(
                f"  ! Could not read {candidate.name}: "
                f"{str(error)[:80]}"
            )

    return None, None


def _resume_path_for(checkpoint_file: Path) -> Path:
    """checkpoints/baseline_best.pt -> checkpoints/baseline_resume.pt"""
    stem = checkpoint_file.stem
    if stem.endswith("_best"):
        stem = stem[: -len("_best")]
    return checkpoint_file.with_name(f"{stem}_resume.pt")


def _make_epoch_loader(
    train_loader,
    epoch: int,
    seed: int,
    start_batch: int,
):
    """
    Build the training DataLoader for one epoch.

    The shuffle order depends only on (seed, epoch), so after a
    crash the SAME order is regenerated and we simply skip the
    batches that were already trained (without reading them).
    """
    dataset = train_loader.dataset
    batch_size = train_loader.batch_size

    generator = torch.Generator()
    generator.manual_seed(int(seed) * 100003 + int(epoch))

    order = torch.randperm(
        len(dataset),
        generator=generator,
    ).tolist()

    order = order[start_batch * batch_size:]

    # Dedicated generator so the DataLoader never consumes the
    # global torch RNG (keeps resumed runs bit-for-bit identical).
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(seed) * 100003 + int(epoch))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=order,
        generator=loader_generator,
        shuffle=False,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last,
    )


def _get_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _set_rng_state(state: dict) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as error:
        print(f"  ! Could not restore RNG state: {error}")


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
    should_stop: Optional[Callable[[], bool]] = None,
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
    # RESUME SUPPORT
    #
    # resume_file  : full training state, saved often.
    #                (model + optimizer + scheduler + epoch +
    #                 batch + history + RNG)
    # checkpoint_file (baseline_best.pt): best model only.
    #
    # Fresh start instead of resume:
    #     set environment variable AMC_FRESH=1
    #     (or delete checkpoints/<name>_resume.pt)
    # ========================================================

    seed = config.get("seed", 42)

    resume_file = _resume_path_for(checkpoint_file)

    save_every_batches = int(
        config.get("resume_save_every_batches", 500)
    )

    start_epoch = 1
    start_batch = 0
    resume_running_loss = 0.0
    resume_correct = 0
    resume_total = 0
    early_stopped = False

    fresh = os.environ.get("AMC_FRESH", "0") == "1"
    warmstart = os.environ.get("AMC_WARMSTART", "0") == "1"

    state, used_path = (
        (None, None) if fresh else _safe_load(resume_file)
    )

    if state is not None:

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])

        start_epoch = state["epoch"]
        start_batch = state["batch"]
        resume_running_loss = state["running_loss"]
        resume_correct = state["correct"]
        resume_total = state["total"]
        best_val_accuracy = state["best_val_accuracy"]
        epochs_without_improvement = state[
            "epochs_without_improvement"
        ]
        history = state["history"]
        early_stopped = state.get("early_stopped", False)

        # If batch size changed, a mid-epoch position is invalid.
        if state.get("batch_size") != batch_size:
            start_batch = 0
            resume_running_loss = 0.0
            resume_correct = 0
            resume_total = 0

        _set_rng_state(state["rng"])

        print()
        print("=" * 70)
        print(f"RESUMING from {used_path}")
        print(
            f"  -> epoch {start_epoch}/{epochs}, "
            f"batch {start_batch} "
            f"(best val acc so far: "
            f"{best_val_accuracy * 100:.2f}%)"
        )
        print("=" * 70)

    elif warmstart and not fresh:

        # One-time rescue: continue from an existing *_best.pt
        # (e.g. a run that started before resume support).
        best_ckpt, best_used = _safe_load(checkpoint_file)

        if best_ckpt is not None:

            model.load_state_dict(best_ckpt["model_state_dict"])

            if "optimizer_state_dict" in best_ckpt:
                optimizer.load_state_dict(
                    best_ckpt["optimizer_state_dict"]
                )

            start_epoch = int(best_ckpt.get("epoch", 0)) + 1
            best_val_accuracy = float(
                best_ckpt.get("val_accuracy", -1.0)
            )

            print()
            print("=" * 70)
            print(f"WARM START from {best_used}")
            print(
                f"  -> continuing at epoch {start_epoch}/{epochs} "
                f"(best val acc: {best_val_accuracy * 100:.2f}%)"
            )
            print("=" * 70)

    elif fresh:
        print("AMC_FRESH=1 -> ignoring any saved resume state.")

    def save_resume_state(
        next_epoch: int,
        next_batch: int,
        run_loss: float,
        run_correct: int,
        run_total: int,
        stopped: bool = False,
    ) -> None:
        """Snapshot everything needed to continue training."""
        _atomic_save(
            {
                "epoch": next_epoch,
                "batch": next_batch,
                "batch_size": batch_size,
                "running_loss": run_loss,
                "correct": run_correct,
                "total": run_total,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_accuracy": best_val_accuracy,
                "epochs_without_improvement": (
                    epochs_without_improvement
                ),
                "history": history,
                "early_stopped": stopped,
                "rng": _get_rng_state(),
                "config": config,
            },
            resume_file,
        )

    # ========================================================
    # Overall training progress
    # ========================================================

    print()
    print("Training progress:")
    print()

    overall_progress = tqdm(
        total=epochs,
        initial=min(start_epoch - 1, epochs),
        desc="Overall Training",
        unit="epoch",
        dynamic_ncols=True,
        position=0,
    )

    # ========================================================
    # Epoch loop
    # ========================================================

    manual_stop_requested = False

    if early_stopped or start_epoch > epochs:
        print()
        print(
            "Saved state says training already finished "
            "(set AMC_FRESH=1 to retrain from scratch)."
        )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        if early_stopped:
            break

        model.train()

        # Continue mid-epoch if we are resuming this very epoch.
        if epoch == start_epoch and start_batch > 0:
            first_batch = start_batch
            running_loss = resume_running_loss
            correct = resume_correct
            total = resume_total
        else:
            first_batch = 0
            running_loss = 0.0
            correct = 0
            total = 0

        epoch_loader = _make_epoch_loader(
            train_loader,
            epoch,
            seed,
            first_batch,
        )

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
            enumerate(epoch_loader, start=first_batch),
            total=epoch_total_batches,
            initial=first_batch,
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

            # Honor a pending dashboard stop before processing this batch.
            if should_stop is not None and should_stop():
                save_resume_state(
                    epoch,
                    batch_index,
                    running_loss,
                    correct,
                    total,
                )
                manual_stop_requested = True
                print("\nDashboard stop requested before next batch. Resume state saved.")
                break

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

            # ------------------------------------------------
            # Periodic crash-safe snapshot (mid-epoch).
            # On restart, training continues from the next batch.
            # ------------------------------------------------

            if (
                save_every_batches > 0
                and (batch_index + 1) % save_every_batches == 0
            ):
                save_resume_state(
                    epoch,
                    batch_index + 1,
                    running_loss,
                    correct,
                    total,
                )

            # Dashboard Stop is cooperative: finish this batch, then
            # persist the exact next-batch position before leaving.
            if should_stop is not None and should_stop():
                save_resume_state(
                    epoch,
                    batch_index + 1,
                    running_loss,
                    correct,
                    total,
                )
                manual_stop_requested = True
                print("\nDashboard stop requested. Resume state saved.")
                break

        # ----------------------------------------------------
        # Close batch progress bar
        # ----------------------------------------------------

        batch_progress.close()

        if manual_stop_requested:
            break

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

            _atomic_save(
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
                f"  [OK] Best checkpoint saved: "
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

        stop_now = (
            epochs_without_improvement >= early_stopping_patience
        )

        # Epoch fully done: next run starts at the NEXT epoch.
        save_resume_state(
            epoch + 1,
            0,
            0.0,
            0,
            0,
            stopped=stop_now,
        )

        if stop_now:

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
    if manual_stop_requested:
        print("TRAINING PAUSED (manual stop)")
    else:
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