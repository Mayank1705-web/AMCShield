"""
evaluate.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 2 (baseline SNR curve) -> Phase 5 (full generalization-gap heatmap)

Responsibility:
    - Clean accuracy vs SNR
    - Attack success rate vs SNR
    - Confusion matrix
    - Generalization-gap heatmap data
    - Saving evaluation metrics

Dataset:
    RadioML 2018.01A

Input:
    I/Q signal shape: (2, 1024)
    Number of classes: 24
    SNR range: -20 dB to 30 dB in 2 dB steps

IMPORTANT:
    SNR is always kept aligned with X and y.
    All evaluation is performed per SNR bucket.

Expected test_data format:

    {
        "X": np.ndarray,       # shape (N, 2, 1024)
        "y": np.ndarray,       # shape (N,)
        "snr": np.ndarray      # shape (N,)
    }

The evaluator also supports a PyTorch DataLoader returning:

    signal, label, snr

where:
    signal -> (batch, 2, 1024)
    label  -> (batch,)
    snr    -> (batch,)
"""

import os
import json
from typing import Dict, Callable, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 24

DEFAULT_CLASS_NAMES = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]


# ============================================================
# Helper functions
# ============================================================

def _get_device(model: torch.nn.Module) -> torch.device:
    """
    Get the device on which the model is located.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_test_data(test_data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and prepare test data.

    Expected:
        X   -> (N, 2, 1024)
        y   -> (N,)
        snr -> (N,)

    Returns:
        X, y, snr
    """

    if not isinstance(test_data, dict):
        raise TypeError(
            "test_data must be a dictionary containing 'X', 'y', and 'snr'."
        )

    required_keys = {"X", "y", "snr"}

    missing = required_keys - set(test_data.keys())

    if missing:
        raise KeyError(
            f"test_data is missing required keys: {sorted(missing)}"
        )

    X = np.asarray(test_data["X"])
    y = np.asarray(test_data["y"]).reshape(-1)
    snr = np.asarray(test_data["snr"]).reshape(-1)

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to have 3 dimensions (N, 2, 1024), got {X.shape}"
        )

    if X.shape[1] != 2 or X.shape[2] != 1024:
        raise ValueError(
            f"Expected X shape (N, 2, 1024), got {X.shape}"
        )

    if len(X) != len(y) or len(X) != len(snr):
        raise ValueError(
            "X, y, and snr must contain the same number of samples."
        )

    return X, y.astype(np.int64), snr.astype(np.int64)


def _predict_numpy(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run model inference on a NumPy array.

    Input:
        X -> (N, 2, 1024)

    Output:
        predictions -> (N,)
    """

    device = _get_device(model)

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(0, len(X), batch_size):

            end = min(start + batch_size, len(X))

            batch = torch.from_numpy(
                X[start:end]
            ).float().to(device)

            logits = model(batch)

            pred = torch.argmax(logits, dim=1)

            predictions.append(
                pred.cpu().numpy()
            )

    if not predictions:
        return np.empty(0, dtype=np.int64)

    return np.concatenate(predictions)


def _get_snr_values(snr: np.ndarray) -> np.ndarray:
    """
    Return sorted unique SNR values.
    """

    return np.sort(
        np.unique(snr)
    )


# ============================================================
# 1. Accuracy by SNR
# ============================================================

def accuracy_by_snr(
    model,
    test_data: dict,
    batch_size: int = 256,
) -> Dict[int, float]:
    """
    Calculate clean classification accuracy separately for
    every SNR bucket.

    Returns:
        {
            -20: accuracy,
            -18: accuracy,
            ...
             30: accuracy
        }

    Accuracy is returned as a value between 0 and 1.
    """

    X, y, snr = _prepare_test_data(test_data)

    predictions = _predict_numpy(
        model,
        X,
        batch_size=batch_size,
    )

    results = {}

    for snr_value in _get_snr_values(snr):

        mask = snr == snr_value

        total = int(np.sum(mask))

        if total == 0:
            results[int(snr_value)] = 0.0
            continue

        correct = int(
            np.sum(
                predictions[mask] == y[mask]
            )
        )

        results[int(snr_value)] = correct / total

    return results


# ============================================================
# 2. Attack Success Rate by SNR
# ============================================================

def attack_success_rate(
    model,
    test_data: dict,
    attack_fn: Callable,
    batch_size: int = 256,
) -> Dict[int, float]:
    """
    Calculate attack success rate separately for every SNR.

    Definition:

        Attack Success Rate =
            correctly classified clean samples
            that become misclassified after attack
            ------------------------------------------------
            correctly classified clean samples

    IMPORTANT:
        Incorrectly classified clean samples are NOT counted
        in the denominator.

    attack_fn must accept:

        attack_fn(model, inputs, labels)

    and return adversarial inputs.

    Returns:
        {
            snr_value: attack_success_rate
        }
    """

    X, y, snr = _prepare_test_data(test_data)

    device = _get_device(model)

    model.eval()

    results = {}

    for snr_value in _get_snr_values(snr):

        mask = snr == snr_value

        X_snr = X[mask]
        y_snr = y[mask]

        clean_predictions = _predict_numpy(
            model,
            X_snr,
            batch_size=batch_size,
        )

        clean_correct_mask = (
            clean_predictions == y_snr
        )

        num_clean_correct = int(
            np.sum(clean_correct_mask)
        )

        if num_clean_correct == 0:
            results[int(snr_value)] = 0.0
            continue

        X_correct = X_snr[
            clean_correct_mask
        ]

        y_correct = y_snr[
            clean_correct_mask
        ]

        successful_attacks = 0

        for start in range(
            0,
            len(X_correct),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(X_correct),
            )

            inputs = torch.from_numpy(
                X_correct[start:end]
            ).float().to(device)

            labels = torch.from_numpy(
                y_correct[start:end]
            ).long().to(device)

            with torch.enable_grad():

                adversarial_inputs = attack_fn(
                    model,
                    inputs,
                    labels,
                )

            with torch.no_grad():

                logits = model(
                    adversarial_inputs
                )

                adversarial_predictions = torch.argmax(
                    logits,
                    dim=1,
                )

            successful_attacks += int(
                torch.sum(
                    adversarial_predictions != labels
                ).item()
            )

        results[int(snr_value)] = (
            successful_attacks /
            num_clean_correct
        )

    return results


# ============================================================
# 3. Generalization Gap Heatmap
# ============================================================

def generalization_gap_heatmap(
    robust_model,
    test_data: dict,
    attack_fns: Dict[str, Callable],
    batch_size: int = 256,
) -> Dict:
    """
    CORE PROJECT DELIVERABLE.

    Evaluate the robust model against:

        - PGD
        - FGSM
        - MIM
        - C&W
        - Black-box

    across every SNR bucket.

    Returns:

        {
            "pgd": {
                "-20": value,
                "-18": value,
                ...
            },

            "fgsm": {
                ...
            }
        }

    The function does not calculate a mathematical difference
    between attacks. Instead, it provides the per-SNR attack
    success rates needed to compare robustness against the
    trained-against attack (PGD) and unseen attacks.

    attack_fns example:

        {
            "pgd": pgd_attack,
            "fgsm": fgsm_attack,
            "mim": mim_attack,
            "cw": cw_attack,
            "black_box": black_box_attack
        }
    """

    if not isinstance(attack_fns, dict):
        raise TypeError(
            "attack_fns must be a dictionary of attack names "
            "to attack functions."
        )

    results = {}

    for attack_name, attack_fn in attack_fns.items():

        print(
            f"Evaluating attack: {attack_name}"
        )

        results[attack_name] = attack_success_rate(
            robust_model,
            test_data,
            attack_fn,
            batch_size=batch_size,
        )

    return results


# ============================================================
# 4. Confusion Matrix
# ============================================================

def confusion_matrix_plot(
    model,
    test_data: dict,
    snr_value: int,
    save_path: str,
    class_names: Optional[list] = None,
    batch_size: int = 256,
) -> None:
    """
    Generate a confusion matrix for one SNR value.

    Example:
        confusion_matrix_plot(
            model,
            test_data,
            snr_value=0,
            save_path="results/plots/confusion_matrix_0db.png"
        )
    """

    X, y, snr = _prepare_test_data(test_data)

    mask = snr == snr_value

    if not np.any(mask):
        raise ValueError(
            f"No samples found for SNR={snr_value} dB."
        )

    X_snr = X[mask]
    y_snr = y[mask]

    predictions = _predict_numpy(
        model,
        X_snr,
        batch_size=batch_size,
    )

    cm = confusion_matrix(
        y_snr,
        predictions,
        labels=np.arange(NUM_CLASSES),
    )

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if len(class_names) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class names, "
            f"got {len(class_names)}."
        )

    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True,
    )

    plt.figure(
        figsize=(14, 12)
    )

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        fmt="d",
    )

    plt.title(
        f"Confusion Matrix - RadioML 2018.01A ({snr_value} dB)"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        rotation=90
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 5. Accuracy vs SNR Plot
# ============================================================

def plot_accuracy_by_snr(
    results: Dict[int, float],
    save_path: str,
    title: str = "RadioML 2018.01A - Accuracy vs SNR",
) -> None:
    """
    Plot clean classification accuracy vs SNR.
    """

    if not results:
        raise ValueError(
            "Accuracy results are empty."
        )

    snr_values = sorted(
        results.keys()
    )

    accuracies = [
        results[snr]
        for snr in snr_values
    ]

    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        snr_values,
        np.array(accuracies) * 100,
        marker="o",
    )

    plt.xlabel(
        "SNR (dB)"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 6. Attack Success Rate Plot
# ============================================================

def plot_attack_success_rates(
    results: Dict[str, Dict[int, float]],
    save_path: str,
    title: str = "Attack Success Rate vs SNR",
) -> None:
    """
    Plot attack success rates for multiple attacks.
    """

    if not results:
        raise ValueError(
            "Attack results are empty."
        )

    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True,
    )

    plt.figure(
        figsize=(11, 7)
    )

    for attack_name, snr_results in results.items():

        snr_values = sorted(
            snr_results.keys()
        )

        rates = [
            snr_results[snr]
            for snr in snr_values
        ]

        plt.plot(
            snr_values,
            np.array(rates) * 100,
            marker="o",
            label=attack_name.upper(),
        )

    plt.xlabel(
        "SNR (dB)"
    )

    plt.ylabel(
        "Attack Success Rate (%)"
    )

    plt.title(
        title
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 7. Save Metrics
# ============================================================

def save_metrics(
    metrics: dict,
    path: str = "results/metrics.json",
) -> None:
    """
    Save evaluation metrics to JSON.

    NumPy integer keys and values are converted into standard
    Python types so that json.dump() works correctly.
    """

    def convert(obj):

        if isinstance(obj, dict):
            return {
                str(k): convert(v)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [
                convert(v)
                for v in obj
            ]

        if isinstance(obj, tuple):
            return [
                convert(v)
                for v in obj
            ]

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return obj

    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    converted_metrics = convert(metrics)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            converted_metrics,
            f,
            indent=2,
        )


# ============================================================
# 8. Evaluation Summary
# ============================================================

def print_accuracy_summary(
    results: Dict[int, float],
) -> None:
    """
    Print clean accuracy for every SNR bucket.
    """

    print()
    print("=" * 70)
    print("CLEAN ACCURACY BY SNR")
    print("=" * 70)

    for snr_value in sorted(results):

        accuracy = results[snr_value] * 100

        print(
            f"SNR {snr_value:>3} dB : "
            f"{accuracy:>7.2f}%"
        )

    print("=" * 70)


def print_attack_summary(
    results: Dict[str, Dict[int, float]],
) -> None:
    """
    Print attack success rates for every attack and SNR.
    """

    print()
    print("=" * 70)
    print("ATTACK SUCCESS RATE BY SNR")
    print("=" * 70)

    for attack_name, attack_results in results.items():

        print()
        print(f"Attack: {attack_name.upper()}")

        for snr_value in sorted(attack_results):

            rate = attack_results[snr_value] * 100

            print(
                f"  SNR {snr_value:>3} dB : "
                f"{rate:>7.2f}%"
            )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AMCShield Evaluation Module")
    print("=" * 70)

    print()
    print("Dataset configuration:")
    print("  Dataset      : RadioML 2018.01A")
    print("  Classes      : 24")
    print("  Input shape  : (2, 1024)")
    print("  SNR range    : -20 to 30 dB")
    print("  SNR step     : 2 dB")

    print()
    print("Evaluation functions available:")
    print("  ✓ accuracy_by_snr()")
    print("  ✓ attack_success_rate()")
    print("  ✓ generalization_gap_heatmap()")
    print("  ✓ confusion_matrix_plot()")
    print("  ✓ plot_accuracy_by_snr()")
    print("  ✓ plot_attack_success_rates()")
    print("  ✓ save_metrics()")

    print()
    print("Evaluation module syntax test passed.")