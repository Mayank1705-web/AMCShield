"""
evaluate.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 2 (baseline SNR curve) -> Phase 5 (full generalization-gap heatmap)

Responsibility: accuracy-vs-SNR, confusion matrix, and the
generalization-gap heatmap -- the project's CORE DELIVERABLE.

Always evaluate per-SNR-bucket, never only an aggregate accuracy number.
Keep the generalization-gap computation in one clearly isolated,
well-named function for traceability (see architecture.md Section 4,
integration point #5).
"""

import torch
import numpy as np
import json
from typing import Dict, Callable


def accuracy_by_snr(model, test_data: dict) -> Dict[int, float]:
    """
    Returns {snr_value: accuracy} for clean (unattacked) evaluation.
    TODO (Person C).
    """
    raise NotImplementedError


def attack_success_rate(model, test_data: dict,
                          attack_fn: Callable) -> Dict[int, float]:
    """
    Returns {snr_value: attack_success_rate} for a given attack function
    from attacks.py. attack_success_rate = fraction of correctly-classified
    clean samples that become misclassified after the attack.
    TODO (Person C).
    """
    raise NotImplementedError


def generalization_gap_heatmap(robust_model, test_data: dict,
                                 attack_fns: Dict[str, Callable]) -> Dict:
    """
    THE CORE DELIVERABLE. For the robust model, compares attack success
    rate under PGD (what it was trained against) vs. every unseen attack
    (FGSM, MIM, C&W, black-box), across every SNR bucket.

    Returns a nested dict: {attack_name: {snr_value: success_rate}}
    with 'pgd' as one of the attack_name keys (the trained-against baseline
    for comparison).

    TODO (Person C, Phase 5). This function's output is what
    frontend/app.py's Dashboard tab reads via metrics.json -- keep the
    schema stable once implemented.
    """
    raise NotImplementedError


def confusion_matrix_plot(model, test_data: dict, snr_value: int,
                            save_path: str) -> None:
    """Confusion matrix at a fixed representative SNR (e.g., 0dB). TODO (Person C)."""
    raise NotImplementedError


def save_metrics(metrics: dict, path: str = "results/metrics.json") -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    # TODO: load models + test data, run full evaluation suite,
    # save to results/metrics.json and results/plots/
    pass
