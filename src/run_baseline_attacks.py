# src/run_baseline_attacks.py

"""
AMCShield - Baseline Attack Evaluation

Evaluates the already-trained baseline model against:
    1. Clean
    2. FGSM
    3. PGD
    4. MIM
    5. C&W
    6. Black-box surrogate transfer

IMPORTANT:
    - Does NOT retrain the baseline.
    - Loads checkpoints/baseline_best.pt only.
    - Black-box attack uses the trained surrogate model.
    - Ground-truth labels are used ONLY for evaluation metrics.
    - Attack epsilon is read from configs/config.yaml.
    - No [0, 1] clipping is performed.
"""

from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.attacks import (
    fgsm_attack,
    pgd_attack,
    mim_attack,
    cw_attack,
    black_box_attack,
)
from src.data_loader import create_dataloaders
from src.model_baseline import BaselineCNN
from src.model_surrogate import SurrogateModel
from src.train import load_config


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

BASELINE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "baseline_best.pt"
SURROGATE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "surrogate_best.pt"

RESULTS_DIR = PROJECT_ROOT / "results"

SAMPLE_PER_SNR = 256
EVAL_BATCH_SIZE = 32

SEED = 42

EXPECTED_SNRS = list(range(-20, 31, 2))


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_baseline(config, device):
    """
    Load the already-trained baseline model.

    No training is performed.
    """

    if not BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found:\n{BASELINE_CHECKPOINT}"
        )

    model = BaselineCNN(
        num_classes=config["dataset"]["num_classes"]
    ).to(device)

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("✓ Baseline checkpoint loaded")
    print(f"  Checkpoint : {BASELINE_CHECKPOINT}")
    print(f"  Epoch      : {checkpoint.get('epoch', 'unknown')}")

    if "val_accuracy" in checkpoint:
        print(
            f"  Validation : "
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

    return model


def load_surrogate(config, device):
    """
    Load the already-trained black-box surrogate.
    """

    if not SURROGATE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Surrogate checkpoint not found:\n{SURROGATE_CHECKPOINT}"
        )

    model = SurrogateModel(
        num_classes=config["dataset"]["num_classes"]
    ).to(device)

    checkpoint = torch.load(
        SURROGATE_CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # The surrogate parameters must not be updated.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print("✓ Surrogate checkpoint loaded")
    print(
        f"  Validation : "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )
    print(
        f"  Query budget : "
        f"{checkpoint.get('query_budget', 'unknown')}"
    )

    return model


# ============================================================
# TEST-DATA SNR COLLECTION
# ============================================================

def collect_snr_samples(test_loader):
    """
    Collect exactly SAMPLE_PER_SNR samples for every SNR.

    Returns:
        {
            snr: (X, y)
        }

    X shape:
        (N, 2, 1024)

    y shape:
        (N,)
    """

    snr_samples = {
        snr: []
        for snr in EXPECTED_SNRS
    }

    snr_labels = {
        snr: []
        for snr in EXPECTED_SNRS
    }

    print()
    print("=" * 70)
    print("COLLECTING TEST SAMPLES BY SNR")
    print("=" * 70)

    with torch.no_grad():
        for X, y, snr in test_loader:

            X = X.cpu()
            y = y.long().cpu()
            snr = snr.cpu()

            for i in range(X.size(0)):

                current_snr = int(round(float(snr[i])))

                if current_snr not in snr_samples:
                    continue

                if len(snr_samples[current_snr]) >= SAMPLE_PER_SNR:
                    continue

                snr_samples[current_snr].append(X[i])
                snr_labels[current_snr].append(y[i])

            if all(
                len(snr_samples[s]) >= SAMPLE_PER_SNR
                for s in EXPECTED_SNRS
            ):
                break

    result = {}

    print()

    for snr in EXPECTED_SNRS:

        if len(snr_samples[snr]) < SAMPLE_PER_SNR:
            raise RuntimeError(
                f"Only {len(snr_samples[snr])} samples found "
                f"for SNR {snr} dB; "
                f"expected {SAMPLE_PER_SNR}."
            )

        X = torch.stack(
            snr_samples[snr][:SAMPLE_PER_SNR]
        )

        y = torch.stack(
            snr_labels[snr][:SAMPLE_PER_SNR]
        ).long()

        result[snr] = (X, y)

        print(
            f"SNR {snr:>3} dB : "
            f"{len(X):>3} samples"
        )

    print()
    print("✓ All 26 SNR levels collected")

    return result


# ============================================================
# METRICS
# ============================================================

def accuracy(model, X, y, device):
    """
    Compute model accuracy.
    """

    correct = 0
    total = 0

    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )

    with torch.no_grad():

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == batch_y
            ).sum().item()

            total += batch_y.size(0)

    return (
        100.0 * correct / total
        if total > 0
        else 0.0
    )


def attack_success_rate(
    clean_predictions,
    adversarial_predictions,
    y,
):
    """
    Attack Success Rate among samples that were
    correctly classified before the attack.

    ASR =
        successful attacks /
        initially-correct samples
    """

    clean_correct = clean_predictions == y

    successful_attack = (
        clean_correct
        & (adversarial_predictions != y)
    )

    denominator = clean_correct.sum().item()
    numerator = successful_attack.sum().item()

    if denominator == 0:
        return 0.0

    return 100.0 * numerator / denominator


def evaluate_predictions(
    model,
    X,
    y,
    device,
):
    """
    Return predictions from a model.
    """

    predictions = []

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )

    with torch.no_grad():

        for batch_X, _ in loader:

            batch_X = batch_X.to(device)

            logits = model(batch_X)

            batch_predictions = torch.argmax(
                logits,
                dim=1,
            )

            predictions.append(
                batch_predictions.cpu()
            )

    return torch.cat(predictions)


def evaluate_adversarial(
    model,
    X,
    y,
    adv_X,
    device,
):
    """
    Evaluate adversarial examples.
    """

    clean_predictions = evaluate_predictions(
        model,
        X,
        y,
        device,
    )

    adv_predictions = evaluate_predictions(
        model,
        adv_X,
        y,
        device,
    )

    clean_correct = (
        clean_predictions == y
    ).sum().item()

    adv_correct = (
        adv_predictions == y
    ).sum().item()

    total = len(y)

    clean_accuracy = (
        100.0 * clean_correct / total
        if total
        else 0.0
    )

    adv_accuracy = (
        100.0 * adv_correct / total
        if total
        else 0.0
    )

    drop = clean_accuracy - adv_accuracy

    asr = attack_success_rate(
        clean_predictions,
        adv_predictions,
        y,
    )

    return (
        clean_accuracy,
        adv_accuracy,
        drop,
        asr,
    )


# ============================================================
# C&W DISTORTION METRICS
# ============================================================

def compute_linf(original, adversarial):
    return (
        adversarial - original
    ).abs().flatten(1).max(dim=1).values


def compute_l2(original, adversarial):
    return (
        adversarial - original
    ).flatten(1).norm(p=2, dim=1)


# ============================================================
# CSV WRITERS
# ============================================================

def write_csv(filename, fieldnames, rows):

    output_path = RESULTS_DIR / filename

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Saved: {output_path}")

    return output_path


# ============================================================
# CLEAN EVALUATION
# ============================================================

def run_clean(
    baseline,
    snr_data,
    device,
):
    print()
    print("=" * 70)
    print("BASELINE CLEAN EVALUATION")
    print("=" * 70)

    rows = []

    for snr in EXPECTED_SNRS:

        X, y = snr_data[snr]

        clean_acc = accuracy(
            baseline,
            X,
            y,
            device,
        )

        rows.append({
            "snr": snr,
            "clean_accuracy": round(
                clean_acc,
                4,
            ),
        })

        print(
            f"SNR {snr:>3} dB | "
            f"Clean: {clean_acc:6.2f}%"
        )

    write_csv(
        "baseline_clean_results.csv",
        [
            "snr",
            "clean_accuracy",
        ],
        rows,
    )


# ============================================================
# STANDARD ATTACK EVALUATION
# ============================================================

def run_standard_attack(
    baseline,
    snr_data,
    device,
    attack_name,
    attack_function,
    epsilon,
    steps,
):
    print()
    print("=" * 70)
    print(f"BASELINE {attack_name.upper()} EVALUATION")
    print("=" * 70)

    rows = []

    for snr in EXPECTED_SNRS:

        X, y = snr_data[snr]

        X_device = X.to(device)
        y_device = y.to(device)

        # ----------------------------------------------------
        # Attack generation
        # ----------------------------------------------------

        if attack_name.upper() == "FGSM":
            adv_X = attack_function(
                baseline,
                X_device,
                y_device,
                epsilon,
            )
        else:
            adv_X = attack_function(
                baseline,
                X_device,
                y_device,
                epsilon,
                steps=steps,
            )

        adv_X = adv_X.detach()

        (
            clean_acc,
            attack_acc,
            drop,
            asr,
        ) = evaluate_adversarial(
            baseline,
            X,
            y,
            adv_X.cpu(),
            device,
        )

        rows.append({
            "snr": snr,
            "clean_accuracy": round(
                clean_acc,
                4,
            ),
            "attack_accuracy": round(
                attack_acc,
                4,
            ),
            "drop": round(
                drop,
                4,
            ),
            "attack_success_rate": round(
                asr,
                4,
            ),
        })

        print(
            f"SNR {snr:>3} dB | "
            f"Clean: {clean_acc:6.2f}% | "
            f"{attack_name}: {attack_acc:6.2f}% | "
            f"Drop: {drop:6.2f}% | "
            f"ASR: {asr:6.2f}%"
        )

    filename = (
        f"baseline_{attack_name.lower()}_results.csv"
    )

    write_csv(
        filename,
        [
            "snr",
            "clean_accuracy",
            "attack_accuracy",
            "drop",
            "attack_success_rate",
        ],
        rows,
    )


# ============================================================
# C&W EVALUATION
# ============================================================

def run_cw(
    baseline,
    snr_data,
    device,
):
    print()
    print("=" * 70)
    print("BASELINE C&W EVALUATION")
    print("=" * 70)

    rows = []

    for snr in EXPECTED_SNRS:

        X, y = snr_data[snr]

        X_device = X.to(device)
        y_device = y.to(device)

        # C&W is computationally expensive.
        # Use the available SNR sample set directly.
        adv_X = cw_attack(
            baseline,
            X_device,
            y_device,
        )

        adv_X = adv_X.detach()

        (
            clean_acc,
            attack_acc,
            drop,
            asr,
        ) = evaluate_adversarial(
            baseline,
            X,
            y,
            adv_X.cpu(),
            device,
        )

        linf = compute_linf(
            X_device,
            adv_X,
        )

        l2 = compute_l2(
            X_device,
            adv_X,
        )

        rows.append({
            "snr": snr,
            "clean_accuracy": round(
                clean_acc,
                4,
            ),
            "cw_accuracy": round(
                attack_acc,
                4,
            ),
            "attack_success_rate": round(
                asr,
                4,
            ),
            "mean_linf": round(
                linf.mean().item(),
                6,
            ),
            "max_linf": round(
                linf.max().item(),
                6,
            ),
            "mean_l2": round(
                l2.mean().item(),
                6,
            ),
            "max_l2": round(
                l2.max().item(),
                6,
            ),
        })

        print(
            f"SNR {snr:>3} dB | "
            f"Clean: {clean_acc:6.2f}% | "
            f"C&W: {attack_acc:6.2f}% | "
            f"Drop: {drop:6.2f}% | "
            f"ASR: {asr:6.2f}% | "
            f"Mean L∞: {linf.mean().item():.4f} | "
            f"Mean L2: {l2.mean().item():.4f}"
        )

    write_csv(
        "baseline_cw_results.csv",
        [
            "snr",
            "clean_accuracy",
            "cw_accuracy",
            "attack_success_rate",
            "mean_linf",
            "max_linf",
            "mean_l2",
            "max_l2",
        ],
        rows,
    )


# ============================================================
# BLACK-BOX EVALUATION
# ============================================================

def run_blackbox(
    baseline,
    surrogate,
    snr_data,
    device,
    epsilon,
    steps,
):
    print()
    print("=" * 70)
    print("BASELINE BLACK-BOX SURROGATE TRANSFER")
    print("=" * 70)

    rows = []

    for snr in EXPECTED_SNRS:

        X, y = snr_data[snr]

        X_device = X.to(device)
        y_device = y.to(device)

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # black_box_attack generates the perturbation using
        # ONLY the surrogate.
        #
        # The baseline is NOT differentiated through.
        #
        # Ground truth y is used only afterward for metrics.
        # ----------------------------------------------------

        adv_X = black_box_attack(
            target_model=baseline,
            x=X_device,
            y=y_device,
            surrogate_model=surrogate,
            epsilon=epsilon,
            steps=steps,
        )

        adv_X = adv_X.detach()

        (
            clean_acc,
            blackbox_acc,
            drop,
            asr,
        ) = evaluate_adversarial(
            baseline,
            X,
            y,
            adv_X.cpu(),
            device,
        )

        rows.append({
            "snr": snr,
            "clean_accuracy": round(
                clean_acc,
                4,
            ),
            "blackbox_accuracy": round(
                blackbox_acc,
                4,
            ),
            "drop": round(
                drop,
                4,
            ),
            "attack_success_rate": round(
                asr,
                4,
            ),
        })

        print(
            f"SNR {snr:>3} dB | "
            f"Clean: {clean_acc:6.2f}% | "
            f"Black-box: {blackbox_acc:6.2f}% | "
            f"Drop: {drop:6.2f}% | "
            f"ASR: {asr:6.2f}%"
        )

    write_csv(
        "baseline_blackbox_results.csv",
        [
            "snr",
            "clean_accuracy",
            "blackbox_accuracy",
            "drop",
            "attack_success_rate",
        ],
        rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMCShield - Baseline Attack Evaluation")
    print("=" * 70)

    set_seed(SEED)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config(CONFIG_PATH)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    epsilon_fgsm = config["epsilon"]["fgsm"]
    epsilon_pgd = config["epsilon"]["pgd"]
    epsilon_mim = config["epsilon"]["mim"]

    pgd_steps = config["pgd_steps"]
    mim_steps = config["mim_steps"]

    print()
    print(f"Dataset       : {config['dataset']['name']}")
    print(f"Device        : {device}")
    print(f"Samples/SNR   : {SAMPLE_PER_SNR}")
    print(f"FGSM epsilon  : {epsilon_fgsm}")
    print(f"PGD epsilon   : {epsilon_pgd}")
    print(f"MIM epsilon   : {epsilon_mim}")
    print(f"PGD steps     : {pgd_steps}")
    print(f"MIM steps     : {mim_steps}")

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print("Loading already-trained models...")

    baseline = load_baseline(
        config,
        device,
    )

    surrogate = load_surrogate(
        config,
        device,
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print()
    print("Creating test DataLoader...")

    _, _, test_loader = create_dataloaders(
        hdf5_path=config["dataset"]["path"],
        processed_dir="data/processed",
        batch_size=config["batch_size"],
        num_workers=0,
    )

    # --------------------------------------------------------
    # Collect fixed SNR evaluation samples
    # --------------------------------------------------------

    snr_data = collect_snr_samples(
        test_loader,
    )

    # --------------------------------------------------------
    # 1. CLEAN
    # --------------------------------------------------------

    run_clean(
        baseline,
        snr_data,
        device,
    )

    # --------------------------------------------------------
    # 2. FGSM
    # --------------------------------------------------------

    run_standard_attack(
        baseline=baseline,
        snr_data=snr_data,
        device=device,
        attack_name="FGSM",
        attack_function=fgsm_attack,
        epsilon=epsilon_fgsm,
        steps=1,
    )

    # --------------------------------------------------------
    # 3. PGD
    # --------------------------------------------------------

    run_standard_attack(
        baseline=baseline,
        snr_data=snr_data,
        device=device,
        attack_name="PGD",
        attack_function=pgd_attack,
        epsilon=epsilon_pgd,
        steps=pgd_steps,
    )

    # --------------------------------------------------------
    # 4. MIM
    # --------------------------------------------------------

    run_standard_attack(
        baseline=baseline,
        snr_data=snr_data,
        device=device,
        attack_name="MIM",
        attack_function=mim_attack,
        epsilon=epsilon_mim,
        steps=mim_steps,
    )

    # --------------------------------------------------------
    # 5. C&W
    # --------------------------------------------------------

    run_cw(
        baseline,
        snr_data,
        device,
    )

    # --------------------------------------------------------
    # 6. BLACK-BOX
    # --------------------------------------------------------

    run_blackbox(
        baseline=baseline,
        surrogate=surrogate,
        snr_data=snr_data,
        device=device,
        epsilon=epsilon_pgd,
        steps=pgd_steps,
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    expected_files = [
        "baseline_clean_results.csv",
        "baseline_fgsm_results.csv",
        "baseline_pgd_results.csv",
        "baseline_mim_results.csv",
        "baseline_cw_results.csv",
        "baseline_blackbox_results.csv",
    ]

    print()
    print("=" * 70)
    print("BASELINE ATTACK EVALUATION COMPLETED")
    print("=" * 70)

    all_exist = True

    for filename in expected_files:

        path = RESULTS_DIR / filename

        exists = path.exists()

        print(
            f"{'✓' if exists else '✗'} "
            f"{filename}"
        )

        if not exists:
            all_exist = False

    if not all_exist:
        raise RuntimeError(
            "One or more expected result files were not generated."
        )

    print()
    print("✓ All six baseline result files generated.")
    print("✓ Baseline model was not retrained.")


if __name__ == "__main__":
    main()