"""
AMCShield - Robust Model Black-Box Surrogate Transfer Evaluation

Evaluates the already-trained robust target model against the already-trained
surrogate model using the existing black_box_attack implementation.

Threat model:
    - Target model is queried only for labels/predictions.
    - Attack gradients are obtained from the surrogate.
    - Ground-truth labels are NOT used to construct the attack.
    - Neither target nor surrogate model is retrained.

Output:
    results/blackbox_results.csv

Expected:
    26 SNR rows (-20 dB to +30 dB)
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import yaml

from src.attacks import black_box_attack
from src.data_loader import create_dataloaders
from src.model_robust import RobustCNN
from src.model_surrogate import SurrogateModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

ROBUST_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "robust_best.pt"
SURROGATE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "surrogate_best.pt"

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = OUTPUT_DIR / "blackbox_results.csv"


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLES_PER_SNR = 256

EPSILON = 0.02
STEPS = 10

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CONFIG
# ============================================================

def load_config():
    """Load project configuration."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# MODEL LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path):
    """
    Load an already-trained checkpoint into a model.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    # Handle standard AMCShield checkpoint format.
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback if the checkpoint itself is a state_dict.
        model.load_state_dict(checkpoint)

    return checkpoint


def build_models(config):
    """Create and load robust target and surrogate models."""

    num_classes = config["dataset"]["num_classes"]

    # --------------------------------------------------------
    # Robust target model
    # --------------------------------------------------------

    robust_model = RobustCNN(
        num_classes=num_classes
    ).to(DEVICE)

    robust_checkpoint = load_checkpoint(
        robust_model,
        ROBUST_CHECKPOINT,
    )

    robust_model.eval()

    # --------------------------------------------------------
    # Surrogate model
    # --------------------------------------------------------

    surrogate_model = SurrogateModel(
        num_classes=num_classes
    ).to(DEVICE)

    surrogate_checkpoint = load_checkpoint(
        surrogate_model,
        SURROGATE_CHECKPOINT,
    )

    surrogate_model.eval()

    return (
        robust_model,
        surrogate_model,
        robust_checkpoint,
        surrogate_checkpoint,
    )


# ============================================================
# TEST DATA COLLECTION
# ============================================================

def collect_samples_by_snr(config):
    """
    Collect exactly SAMPLES_PER_SNR test samples for each SNR.

    Returns:
        {
            snr: {
                "x": Tensor [N, 2, 1024],
                "y": Tensor [N]
            }
        }
    """

    print()
    print("=" * 70)
    print("COLLECTING TEST SAMPLES BY SNR")
    print("=" * 70)

    _, _, test_loader = create_dataloaders(
        hdf5_path=config["dataset"]["path"],
        processed_dir="data/processed",
        batch_size=config["batch_size"],
        num_workers=0,
    )

    snr_values = list(
        range(
            config["dataset"]["snr_range"][0],
            config["dataset"]["snr_range"][1] + 1,
            config["dataset"]["snr_step"],
        )
    )

    samples = {
        snr: {
            "x": [],
            "y": [],
        }
        for snr in snr_values
    }

    remaining = set(snr_values)

    for batch in test_loader:

        # ----------------------------------------------------
        # data_loader.py returns:
        # signal, label, snr
        # ----------------------------------------------------

        X, y, snr = batch

        for i in range(X.shape[0]):

            current_snr = int(snr[i].item())

            if current_snr not in remaining:
                continue

            if len(samples[current_snr]["x"]) >= SAMPLES_PER_SNR:
                continue

            samples[current_snr]["x"].append(
                X[i].clone()
            )

            samples[current_snr]["y"].append(
                y[i].clone()
            )

            if (
                len(samples[current_snr]["x"])
                >= SAMPLES_PER_SNR
            ):
                remaining.discard(current_snr)

        if not remaining:
            break

    # --------------------------------------------------------
    # Convert lists to tensors
    # --------------------------------------------------------

    result = {}

    for snr in snr_values:

        if len(samples[snr]["x"]) < SAMPLES_PER_SNR:
            raise RuntimeError(
                f"Could not collect {SAMPLES_PER_SNR} samples "
                f"for SNR {snr} dB. "
                f"Only collected {len(samples[snr]['x'])}."
            )

        result[snr] = {
            "x": torch.stack(
                samples[snr]["x"]
            ),
            "y": torch.stack(
                samples[snr]["y"]
            ).long(),
        }

        print(
            f"SNR {snr:>3} dB : "
            f"{len(result[snr]['x'])} samples"
        )

    print()
    print(f"✓ All {len(snr_values)} SNR levels collected")

    return result


# ============================================================
# ACCURACY
# ============================================================

@torch.no_grad()
def model_accuracy(model, X, y):
    """
    Calculate classification accuracy.
    """

    model.eval()

    logits = model(X)

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    correct = (
        predictions == y
    ).sum().item()

    total = y.numel()

    return (
        100.0 * correct / total
        if total > 0
        else 0.0
    )


# ============================================================
# BLACK-BOX EVALUATION
# ============================================================

def evaluate_blackbox(
    robust_model,
    surrogate_model,
    samples_by_snr,
):
    """
    Evaluate black-box surrogate-transfer attack.

    The target model is never differentiated.

    The existing black_box_attack function is responsible for:
        1. Querying the surrogate for labels.
        2. Generating surrogate-gradient adversarial examples.
        3. Returning adversarial signals.

    Ground-truth labels are only used for measuring final accuracy
    and attack success rate.
    """

    print()
    print("=" * 70)
    print("ROBUST MODEL BLACK-BOX SURROGATE TRANSFER")
    print("=" * 70)

    results = []

    for snr in sorted(samples_by_snr):

        X = samples_by_snr[snr]["x"].to(
            DEVICE,
            non_blocking=True,
        )

        y = samples_by_snr[snr]["y"].to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Clean target accuracy
        # ----------------------------------------------------

        clean_accuracy = model_accuracy(
            robust_model,
            X,
            y,
        )

        # ----------------------------------------------------
        # Generate black-box adversarial examples
        # ----------------------------------------------------

        adv_X = black_box_attack(
            robust_model,
            X,
            y,
            surrogate_model,
            EPSILON,
            steps=STEPS,
        )

        # ----------------------------------------------------
        # Adversarial target accuracy
        # ----------------------------------------------------

        adv_accuracy = model_accuracy(
            robust_model,
            adv_X,
            y,
        )

        # ----------------------------------------------------
        # Accuracy drop
        # ----------------------------------------------------

        drop = (
            clean_accuracy
            - adv_accuracy
        )

        # ----------------------------------------------------
        # Attack Success Rate
        #
        # ASR is measured only among samples that the target
        # classified correctly before the attack.
        # ----------------------------------------------------

        with torch.no_grad():

            clean_predictions = torch.argmax(
                robust_model(X),
                dim=1,
            )

            adv_predictions = torch.argmax(
                robust_model(adv_X),
                dim=1,
            )

        originally_correct = (
            clean_predictions == y
        )

        successful_attacks = (
            originally_correct
            & (
                adv_predictions != y
            )
        )

        num_originally_correct = (
            originally_correct.sum().item()
        )

        num_successful_attacks = (
            successful_attacks.sum().item()
        )

        if num_originally_correct > 0:
            attack_success_rate = (
                100.0
                * num_successful_attacks
                / num_originally_correct
            )
        else:
            attack_success_rate = 0.0

        # ----------------------------------------------------
        # Perturbation metrics
        # ----------------------------------------------------

        perturbation = (
            adv_X - X
        ).detach()

        flat_perturbation = perturbation.view(
            perturbation.shape[0],
            -1,
        )

        linf_per_sample = (
            flat_perturbation
            .abs()
            .max(dim=1)
            .values
        )

        l2_per_sample = torch.norm(
            flat_perturbation,
            p=2,
            dim=1,
        )

        mean_linf = (
            linf_per_sample
            .mean()
            .item()
        )

        max_linf = (
            linf_per_sample
            .max()
            .item()
        )

        mean_l2 = (
            l2_per_sample
            .mean()
            .item()
        )

        max_l2 = (
            l2_per_sample
            .max()
            .item()
        )

        row = {
            "snr": snr,
            "clean_accuracy": round(
                clean_accuracy,
                4,
            ),
            "blackbox_accuracy": round(
                adv_accuracy,
                4,
            ),
            "accuracy_drop": round(
                drop,
                4,
            ),
            "attack_success_rate": round(
                attack_success_rate,
                4,
            ),
            "mean_linf": round(
                mean_linf,
                6,
            ),
            "max_linf": round(
                max_linf,
                6,
            ),
            "mean_l2": round(
                mean_l2,
                6,
            ),
            "max_l2": round(
                max_l2,
                6,
            ),
        }

        results.append(row)

        print(
            f"SNR {snr:>3} dB | "
            f"Clean: {clean_accuracy:6.2f}% | "
            f"Black-box: {adv_accuracy:6.2f}% | "
            f"Drop: {drop:6.2f}% | "
            f"ASR: {attack_success_rate:6.2f}%"
        )

        # Release per-SNR GPU memory.
        del X
        del y
        del adv_X

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):
    """Save CSV and a small metadata JSON."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    # Ensure SNR ordering.
    df = df.sort_values(
        "snr"
    ).reset_index(drop=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"✓ Saved: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_rows = 26

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"but generated {len(df)}."
        )

    expected_snr = list(
        range(-20, 31, 2)
    )

    actual_snr = df["snr"].tolist()

    if actual_snr != expected_snr:
        raise RuntimeError(
            "SNR validation failed.\n"
            f"Expected: {expected_snr}\n"
            f"Actual:   {actual_snr}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "evaluation": "robust_blackbox_surrogate_transfer",
        "target_model": "robust",
        "target_checkpoint": str(
            ROBUST_CHECKPOINT
        ),
        "surrogate_checkpoint": str(
            SURROGATE_CHECKPOINT
        ),
        "dataset": "RadioML2018.01A",
        "samples_per_snr": SAMPLES_PER_SNR,
        "epsilon": EPSILON,
        "steps": STEPS,
        "num_snr_levels": len(df),
        "snr_range": [
            -20,
            30,
        ],
        "snr_step": 2,
        "query_budget": 5000,
        "target_retrained": False,
        "surrogate_retrained": False,
    }

    metadata_path = (
        OUTPUT_DIR
        / "blackbox_results_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"✓ Saved metadata: {metadata_path}"
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMCShield - Robust Black-Box Attack Evaluation")
    print("=" * 70)

    print()
    print(
        f"Dataset       : RadioML2018.01A"
    )
    print(
        f"Device        : {DEVICE}"
    )
    print(
        f"Samples/SNR   : {SAMPLES_PER_SNR}"
    )
    print(
        f"Black-box eps : {EPSILON}"
    )
    print(
        f"Attack steps  : {STEPS}"
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------

    config = load_config()

    # --------------------------------------------------------
    # Check checkpoints
    # --------------------------------------------------------

    if not ROBUST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Robust checkpoint not found:\n"
            f"{ROBUST_CHECKPOINT}"
        )

    if not SURROGATE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Surrogate checkpoint not found:\n"
            f"{SURROGATE_CHECKPOINT}"
        )

    print()
    print("=" * 70)
    print("LOADING ALREADY-TRAINED MODELS")
    print("=" * 70)

    (
        robust_model,
        surrogate_model,
        robust_checkpoint,
        surrogate_checkpoint,
    ) = build_models(config)

    print(
        "✓ Robust checkpoint loaded"
    )

    if "epoch" in robust_checkpoint:
        print(
            f"  Epoch      : "
            f"{robust_checkpoint['epoch']}"
        )

    if "val_accuracy" in robust_checkpoint:
        print(
            f"  Validation : "
            f"{robust_checkpoint['val_accuracy'] * 100:.2f}%"
        )

    print(
        "✓ Surrogate checkpoint loaded"
    )

    if "epoch" in surrogate_checkpoint:
        print(
            f"  Epoch      : "
            f"{surrogate_checkpoint['epoch']}"
        )

    if "val_accuracy" in surrogate_checkpoint:
        print(
            f"  Validation : "
            f"{surrogate_checkpoint['val_accuracy'] * 100:.2f}%"
        )

    if "query_budget" in surrogate_checkpoint:
        print(
            f"  Query budget : "
            f"{surrogate_checkpoint['query_budget']}"
        )

    # --------------------------------------------------------
    # Collect samples
    # --------------------------------------------------------

    samples_by_snr = collect_samples_by_snr(
        config
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate_blackbox(
        robust_model,
        surrogate_model,
        samples_by_snr,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df = save_results(
        results
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROBUST BLACK-BOX EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"✓ Rows generated : {len(df)}"
    )

    print(
        f"✓ SNR levels     : {df['snr'].nunique()}"
    )

    print(
        f"✓ Mean clean acc : "
        f"{df['clean_accuracy'].mean():.2f}%"
    )

    print(
        f"✓ Mean black-box : "
        f"{df['blackbox_accuracy'].mean():.2f}%"
    )

    print(
        f"✓ Mean ASR       : "
        f"{df['attack_success_rate'].mean():.2f}%"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()