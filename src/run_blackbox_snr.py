"""
run_blackbox_snr.py

AMCShield - Black-Box Surrogate Transfer Evaluation

Attack:
    PGD generated against surrogate

Target:
    RobustCNN

Surrogate:
    Trained using baseline-model query responses

Results:
    results/blackbox_results.csv

Schema:
    snr,
    clean_accuracy,
    blackbox_accuracy,
    drop,
    attack_success_rate
"""

from pathlib import Path

import csv
import numpy as np
import torch
import yaml

from src.data_loader import create_dataloaders
from src.model_baseline import BaselineCNN
from src.model_robust import RobustCNN
from src.model_surrogate import SurrogateModel
from src.attacks import black_box_attack


# ============================================================
# Paths
# ============================================================

BASELINE_CHECKPOINT = (
    "checkpoints/baseline_best.pt"
)

ROBUST_CHECKPOINT = (
    "checkpoints/robust_best.pt"
)

SURROGATE_CHECKPOINT = (
    "checkpoints/surrogate_best.pt"
)

CONFIG_PATH = (
    "configs/config.yaml"
)

OUTPUT_PATH = (
    "results/blackbox_results.csv"
)


# ============================================================
# Evaluation settings
# ============================================================

BATCH_SIZE = 128

SAMPLES_PER_SNR = 256


# ============================================================
# Configuration
# ============================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


# ============================================================
# Checkpoint loading
# ============================================================

def load_model(
    model,
    checkpoint_path,
    device,
    name,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)

    model.eval()

    print(
        f"✓ {name} checkpoint loaded"
    )

    if "epoch" in checkpoint:

        print(
            f"  Epoch: "
            f"{checkpoint['epoch']}"
        )

    if "val_accuracy" in checkpoint:

        print(
            f"  Validation accuracy: "
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

    return model


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "AMCShield - Black-Box Surrogate Transfer by SNR"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    num_classes = int(
        config["dataset"].get(
            "num_classes",
            24,
        )
    )

    epsilon_config = config.get(
        "epsilon",
        {},
    )

    if isinstance(
        epsilon_config,
        dict,
    ):

        epsilon = float(
            epsilon_config.get(
                "pgd",
                0.02,
            )
        )

    else:

        epsilon = float(
            epsilon_config
        )

    steps = int(
        config.get(
            "pgd_steps",
            10,
        )
    )

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    print()
    print(
        f"Dataset        : "
        f"{config['dataset']['name']}"
    )

    print(
        f"Device         : "
        f"{device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU            : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Samples / SNR  : "
        f"{SAMPLES_PER_SNR}"
    )

    print(
        f"Batch size     : "
        f"{BATCH_SIZE}"
    )

    print(
        f"PGD epsilon    : "
        f"{epsilon}"
    )

    print(
        f"PGD steps      : "
        f"{steps}"
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    # --------------------------------------------------------
    # Load baseline
    #
    # The baseline is needed only because the surrogate was
    # trained from it. It is NOT used to generate gradients
    # during the black-box attack.
    # --------------------------------------------------------

    print()
    print(
        "Loading baseline..."
    )

    baseline = load_model(
        BaselineCNN(
            num_classes=num_classes
        ),
        BASELINE_CHECKPOINT,
        device,
        "Baseline",
    )

    # --------------------------------------------------------
    # Load robust target
    # --------------------------------------------------------

    print()
    print(
        "Loading robust target..."
    )

    robust = load_model(
        RobustCNN(
            num_classes=num_classes
        ),
        ROBUST_CHECKPOINT,
        device,
        "Robust",
    )

    # --------------------------------------------------------
    # Load surrogate
    # --------------------------------------------------------

    print()
    print(
        "Loading surrogate..."
    )

    surrogate = load_model(
        SurrogateModel(
            num_classes=num_classes
        ),
        SURROGATE_CHECKPOINT,
        device,
        "Surrogate",
    )

    # --------------------------------------------------------
    # Freeze surrogate parameters.
    #
    # We need gradients with respect to INPUT,
    # but never update surrogate weights.
    # --------------------------------------------------------

    for parameter in surrogate.parameters():

        parameter.requires_grad = False

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print()
    print(
        "Creating test DataLoader..."
    )

    _, _, test_loader = create_dataloaders(
        batch_size=1024,
        num_workers=0,
    )

    dataset = test_loader.dataset

    print(
        f"✓ Test samples: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # Build SNR buckets
    # --------------------------------------------------------

    print()
    print(
        "Collecting SNR buckets..."
    )

    rng = np.random.default_rng(
        seed
    )

    all_snr_indices = {}

    for index in range(
        len(dataset)
    ):

        _, _, snr = dataset[index]

        snr_value = int(
            snr.item()
        )

        if snr_value not in all_snr_indices:

            all_snr_indices[
                snr_value
            ] = []

        all_snr_indices[
            snr_value
        ].append(index)

    snr_indices = {}

    for snr_value in sorted(
        all_snr_indices
    ):

        available = np.asarray(
            all_snr_indices[
                snr_value
            ],
            dtype=np.int64,
        )

        sample_count = min(
            SAMPLES_PER_SNR,
            len(available),
        )

        selected = rng.choice(
            available,
            size=sample_count,
            replace=False,
        )

        snr_indices[
            snr_value
        ] = selected.tolist()

    print(
        f"✓ SNR buckets: "
        f"{len(snr_indices)}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    print()
    print(
        "=" * 70
    )
    print(
        "RUNNING BLACK-BOX EVALUATION"
    )
    print(
        "=" * 70
    )

    for snr_value in sorted(
        snr_indices
    ):

        indices = snr_indices[
            snr_value
        ]

        clean_correct = 0
        blackbox_correct = 0
        total = 0

        # ----------------------------------------------------
        # Batch evaluation
        # ----------------------------------------------------

        for start in range(
            0,
            len(indices),
            BATCH_SIZE,
        ):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            signals = []
            labels = []

            for index in batch_indices:

                x, y, _ = dataset[index]

                signals.append(x)
                labels.append(y)

            x = torch.stack(
                signals
            ).float().to(
                device,
                non_blocking=True,
            )

            y = torch.stack(
                labels
            ).long().to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Clean target prediction
            # ------------------------------------------------

            with torch.no_grad():

                clean_logits = robust(x)

                clean_pred = torch.argmax(
                    clean_logits,
                    dim=1,
                )

            clean_mask = (
                clean_pred == y
            )

            clean_correct += (
                clean_mask.sum().item()
            )

            # ------------------------------------------------
            # Black-box attack
            #
            # Ground-truth y is passed only because the
            # evaluation function needs it. The attack
            # itself uses surrogate predictions.
            # ------------------------------------------------

            x_adv = black_box_attack(
                target_model=robust,
                x=x,
                y=y,
                surrogate_model=surrogate,
                epsilon=epsilon,
                steps=steps,
            )

            # ------------------------------------------------
            # Target prediction on transferred examples
            # ------------------------------------------------

            with torch.no_grad():

                adv_logits = robust(
                    x_adv
                )

                adv_pred = torch.argmax(
                    adv_logits,
                    dim=1,
                )

            blackbox_correct += (
                adv_pred == y
            ).sum().item()

            total += y.size(0)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        clean_accuracy = (
            clean_correct
            / total
            if total > 0
            else 0.0
        )

        blackbox_accuracy = (
            blackbox_correct
            / total
            if total > 0
            else 0.0
        )

        drop = (
            clean_accuracy
            - blackbox_accuracy
        )

        attack_success_rate = (
            drop
            / clean_accuracy
            if clean_accuracy > 0
            else 0.0
        )

        results.append(
            {
                "snr": snr_value,
                "clean_accuracy": clean_accuracy,
                "blackbox_accuracy": blackbox_accuracy,
                "drop": drop,
                "attack_success_rate": (
                    attack_success_rate
                ),
            }
        )

        print(
            f"SNR {snr_value:>3} dB | "
            f"Clean: {clean_accuracy * 100:6.2f}% | "
            f"Black-box: {blackbox_accuracy * 100:6.2f}% | "
            f"Drop: {drop * 100:6.2f}% | "
            f"ASR: {attack_success_rate * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_path = Path(
        OUTPUT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "snr",
        "clean_accuracy",
        "blackbox_accuracy",
        "drop",
        "attack_success_rate",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    expected_snr_values = list(
        range(
            -20,
            31,
            2,
        )
    )

    actual_snr_values = [
        row["snr"]
        for row in results
    ]

    assert (
        actual_snr_values
        == expected_snr_values
    ), (
        "SNR coverage mismatch: "
        f"{actual_snr_values}"
    )

    assert len(results) == 26

    print()
    print("=" * 70)
    print(
        "BLACK-BOX EVALUATION COMPLETED"
    )
    print("=" * 70)

    print(
        f"Rows written : {len(results)}"
    )

    print(
        f"Output       : {OUTPUT_PATH}"
    )

    print(
        "✓ All 26 SNR values evaluated"
    )

    print(
        "✓ No [0,1] clipping used"
    )

    print(
        "✓ Perturbations generated on surrogate only"
    )


if __name__ == "__main__":
    main()