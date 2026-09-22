import torch
import numpy as np

from src.model_robust import RobustCNN
from src.data_loader import create_dataloaders
from src.attacks import pgd_attack


CHECKPOINT = r"checkpoints\robust_best.pt"

BATCH_SIZE = 128
SAMPLES_PER_SNR = 256

EPSILON = 0.02
PGD_STEPS = 10


def main():

    print("=" * 70)
    print("AMCShield - PGD-10 Robustness by SNR")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice          : {device}")
    print(f"Samples / SNR   : {SAMPLES_PER_SNR}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"PGD epsilon     : {EPSILON}")
    print(f"PGD steps       : {PGD_STEPS}")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = RobustCNN(num_classes=24)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("\n✓ Robust checkpoint loaded")

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------

    _, _, test_loader = create_dataloaders(
        batch_size=1024,
        num_workers=0,
    )

    dataset = test_loader.dataset

    print(
        f"✓ Test dataset loaded: "
        f"{len(dataset):,} samples"
    )

    # --------------------------------------------------
    # Collect representative random indices by SNR
    # --------------------------------------------------

    print("\nCollecting representative samples by SNR...")

    # Reproducible random generator
    rng = np.random.default_rng(42)

    snr_indices = {}

    # First collect ALL indices for each SNR.
    all_snr_indices = {}

    for index in range(len(dataset)):

        _, _, snr = dataset[index]

        snr_value = int(snr.item())

        if snr_value not in all_snr_indices:
            all_snr_indices[snr_value] = []

        all_snr_indices[snr_value].append(index)

    # Randomly sample from each SNR bucket.
    for snr_value in sorted(all_snr_indices):

        available = np.array(
            all_snr_indices[snr_value],
            dtype=np.int64
        )

        sample_count = min(
            SAMPLES_PER_SNR,
            len(available)
        )

        selected = rng.choice(
            available,
            size=sample_count,
            replace=False
        )

        snr_indices[snr_value] = selected.tolist()

    print(
        f"✓ SNR buckets collected: {len(snr_indices)}"
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results = {}

    print("\nRunning PGD-10 evaluation...")
    print()

    for snr_value in sorted(snr_indices):

        indices = snr_indices[snr_value]

        clean_correct = 0
        pgd_correct = 0
        total = 0

        # ----------------------------------------------
        # Evaluate this SNR bucket
        # ----------------------------------------------

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

            for idx in batch_indices:

                x, y, _ = dataset[idx]

                signals.append(x)
                labels.append(y)

            x = torch.stack(
                signals
            ).float().to(device)

            y = torch.stack(
                labels
            ).long().to(device)

            # ------------------------------------------
            # Clean prediction
            # ------------------------------------------

            with torch.no_grad():

                clean_logits = model(x)

                clean_pred = torch.argmax(
                    clean_logits,
                    dim=1,
                )

            clean_correct += (
                clean_pred == y
            ).sum().item()

            # ------------------------------------------
            # PGD attack
            # ------------------------------------------

            x_adv = pgd_attack(
                model=model,
                x=x,
                y=y,
                epsilon=EPSILON,
                steps=PGD_STEPS,
            )

            # ------------------------------------------
            # Adversarial prediction
            # ------------------------------------------

            with torch.no_grad():

                adv_logits = model(x_adv)

                adv_pred = torch.argmax(
                    adv_logits,
                    dim=1,
                )

            pgd_correct += (
                adv_pred == y
            ).sum().item()

            total += y.size(0)

        clean_accuracy = (
            clean_correct / total
        )

        pgd_accuracy = (
            pgd_correct / total
        )

        accuracy_drop = (
            clean_accuracy - pgd_accuracy
        )

        # Attack success among clean-correct samples.
        attack_success = (
            accuracy_drop / clean_accuracy
            if clean_accuracy > 0
            else 0.0
        )

        results[snr_value] = {
            "clean_accuracy": clean_accuracy,
            "pgd_accuracy": pgd_accuracy,
            "accuracy_drop": accuracy_drop,
            "attack_success_rate": attack_success,
            "samples": total,
        }

        print(
            f"SNR {snr_value:>3} dB | "
            f"Clean: {clean_accuracy * 100:6.2f}% | "
            f"PGD: {pgd_accuracy * 100:6.2f}% | "
            f"Drop: {accuracy_drop * 100:6.2f}% | "
            f"ASR: {attack_success * 100:6.2f}%"
        )

    # --------------------------------------------------
    # Final table
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL PGD-10 SNR RESULTS")
    print("=" * 70)

    print(
        f"{'SNR':>6} | "
        f"{'Clean':>8} | "
        f"{'PGD-10':>8} | "
        f"{'Drop':>8} | "
        f"{'ASR':>8}"
    )

    print("-" * 70)

    for snr_value in sorted(results):

        r = results[snr_value]

        print(
            f"{snr_value:>6} | "
            f"{r['clean_accuracy'] * 100:>7.2f}% | "
            f"{r['pgd_accuracy'] * 100:>7.2f}% | "
            f"{r['accuracy_drop'] * 100:>7.2f}% | "
            f"{r['attack_success_rate'] * 100:>7.2f}%"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
