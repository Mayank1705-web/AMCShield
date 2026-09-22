import csv
from pathlib import Path

import torch

from src.model_robust import RobustCNN
from src.data_loader import create_dataloaders


CHECKPOINT = r"checkpoints\robust_best.pt"
BATCH_SIZE = 256

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "clean_results.csv"


def main():

    print("=" * 70)
    print("AMCShield - Robust CNN Clean Accuracy by SNR")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

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

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------

    print("\nLoading test data...")

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # --------------------------------------------------
    # SNR statistics
    # --------------------------------------------------

    correct_by_snr = {}
    total_by_snr = {}

    print("\nEvaluating clean accuracy by SNR...")

    with torch.no_grad():

        for batch_idx, (x, y, snr) in enumerate(test_loader):

            x = x.float().to(device)
            y = y.long().to(device)

            logits = model(x)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct = (
                predictions == y
            ).cpu()

            for i in range(len(y)):

                snr_value = int(snr[i].item())

                if snr_value not in correct_by_snr:
                    correct_by_snr[snr_value] = 0
                    total_by_snr[snr_value] = 0

                correct_by_snr[snr_value] += int(
                    correct[i].item()
                )

                total_by_snr[snr_value] += 1

            if (batch_idx + 1) % 100 == 0:

                processed = min(
                    (batch_idx + 1) * BATCH_SIZE,
                    len(test_loader.dataset),
                )

                print(
                    f"Processed: "
                    f"{processed:,}/"
                    f"{len(test_loader.dataset):,}"
                )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("CLEAN ACCURACY BY SNR")
    print("=" * 70)

    total_correct = 0
    total_samples = 0

    results = []

    for snr_value in sorted(total_by_snr):

        total = total_by_snr[snr_value]
        correct = correct_by_snr[snr_value]

        accuracy = correct / total

        total_correct += correct
        total_samples += total

        results.append({
            "snr": snr_value,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        })

        print(
            f"SNR {snr_value:>3} dB | "
            f"Correct: {correct:>6,} | "
            f"Total: {total:>6,} | "
            f"Accuracy: {accuracy * 100:>6.2f}%"
        )

    print("-" * 70)

    overall_accuracy = (
        total_correct / total_samples
    )

    print(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    print("=" * 70)

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr",
                "correct",
                "total",
                "accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved clean results to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()