import torch
from src.model_robust import RobustCNN
from src.data_loader import create_dataloaders


CHECKPOINT = r"checkpoints\robust_best.pt"
BATCH_SIZE = 256


def main():

    print("=" * 70)
    print("AMCShield - Robust CNN Clean Test Evaluation")
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

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"Validation accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------

    print("\nCreating test DataLoader...")

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(
        f"Test samples: {len(test_loader.dataset):,}"
    )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    correct = 0
    total = 0

    print("\nEvaluating clean test set...")

    with torch.no_grad():

        for signals, labels, _ in test_loader:

            signals = signals.float().to(device)
            labels = labels.long().to(device)

            logits = model(signals)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("CLEAN TEST RESULTS")
    print("=" * 70)

    print(f"Correct predictions : {correct:,}")
    print(f"Total samples       : {total:,}")
    print(f"Test accuracy       : {accuracy * 100:.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()