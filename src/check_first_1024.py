import torch

from src.model_robust import RobustCNN
from src.data_loader import create_dataloaders


CHECKPOINT = r"checkpoints\robust_best.pt"
BATCH_SIZE = 128
NUM_SAMPLES = 1024


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("AMCShield - First 1024 Clean Sample Check")
    print("=" * 70)

    model = RobustCNN(num_classes=24)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y, snr in test_loader:

            if total >= NUM_SAMPLES:
                break

            remaining = NUM_SAMPLES - total

            x = x[:remaining].float().to(device)
            y = y[:remaining].long().to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            if total == remaining:
                break

    print()
    print(f"Samples : {total}")
    print(f"Correct : {correct}")
    print(f"Accuracy: {correct / total * 100:.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()