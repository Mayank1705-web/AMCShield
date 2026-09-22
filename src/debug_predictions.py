import torch

from src.model_robust import RobustCNN
from src.data_loader import create_dataloaders


CHECKPOINT = r"checkpoints\robust_best.pt"


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = RobustCNN(num_classes=24)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader = create_dataloaders(
        batch_size=16,
        num_workers=0,
    )

    x, y, snr = next(iter(test_loader))

    x = x.float().to(device)
    y = y.long().to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

    print("=" * 70)
    print("AMCShield - Prediction Debug")
    print("=" * 70)

    for i in range(min(16, len(y))):
        print(
            f"Sample {i:2d} | "
            f"True: {y[i].item():2d} | "
            f"Pred: {pred[i].item():2d} | "
            f"SNR: {snr[i].item():3d}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()