import torch

from src.model_robust import RobustCNN
from src.data_loader import create_dataloaders
from src.attacks import pgd_attack


CHECKPOINT = r"checkpoints\robust_best.pt"

BATCH_SIZE = 128
NUM_SAMPLES = 1024

EPSILON = 0.02
PGD_STEPS = 10


def main():

    print("=" * 70)
    print("AMCShield - PGD Robustness Pilot")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice       : {device}")
    print(f"Samples      : {NUM_SAMPLES}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"PGD epsilon  : {EPSILON}")
    print(f"PGD steps    : {PGD_STEPS}")

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
    # Load test data
    # --------------------------------------------------

    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(
        f"✓ Test dataset loaded: "
        f"{len(test_loader.dataset):,} samples"
    )

    # --------------------------------------------------
    # PGD evaluation
    # --------------------------------------------------

    clean_correct = 0
    adv_correct = 0
    total = 0

    print("\nRunning PGD-10 attack...")

    for batch_idx, (x, y, snr) in enumerate(test_loader):

        if total >= NUM_SAMPLES:
            break

        remaining = NUM_SAMPLES - total

        x = x[:remaining].float().to(device)
        y = y[:remaining].long().to(device)

        # ------------------------------
        # Clean prediction
        # ------------------------------

        with torch.no_grad():

            clean_logits = model(x)

            clean_pred = torch.argmax(
                clean_logits,
                dim=1,
            )

        clean_correct += (
            clean_pred == y
        ).sum().item()

        # ------------------------------
        # PGD attack
        # ------------------------------

        adv_x = pgd_attack(
            model=model,
            x=x,
            y=y,
            epsilon=EPSILON,
            steps=PGD_STEPS,
        )

        # ------------------------------
        # Adversarial prediction
        # ------------------------------

        with torch.no_grad():

            adv_logits = model(adv_x)

            adv_pred = torch.argmax(
                adv_logits,
                dim=1,
            )

        adv_correct += (
            adv_pred == y
        ).sum().item()

        total += y.size(0)

        print(
            f"Batch {batch_idx + 1:>3} | "
            f"Samples: {total:>5}/{NUM_SAMPLES} | "
            f"Clean: {clean_correct / total * 100:6.2f}% | "
            f"PGD: {adv_correct / total * 100:6.2f}%"
        )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    clean_accuracy = clean_correct / total
    adversarial_accuracy = adv_correct / total

    robustness_drop = (
        clean_accuracy - adversarial_accuracy
    )

    print()
    print("=" * 70)
    print("PGD PILOT RESULTS")
    print("=" * 70)

    print(
        f"Samples evaluated       : {total:,}"
    )

    print(
        f"Clean accuracy           : "
        f"{clean_accuracy * 100:.2f}%"
    )

    print(
        f"PGD-10 accuracy          : "
        f"{adversarial_accuracy * 100:.2f}%"
    )

    print(
        f"Accuracy drop            : "
        f"{robustness_drop * 100:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()