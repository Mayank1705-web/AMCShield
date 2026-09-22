import torch

from src.model_robust import RobustCNN
from src.data_loader import create_dataloaders
from src.attacks import pgd_attack


CHECKPOINT = r"checkpoints\robust_best.pt"

BATCH_SIZE = 128
MAX_SAMPLES = 4096

EPSILON = 0.02
PGD_STEPS = 10


def main():

    print("=" * 70)
    print("AMCShield - PGD-10 Robustness Evaluation")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice       : {device}")
    print(f"Samples      : {MAX_SAMPLES:,}")
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
    # Load test set
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
    # Evaluation
    # --------------------------------------------------

    clean_correct = 0
    pgd_correct = 0
    total = 0

    print("\nRunning PGD-10 attack...")

    for batch_idx, (x, y, snr) in enumerate(test_loader):

        if total >= MAX_SAMPLES:
            break

        remaining = MAX_SAMPLES - total

        x = x[:remaining].float().to(device)
        y = y[:remaining].long().to(device)

        # ----------------------------------------------
        # Clean prediction
        # ----------------------------------------------

        with torch.no_grad():

            clean_logits = model(x)

            clean_pred = torch.argmax(
                clean_logits,
                dim=1,
            )

        clean_correct += (
            clean_pred == y
        ).sum().item()

        # ----------------------------------------------
        # PGD attack
        # ----------------------------------------------

        x_adv = pgd_attack(
            model=model,
            x=x,
            y=y,
            epsilon=EPSILON,
            steps=PGD_STEPS,
        )

        # ----------------------------------------------
        # Adversarial prediction
        # ----------------------------------------------

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

        clean_acc = (
            clean_correct / total * 100
        )

        pgd_acc = (
            pgd_correct / total * 100
        )

        print(
            f"Batch {batch_idx + 1:3d} | "
            f"Samples: {total:5d}/{MAX_SAMPLES} | "
            f"Clean: {clean_acc:6.2f}% | "
            f"PGD-10: {pgd_acc:6.2f}%"
        )

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    clean_accuracy = (
        clean_correct / total
    )

    pgd_accuracy = (
        pgd_correct / total
    )

    accuracy_drop = (
        clean_accuracy - pgd_accuracy
    )

    print()
    print("=" * 70)
    print("PGD-10 RESULTS")
    print("=" * 70)

    print(
        f"Samples evaluated : {total:,}"
    )

    print(
        f"Clean accuracy    : "
        f"{clean_accuracy * 100:.2f}%"
    )

    print(
        f"PGD-10 accuracy   : "
        f"{pgd_accuracy * 100:.2f}%"
    )

    print(
        f"Accuracy drop     : "
        f"{accuracy_drop * 100:.2f} percentage points"
    )

    print(
        f"Attack epsilon    : {EPSILON}"
    )

    print(
        f"PGD steps         : {PGD_STEPS}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()