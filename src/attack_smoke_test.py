"""
attack_smoke_test.py

Small integration test for AMCShield attacks.

Tests:
    - FGSM
    - PGD
    - MIM
    - C&W
    - Black-box surrogate transfer

Uses only a tiny batch so no expensive full-dataset attack is performed.
"""

import torch

from src.data_loader import create_dataloaders
from src.model_baseline import BaselineCNN
from src.attacks import (
    fgsm_attack,
    pgd_attack,
    mim_attack,
    cw_attack,
    blackbox_transfer_attack,
)


BATCH_SIZE = 4
EPSILON = 0.02
STEPS = 2


def check_shape(original, adversarial, name):
    assert adversarial.shape == original.shape, (
        f"{name}: shape mismatch. "
        f"Expected {original.shape}, got {adversarial.shape}"
    )
    print(f"  ✓ {name} shape: {tuple(adversarial.shape)}")


def check_finite(adversarial, name):
    assert torch.isfinite(adversarial).all(), (
        f"{name}: NaN or Inf detected"
    )
    print(f"  ✓ {name} contains only finite values")


def check_perturbation(original, adversarial, name, epsilon):
    perturbation = (adversarial - original).abs()
    max_perturbation = perturbation.max().item()

    print(
        f"  ✓ {name} max perturbation: "
        f"{max_perturbation:.6f}"
    )

    # Small floating-point tolerance.
    assert max_perturbation <= epsilon + 1e-5, (
        f"{name}: perturbation exceeded epsilon. "
        f"max={max_perturbation}, epsilon={epsilon}"
    )


def main():
    print("=" * 70)
    print("AMCShield Attack Smoke Test")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\nCreating DataLoader...")

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    x, y, snr = next(iter(test_loader))

    x = x.to(device)
    y = y.to(device)

    print(f"Input shape : {tuple(x.shape)}")
    print(f"Labels shape: {tuple(y.shape)}")
    print(f"SNR shape   : {tuple(snr.shape)}")

    assert x.shape == (BATCH_SIZE, 2, 1024)

    # --------------------------------------------------------
    # Target model
    # --------------------------------------------------------

    print("\nCreating target model...")

    target_model = BaselineCNN(
        num_classes=24
    ).to(device)

    target_model.eval()

    print("  ✓ Target model created")

    # --------------------------------------------------------
    # FGSM
    # --------------------------------------------------------

    print("\n[1/5] Testing FGSM...")

    adv_fgsm = fgsm_attack(
        target_model,
        x,
        y,
        EPSILON,
    )

    check_shape(x, adv_fgsm, "FGSM")
    check_finite(adv_fgsm, "FGSM")
    check_perturbation(
        x,
        adv_fgsm,
        "FGSM",
        EPSILON,
    )

    # --------------------------------------------------------
    # PGD
    # --------------------------------------------------------

    print("\n[2/5] Testing PGD...")

    adv_pgd = pgd_attack(
        target_model,
        x,
        y,
        EPSILON,
        steps=STEPS,
    )

    check_shape(x, adv_pgd, "PGD")
    check_finite(adv_pgd, "PGD")
    check_perturbation(
        x,
        adv_pgd,
        "PGD",
        EPSILON,
    )

    # --------------------------------------------------------
    # MIM
    # --------------------------------------------------------

    print("\n[3/5] Testing MIM...")

    adv_mim = mim_attack(
        target_model,
        x,
        y,
        EPSILON,
        steps=STEPS,
    )

    check_shape(x, adv_mim, "MIM")
    check_finite(adv_mim, "MIM")
    check_perturbation(
        x,
        adv_mim,
        "MIM",
        EPSILON,
    )

    # --------------------------------------------------------
    # C&W
    # --------------------------------------------------------

    print("\n[4/5] Testing C&W...")
    print("  Using only 1 sample because C&W is expensive.")

    adv_cw = cw_attack(
        target_model,
        x[:1],
        y[:1],
        subset_size=1,
    )

    check_shape(
        x[:1],
        adv_cw,
        "C&W",
    )

    check_finite(
        adv_cw,
        "C&W",
    )

    # C&W does NOT use the same epsilon-bounded formulation
    # as FGSM/PGD/MIM, so we intentionally do not apply the
    # epsilon perturbation assertion here.

    # --------------------------------------------------------
    # Black-box transfer
    # --------------------------------------------------------

    print("\n[5/5] Testing black-box transfer...")

    surrogate_model = BaselineCNN(
        num_classes=24
    ).to(device)

    surrogate_model.eval()

    adv_blackbox = blackbox_transfer_attack(
        surrogate_model,
        target_model,
        x,
        y,
        EPSILON,
        steps=STEPS,
    )

    check_shape(
        x,
        adv_blackbox,
        "Black-box",
    )

    check_finite(
        adv_blackbox,
        "Black-box",
    )

    check_perturbation(
        x,
        adv_blackbox,
        "Black-box",
        EPSILON,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ATTACK SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()