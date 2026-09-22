"""
attacks.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 3 (PGD, FGSM) -> Phase 5 (MIM, C&W, black-box)

RadioML 2018.01A:
    24 classes
    Input shape: (B, 2, 1024)

Important:
    RF/IQ signals are NOT image data and must NOT be clipped to [0, 1].

    All epsilon-bounded attacks explicitly enforce:

        ||x_adv - x||_inf <= epsilon
"""

from typing import Optional

import torch
import torchattacks


# ============================================================
# Helpers
# ============================================================

def _prepare(model, x, y):
    model.eval()

    x = x.detach()
    y = y.detach().long()

    return x, y


def _project_linf(
    adv_x: torch.Tensor,
    original_x: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Project adversarial examples back into the L-infinity
    epsilon ball around the original RF signal.

    IMPORTANT:
        No [0,1] clipping is performed.
    """

    delta = torch.clamp(
        adv_x - original_x,
        min=-epsilon,
        max=epsilon,
    )

    return original_x + delta


# ============================================================
# FGSM
# ============================================================

def fgsm_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    RF-domain FGSM.

    Computes:

        x_adv = x + epsilon * sign(gradient)

    and explicitly enforces the L-infinity constraint.
    """

    x, y = _prepare(model, x, y)

    x_adv = x.clone().detach()
    x_adv.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    logits = model(x_adv)
    loss = torch.nn.functional.cross_entropy(logits, y)

    loss.backward()

    gradient = x_adv.grad.detach()

    x_adv = x_adv.detach() + epsilon * gradient.sign()

    x_adv = _project_linf(
        x_adv,
        x,
        epsilon,
    )

    return x_adv.detach()


# ============================================================
# PGD
# ============================================================

def pgd_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int = 10,
) -> torch.Tensor:
    """
    RF-domain Projected Gradient Descent.

    No [0,1] clipping is performed.

    Random initialization occurs inside the epsilon ball.
    """

    x, y = _prepare(model, x, y)

    # Random point inside the L-inf epsilon ball.
    delta = torch.empty_like(x).uniform_(
        -epsilon,
        epsilon,
    )

    adv_x = (x + delta).detach()

    alpha = epsilon / max(steps, 1)

    for _ in range(steps):

        adv_x.requires_grad_(True)

        model.zero_grad(set_to_none=True)

        logits = model(adv_x)
        loss = torch.nn.functional.cross_entropy(
            logits,
            y,
        )

        loss.backward()

        gradient = adv_x.grad.detach()

        adv_x = adv_x.detach() + alpha * gradient.sign()

        adv_x = _project_linf(
            adv_x,
            x,
            epsilon,
        ).detach()

    return adv_x


# ============================================================
# MIM
# ============================================================

def mim_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int = 10,
) -> torch.Tensor:
    """
    RF-domain Momentum Iterative Method.

    Uses normalized gradients with momentum.
    """

    x, y = _prepare(model, x, y)

    delta = torch.empty_like(x).uniform_(
        -epsilon,
        epsilon,
    )

    adv_x = (x + delta).detach()

    alpha = epsilon / max(steps, 1)

    momentum = torch.zeros_like(x)

    for _ in range(steps):

        adv_x.requires_grad_(True)

        model.zero_grad(set_to_none=True)

        logits = model(adv_x)

        loss = torch.nn.functional.cross_entropy(
            logits,
            y,
        )

        loss.backward()

        gradient = adv_x.grad.detach()

        # Normalize gradient using mean absolute value.
        grad_norm = gradient.abs().mean(
            dim=(1, 2),
            keepdim=True,
        )

        gradient = gradient / (
            grad_norm + 1e-12
        )

        momentum = momentum + gradient

        adv_x = adv_x.detach() + (
            alpha * momentum.sign()
        )

        adv_x = _project_linf(
            adv_x,
            x,
            epsilon,
        ).detach()

    return adv_x


# ============================================================
# C&W
# ============================================================

def cw_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Carlini & Wagner attack for RadioML RF/IQ signals.

    Input:
        x -> RF/IQ tensor with shape (B, 2, 1024)
        y -> class labels with shape (B,)

    Important:
        - No subset selection is performed here.
        - No [0, 1] clipping is performed.
        - No artificial epsilon/L-infinity projection is applied.
        - The C&W optimizer is allowed to determine the perturbation.
        - The returned adversarial examples preserve the RF tensor
          shape (B, 2, 1024).
        - Returned tensor is detached from the computation graph.
    """

    x, y = _prepare(model, x, y)

    # Preserve the expected RF/IQ input shape.
    if x.ndim != 3:
        raise ValueError(
            f"Expected RF input with 3 dimensions "
            f"(B, 2, 1024), got {tuple(x.shape)}"
        )

    if x.shape[1] != 2 or x.shape[2] != 1024:
        raise ValueError(
            f"Expected RF input shape (B, 2, 1024), "
            f"got {tuple(x.shape)}"
        )

    attack = torchattacks.CW(
        model,
        c=1.0,
        kappa=0.0,
        steps=50,
        lr=0.01,
    )

    # C&W is intentionally NOT followed by _project_linf().
    # Unlike FGSM/PGD/MIM, this attack is not artificially
    # constrained to epsilon=0.02 here.
    adv_x = attack(x, y)

    # Ensure the attack did not alter the RF tensor shape.
    if adv_x.shape != x.shape:
        raise RuntimeError(
            f"C&W changed the RF tensor shape: "
            f"input={tuple(x.shape)}, "
            f"adversarial={tuple(adv_x.shape)}"
        )

    return adv_x.detach()

# ============================================================
# BLACK-BOX TRANSFER
# ============================================================

def blackbox_transfer_attack(
    surrogate_model,
    target_model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int = 10,
) -> torch.Tensor:
    """
    Black-box transfer attack.

    Gradients are computed ONLY through surrogate_model.

    target_model is NOT used to generate the attack.
    """

    surrogate_model.eval()
    target_model.eval()

    x = x.detach()
    y = y.detach().long()

    delta = torch.empty_like(x).uniform_(
        -epsilon,
        epsilon,
    )

    adv_x = (x + delta).detach()

    alpha = epsilon / max(steps, 1)

    for _ in range(steps):

        adv_x.requires_grad_(True)

        surrogate_model.zero_grad(
            set_to_none=True
        )

        logits = surrogate_model(adv_x)

        loss = torch.nn.functional.cross_entropy(
            logits,
            y,
        )

        loss.backward()

        gradient = adv_x.grad.detach()

        adv_x = adv_x.detach() + (
            alpha * gradient.sign()
        )

        adv_x = _project_linf(
            adv_x,
            x,
            epsilon,
        ).detach()

    return adv_x


# ============================================================
# Self test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AMCShield Attack Module")
    print("=" * 70)

    print()
    print("Dataset       : RadioML 2018.01A")
    print("Classes       : 24")
    print("Input shape   : (B, 2, 1024)")
    print("Default eps   : 0.02")

    print()
    print("Available attacks:")
    print("  ✓ RF-domain FGSM")
    print("  ✓ RF-domain PGD")
    print("  ✓ RF-domain MIM")
    print("  ✓ C&W")
    print("  ✓ Black-box surrogate transfer")

    print()
    print("RF attacks explicitly avoid [0,1] clipping.")
    print("Attack module syntax test passed.")

    print("=" * 70)