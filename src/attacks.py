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

Patch notes (attack-correctness fixes):
    1. Attacks no longer leave the caller's model in eval mode. Robust
       training calls pgd_attack() mid-epoch; previously that silently
       switched the model to eval mode, so BatchNorm never updated and
       Dropout was off for the rest of training.
    2. C&W is implemented here in signal space. torchattacks.CW clamps
       inputs/outputs to [0, 1], which destroys signed RF/IQ data.
    3. The black-box surrogate keeps BatchNorm in eval mode (running
       statistics are never modified by an attack) and all of its
       parameters/flags are restored afterwards.
"""

import functools
from contextlib import contextmanager
from typing import Optional
import torch
import torch.nn.functional as F


# ============================================================
# Helpers
# ============================================================

def _prepare(model, x, y):
    model.eval()

    x = x.detach()
    y = y.detach().long()

    return x, y


@contextmanager
def _eval_mode(model):
    """
    Put `model` in eval mode for the duration of an attack, then restore
    the training flag of EVERY submodule exactly as it was.
    """
    saved = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in saved:
            module.training = was_training


def _restores_mode(fn):
    """Decorator: run an attack(model, ...) without changing model.training."""
    @functools.wraps(fn)
    def wrapper(model, *args, **kwargs):
        with _eval_mode(model):
            return fn(model, *args, **kwargs)
    return wrapper


@contextmanager
def _surrogate_attack_mode(surrogate):
    """
    Prepare a frozen surrogate for gradient-based attack generation.

    - Whole network in eval mode: BatchNorm uses running statistics and
      does NOT update them; Dropout is inactive.
    - Recurrent layers are switched back to train mode with their internal
      dropout set to 0.0, because cuDNN RNN backward() requires train mode.
      With dropout 0.0 this is still deterministic.
    - Parameters are frozen for the attack and restored afterwards.
    Everything is restored on exit, even if the attack raises.
    """
    saved_flags = [(m, m.training) for m in surrogate.modules()]
    saved_req = [(p, p.requires_grad) for p in surrogate.parameters()]
    rnns = [m for m in surrogate.modules() if isinstance(m, torch.nn.RNNBase)]
    saved_rnn_dropout = [(m, m.dropout) for m in rnns]

    surrogate.eval()
    for rnn in rnns:
        rnn.train()
        rnn.dropout = 0.0
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for module, was_training in saved_flags:
            module.training = was_training
        for parameter, flag in saved_req:
            parameter.requires_grad_(flag)
        for rnn, p in saved_rnn_dropout:
            rnn.dropout = p


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

@_restores_mode
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

@_restores_mode
def pgd_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int = 10,
    alpha: Optional[float] = None,
    random_start: bool = True,
) -> torch.Tensor:
    """
    RF-domain Projected Gradient Descent.

    No [0,1] clipping is performed.

    Random initialization occurs inside the epsilon ball.

    alpha (step size) defaults to 2.5 * epsilon / steps (Madry et al.).
    The earlier default, epsilon / steps, combined with the random start,
    could not reach the epsilon boundary for many coordinates, which made
    10-step PGD WEAKER than single-step FGSM on this project's models.
    """

    x, y = _prepare(model, x, y)

    # Random point inside the L-inf epsilon ball (or start exactly at x).
    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        adv_x = (x + delta).detach()
    else:
        adv_x = x.clone().detach()

    if alpha is None:
        alpha = 2.5 * epsilon / max(steps, 1)

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

@_restores_mode
def mim_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int = 10,
    alpha: Optional[float] = None,
    random_start: bool = False,
) -> torch.Tensor:
    """
    RF-domain Momentum Iterative Method.

    Uses normalized gradients with momentum.

    Momentum Iterative Method (Dong et al. 2018) with sign steps.

    alpha (step size) defaults to 2.5 * epsilon / steps, the same ratio the
    torchattacks defaults use (epsilon/4 for 10 steps). With epsilon / steps
    and momentum signs that keep flipping, most coordinates never reached the
    epsilon boundary and MIM came out WEAKER than single-step FGSM on this
    project's baseline (6.5% vs 4.8% accuracy). Pass alpha=epsilon/steps for
    the textbook step, and random_start=True for a random initial point.
    """

    x, y = _prepare(model, x, y)

    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        adv_x = (x + delta).detach()
    else:
        adv_x = x.clone().detach()

    if alpha is None:
        alpha = 2.5 * epsilon / max(steps, 1)

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

@_restores_mode
def cw_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 50,
    lr: float = 0.01,
    bound: Optional[float] = None,
) -> torch.Tensor:
    """
    Carlini & Wagner L2 attack for signed RF/IQ signals (B, 2, 1024).

    Implemented in signal space on purpose: torchattacks.CW maps inputs
    through a [0, 1] tanh box and clamps, which corrupts signed IQ data
    (the returned "adversarial" signal was not a small perturbation of x).

    The perturbation is parametrised as  x_adv = bound * tanh(w)  so that
    x_adv stays inside [-bound, +bound]; `bound` only needs to exceed
    max|x|. By default it is 1.5 * max|x| of the batch; pass a fixed value
    (e.g. 1.5 * max|X_train|) for results that do not depend on batch
    composition.

    Minimises  ||x_adv - x||_2^2 + c * max(real_logit - other_logit + kappa, 0)
    with Adam. For each sample the lowest-L2 adversarial example found is
    returned; samples where no adversarial example was found are returned
    unchanged (same convention as torchattacks).

    No epsilon / L-infinity projection is applied, by design.
    """

    x, y = _prepare(model, x, y)

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

    if bound is None:
        bound = float(x.abs().max().item()) * 1.5 + 1e-8

    w = torch.atanh(
        torch.clamp(x / bound, -1 + 1e-6, 1 - 1e-6)
    ).detach().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)

    best_adv = x.clone()
    best_l2 = torch.full(
        (x.shape[0],), float("inf"), device=x.device
    )

    for _ in range(steps):

        adv = bound * torch.tanh(w)
        logits = model(adv)

        l2 = ((adv - x) ** 2).sum(dim=(1, 2))

        true_mask = F.one_hot(y, logits.shape[1]).bool()
        real = logits[true_mask]
        other = logits.masked_fill(true_mask, float("-inf")).max(dim=1).values
        margin = torch.clamp(real - other + kappa, min=0.0)

        loss = (l2 + c * margin).sum()

        # Gradient w.r.t. w only: model parameter .grad is left untouched.
        w.grad = torch.autograd.grad(loss, w)[0]
        optimizer.step()

        with torch.no_grad():
            success = logits.argmax(dim=1) != y
            improved = success & (l2 < best_l2)
            best_l2 = torch.where(improved, l2, best_l2)
            best_adv[improved] = adv.detach()[improved]

    return best_adv.detach()


# ============================================================
# BLACK-BOX TRANSFER
# ============================================================

def black_box_attack(
    target_model,
    x,
    y,
    surrogate_model,
    epsilon,
    steps=10,
):
    """
    Strict query-only black-box transfer attack.

    Attack generation uses ONLY the surrogate model. The target model is
    never differentiated through, and ground-truth y is NOT used to
    generate the perturbation (it is kept only for API compatibility).

    Surrogate handling (see _surrogate_attack_mode): BatchNorm stays in
    eval mode so the surrogate's running statistics are never modified by
    an attack, parameters are frozen only for the duration of the call,
    and every flag is restored afterwards.
    """

    with _eval_mode(target_model), _surrogate_attack_mode(surrogate_model):

        x = x.detach()

        # Labels come from the surrogate's own predictions, never from y.
        with torch.no_grad():
            surrogate_labels = torch.argmax(
                surrogate_model(x),
                dim=1,
            )

        # Random initialization inside the epsilon L-infinity ball.
        delta = torch.empty_like(x).uniform_(
            -epsilon,
            epsilon,
        )

        adv_x = (x + delta).detach()

        alpha = 2.5 * epsilon / max(steps, 1)

        # PGD on the SURROGATE only.
        for _ in range(steps):

            adv_x.requires_grad_(True)

            logits = surrogate_model(adv_x)

            loss = F.cross_entropy(
                logits,
                surrogate_labels,
            )

            gradient = torch.autograd.grad(loss, adv_x)[0]

            adv_x = (
                adv_x.detach()
                + alpha * gradient.sign()
            )

            # No [0,1] clipping because these are RF/IQ signals.
            adv_x = _project_linf(
                adv_x,
                x,
                epsilon,
            ).detach()

    return adv_x.detach()


def blackbox_transfer_attack(
    surrogate_model,
    target_model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int = 10,
) -> torch.Tensor:
    """
    Kept for backward compatibility (used by attack_smoke_test.py).

    Delegates to black_box_attack() so there is a single black-box
    implementation. The previous version used ground-truth labels as the
    loss target and left the surrogate in train mode with its Dropout
    disabled and parameters frozen.
    """
    return black_box_attack(
        target_model, x, y, surrogate_model, epsilon, steps
    )


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