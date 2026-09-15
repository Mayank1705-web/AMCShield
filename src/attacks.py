"""
attacks.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 3 (PGD, FGSM) -> Phase 5 (MIM, C&W, black-box)

Responsibility: attack implementations via torchattacks, plus the
black-box surrogate-transfer logic.

CRITICAL DISTINCTION (see architecture.md Section 4, integration point #3):
- White-box attacks (FGSM, PGD, MIM, C&W): gradients flow through the
  REAL target model. Model must be in eval() mode, requires_grad=True on
  the INPUT tensor, never on model weights.
- Black-box attack: gradients flow ONLY through the surrogate model.
  Only the resulting adversarial examples are transferred to attack the
  real baseline/robust models. Mixing these up silently invalidates the
  black-box claim -- this is the easiest place to introduce a hard-to-spot bug.

C&W is expensive -- always respect config['cw_eval_subset_size'], never
run it on the full test set by default (see CONSTRAINTS.md).
"""

import torch
import torchattacks
from typing import Optional


def fgsm_attack(model, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    """White-box FGSM via torchattacks. TODO (Person B)."""
    raise NotImplementedError


def pgd_attack(model, x: torch.Tensor, y: torch.Tensor, epsilon: float,
                steps: int = 10) -> torch.Tensor:
    """White-box PGD via torchattacks. Used both for robust training and evaluation.
    TODO (Person B)."""
    raise NotImplementedError


def mim_attack(model, x: torch.Tensor, y: torch.Tensor, epsilon: float,
                steps: int = 10) -> torch.Tensor:
    """White-box MIM via torchattacks. TODO (Person B, Phase 5)."""
    raise NotImplementedError


def cw_attack(model, x: torch.Tensor, y: torch.Tensor,
               subset_size: Optional[int] = None) -> torch.Tensor:
    """
    White-box C&W via torchattacks. EXPENSIVE -- respect subset_size
    (from config['cw_eval_subset_size']). TODO (Person B, Phase 5).
    """
    raise NotImplementedError


def blackbox_transfer_attack(surrogate_model, target_model,
                               x: torch.Tensor, y: torch.Tensor,
                               epsilon: float, steps: int = 10) -> torch.Tensor:
    """
    Craft PGD adversarial examples ON THE SURROGATE, then return them
    (unmodified) for evaluation against target_model. Gradients must
    NEVER flow through target_model here.

    TODO (Person B, Phase 5). See FLOW.md Section 2 for the full flow.
    """
    raise NotImplementedError
