"""
model_surrogate.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 5 (Week 7)

Responsibility: a model architecturally DIFFERENT from BaselineCNN, trained
ONLY on (input, predicted-label) query pairs collected from the baseline --
never on ground-truth labels, never sharing weights or architecture code
with model_baseline.py.

CONSTRAINT (see CONSTRAINTS.md "Protected Areas"): this file must never
import BaselineCNN or RobustCNN, and must never load their weights. Doing
so invalidates the black-box threat model that the project's novelty claim
depends on. See DECISIONS.md "Surrogate Model Must Be Architecturally
Different from Baseline" for the full reasoning.
"""

import torch
import torch.nn as nn


class SurrogateModel(nn.Module):
    """
    Deliberately different architecture from BaselineCNN -- e.g. a
    CNN-LSTM hybrid, or a CNN with different depth/kernel sizes.

    TODO (Person B): implement per design.md Section 5. Do NOT copy
    BaselineCNN's architecture.
    """

    def __init__(self, num_classes: int = 11):
        super().__init__()
        # TODO: define a DIFFERENT architecture than BaselineCNN
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def collect_query_pairs(baseline_model, X_query, query_budget: int = 5000):
    """
    Query the baseline model with X_query and collect (input, predicted-label)
    pairs. This simulates an attacker with query-only access -- no gradients,
    no weights, no ground-truth labels.

    TODO (Person B): implement per FLOW.md Section 2.
    """
    raise NotImplementedError
