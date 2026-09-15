"""
model_baseline.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 2 (Week 2-3)

Responsibility: standard CNN architecture for modulation classification.
No training loop logic belongs here -- that lives in train.py.

Input shape: (batch, 2, 128) -- I and Q channels, 128 time steps.
Output: logits over 11 modulation classes (RadioML 2016.10a).

See design.md Section 3 for architecture rationale.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple CNN: a few Conv1d layers -> pooling -> dense -> softmax.

    TODO (Person A): implement per design.md Section 3.
    Keep it simple first -- this is the MVP model, tune later.
    """

    def __init__(self, num_classes: int = 11):
        super().__init__()
        # TODO: define Conv1d layers, pooling, dense layers
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward pass
        raise NotImplementedError
