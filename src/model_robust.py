"""
model_robust.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 4 (Week 5-6)

Responsibility: same base architecture as BaselineCNN, trained via
PGD-adversarial training (the training procedure itself lives in train.py's
adversarial_fn hook -- this file only defines the model).

CONSTRAINT: adversarial training uses PGD ONLY, not a mix of attacks.
This is a deliberate research-design choice -- see DECISIONS.md
"Robust Model Trained Against PGD Only". Do not change without team
discussion; it defines the entire generalization-gap experiment.
"""

import torch
import torch.nn as nn
from src.model_baseline import BaselineCNN


class RobustCNN(BaselineCNN):
    """
    Same architecture as BaselineCNN. The 'robustness' comes entirely from
    HOW it's trained (see train.py), not from architectural differences.

    TODO (Person B): confirm this can simply reuse BaselineCNN's forward()
    unchanged. If an architectural tweak (denoising layer / attention) is
    added later as a stretch goal, document it here and in design.md.
    """

    def __init__(self, num_classes: int = 11):
        super().__init__(num_classes=num_classes)
