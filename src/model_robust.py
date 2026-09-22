"""
model_robust.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 4 (Week 5-6)

Responsibility:
    Same base architecture as BaselineCNN, trained via PGD-adversarial
    training.

CONSTRAINT:
    Adversarial training uses PGD ONLY.

Dataset:
    RadioML 2018.01A
    24 classes
    Input shape: (2, 1024)
"""

from src.model_baseline import BaselineCNN


class RobustCNN(BaselineCNN):
    """
    Robust CNN for RadioML 2018.01A.

    Same architecture as BaselineCNN.
    Robustness comes from PGD adversarial training.
    """

    def __init__(self, num_classes: int = 24):
        super().__init__(num_classes=num_classes)