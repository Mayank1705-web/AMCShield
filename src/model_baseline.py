"""
model_baseline.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 2 (Week 2-3)

Responsibility:
    Standard CNN architecture for RadioML 2018.01A modulation
    classification.

Input:
    (batch, 2, 1024)

Output:
    (batch, 24) logits

Important:
    The model returns raw logits.
    Softmax is NOT applied here because CrossEntropyLoss expects
    raw logits during training.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Baseline 1D CNN for automatic modulation classification.

    Architecture:

        Input
        (2, 1024)
           ↓
        Conv1d(2 → 64)
           ↓
        BatchNorm
           ↓
        ReLU
           ↓
        MaxPool
           ↓
        Conv1d(64 → 128)
           ↓
        BatchNorm
           ↓
        ReLU
           ↓
        MaxPool
           ↓
        Conv1d(128 → 256)
           ↓
        BatchNorm
           ↓
        ReLU
           ↓
        Adaptive Average Pooling
           ↓
        Fully Connected
           ↓
        24 class logits
    """

    def __init__(
        self,
        num_classes: int = 24,
    ):
        super().__init__()

        self.features = nn.Sequential(

            # ------------------------------------------------
            # Block 1
            # Input: (B, 2, 1024)
            # Output: (B, 64, 512)
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=2,
                out_channels=64,
                kernel_size=7,
                padding=3,
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
            ),

            # ------------------------------------------------
            # Block 2
            # Input: (B, 64, 512)
            # Output: (B, 128, 256)
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
            ),

            # ------------------------------------------------
            # Block 3
            # Input: (B, 128, 256)
            # Output: (B, 256, 128)
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            # ------------------------------------------------
            # Global feature aggregation
            #
            # (B, 256, 128)
            #       ↓
            # (B, 256, 1)
            # ------------------------------------------------

            nn.AdaptiveAvgPool1d(1),
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.3
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# Manual model test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AMCShield Baseline CNN Test")
    print("=" * 70)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = BaselineCNN(
        num_classes=24
    )

    print()
    print("Model created successfully.")

    # --------------------------------------------------------
    # Create dummy RadioML input
    # --------------------------------------------------------

    x = torch.randn(
        4,
        2,
        1024,
    )

    print()
    print(
        f"Input shape  : {tuple(x.shape)}"
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(x)

    print(
        f"Output shape : {tuple(output.shape)}"
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    assert output.shape == (
        4,
        24,
    )

    print()
    print("✓ Input shape verified")
    print("✓ Output shape verified")
    print("✓ 24 modulation classes verified")

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Total parameters     : {parameters:,}"
    )

    print(
        f"Trainable parameters : {trainable_parameters:,}"
    )

    print()
    print("=" * 70)
    print("BASELINE CNN TEST PASSED")
    print("=" * 70)