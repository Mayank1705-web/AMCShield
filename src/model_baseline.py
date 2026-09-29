"""
model_baseline.py
Owner: Mayank Ingole [EN24CS3T10017]
Phase: 2 (Week 2-3)

Responsibility:
    CNN architecture for RadioML 2018.01A modulation
    classification.

Input:
    (batch, 2, 1024)

Output:
    (batch, 24) logits

Important:
    The model returns raw logits.
    Softmax is NOT applied here because CrossEntropyLoss expects
    raw logits during training.

Revision note:
    Upgraded from the original 3-block plain CNN to a deeper
    residual CNN with squeeze-excite (SE) attention and combined
    avg+max global pooling. The original architecture pooled to
    a single timestep after only 128 timesteps of context, which
    discards temporal/phase structure that matters for
    discriminating modulations (e.g. PSK vs QAM families) and is
    part of why accuracy plateaued below what this dataset can
    support. RobustCNN (model_robust.py) subclasses this class,
    so it benefits from the same upgrade automatically.
"""


import torch
import torch.nn as nn


class SEBlock1d(nn.Module):
    """
    Squeeze-and-excitation block for 1D feature maps.

    Learns a per-channel gate from the global context of the
    sequence, letting the network emphasize the channels that
    are most discriminative for a given signal rather than
    treating every channel equally.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            4,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        b, c, _ = x.shape

        weights = self.pool(x).view(b, c)
        weights = self.gate(weights).view(b, c, 1)

        return x * weights


class ResidualBlock1d(nn.Module):
    """
    Conv1d -> BN -> ReLU -> Conv1d -> BN, with a (possibly
    projected) skip connection, followed by SE attention,
    ReLU, and optional max-pooling.

    Residual connections let gradients flow through the deeper
    stack without the vanishing-gradient issues that made the
    original 3-block plain CNN hard to extend.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        self.pool = (
            nn.MaxPool1d(kernel_size=2, stride=2)
            if pool
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)

        out = self.relu(out + identity)

        out = self.pool(out)

        return out


class BaselineCNN(nn.Module):
    """
    Residual 1D CNN for automatic modulation classification.

    Architecture:

        Input
        (2, 1024)
           ↓
        ResidualBlock(2 → 64,  k=7, pool)   -> (B,  64, 512)
           ↓
        ResidualBlock(64 → 128, k=5, pool)  -> (B, 128, 256)
           ↓
        ResidualBlock(128 → 256, k=3, pool) -> (B, 256, 128)
           ↓
        ResidualBlock(256 → 256, k=3, no pool) -> (B, 256, 128)
           ↓
        Global Avg Pool ⊕ Global Max Pool
           ↓
        (B, 512)
           ↓
        Fully Connected
           ↓
        24 class logits

    Each ResidualBlock includes SE (squeeze-excite) channel
    attention. The final block keeps full temporal resolution
    (no pooling) before global pooling, instead of collapsing
    to a single timestep right after block 3 as in the
    original architecture.
    """

    def __init__(
        self,
        num_classes: int = 24,
    ):
        super().__init__()

        self.features = nn.Sequential(
            ResidualBlock1d(2, 64, kernel_size=7, pool=True),
            ResidualBlock1d(64, 128, kernel_size=5, pool=True),
            ResidualBlock1d(128, 256, kernel_size=3, pool=True),
            ResidualBlock1d(256, 256, kernel_size=3, pool=False),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # ----------------------------------------------------
        # Classifier
        #
        # Input is 512 = 256 (avg-pooled) + 256 (max-pooled).
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                512,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.4
            ),

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

        avg_pooled = self.global_avg_pool(x)
        max_pooled = self.global_max_pool(x)

        x = torch.cat(
            [avg_pooled, max_pooled],
            dim=1,
        )

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