"""
model_surrogate.py

AMCShield - Black-Box Surrogate Model

The surrogate is intentionally architecturally different from BaselineCNN.

Threat model:
    1. Attacker has query-only access to the baseline model.
    2. Attacker receives predicted labels only.
    3. Ground-truth labels are NOT used to train the surrogate.
    4. Baseline/robust model weights are never copied.
    5. Gradients are never computed through the target model.

Architecture:
    CNN feature extractor -> BiLSTM -> classifier

Input:
    (B, 2, 1024)

Output:
    (B, 24)
"""

from typing import Tuple

import torch
import torch.nn as nn


class SurrogateModel(nn.Module):
    """
    CNN-LSTM surrogate.

    This is deliberately different from BaselineCNN:
        Baseline:
            Conv1d 2->64 -> Conv1d 64->128 -> Conv1d 128->256
            -> AdaptiveAvgPool -> FC

        Surrogate:
            Conv1d 2->32 -> Conv1d 32->64 -> Conv1d 64->64
            -> BiLSTM -> LayerNorm -> FC

    No baseline or robust weights are loaded here.
    """

    def __init__(
        self,
        num_classes: int = 24,
    ):
        super().__init__()

        # ----------------------------------------------------
        # CNN feature extractor
        # ----------------------------------------------------

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=2,
                out_channels=32,
                kernel_size=9,
                padding=4,
            ),

            nn.BatchNorm1d(32),

            nn.GELU(),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                padding=3,
            ),

            nn.BatchNorm1d(64),

            nn.GELU(),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),

            nn.GELU(),
        )

        # ----------------------------------------------------
        # Sequence model
        #
        # Input after CNN:
        #     (B, 64, 256)
        #
        # LSTM expects:
        #     (B, sequence, features)
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # BiLSTM output:
        #     64 * 2 = 128
        self.classifier = nn.Sequential(

            nn.LayerNorm(128),

            nn.Linear(
                128,
                64,
            ),

            nn.GELU(),

            nn.Dropout(
                p=0.2
            ),

            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input (B, 2, 1024), "
                f"got {tuple(x.shape)}"
            )

        if tuple(x.shape[1:]) != (2, 1024):
            raise ValueError(
                f"Expected input shape (B, 2, 1024), "
                f"got {tuple(x.shape)}"
            )

        # CNN
        x = self.features(x)

        # (B, 64, 256)
        # ->
        # (B, 256, 64)
        x = x.transpose(1, 2)

        # BiLSTM
        x, _ = self.lstm(x)

        # Last sequence representation
        x = x[:, -1, :]

        # Classification
        x = self.classifier(x)

        return x


# ============================================================
# Query Collection
# ============================================================

def collect_query_pairs(
    baseline_model,
    X_query,
    query_budget: int = 5000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect query-response pairs from the baseline.

    The attacker receives:

        input signal -> predicted class

    The attacker does NOT receive:

        - ground-truth labels
        - baseline gradients
        - baseline weights

    Args:
        baseline_model:
            Trained BaselineCNN.

        X_query:
            Tensor containing query signals:
            (N, 2, 1024)

        query_budget:
            Maximum number of baseline queries.

    Returns:
        X_pairs:
            Queried signals.

        predicted_labels:
            Labels predicted by the baseline.

    Important:
        Ground-truth labels are completely ignored.
    """

    if query_budget <= 0:
        raise ValueError(
            "query_budget must be greater than zero."
        )

    # --------------------------------------------------------
    # Convert input to tensor
    # --------------------------------------------------------

    if isinstance(X_query, torch.Tensor):

        X = X_query.detach()

    else:

        X = torch.as_tensor(
            X_query
        )

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(
            f"Expected X_query with 3 dimensions, "
            f"got {tuple(X.shape)}"
        )

    if tuple(X.shape[1:]) != (2, 1024):

        raise ValueError(
            f"Expected X_query shape (N, 2, 1024), "
            f"got {tuple(X.shape)}"
        )

    # --------------------------------------------------------
    # Respect query budget
    # --------------------------------------------------------

    number_of_queries = min(
        int(query_budget),
        len(X),
    )

    if number_of_queries == 0:

        raise ValueError(
            "X_query contains no samples."
        )

    X = X[
        :number_of_queries
    ].float().cpu()

    # --------------------------------------------------------
    # Baseline inference
    # --------------------------------------------------------

    device = next(
        baseline_model.parameters()
    ).device

    baseline_model.eval()

    predicted_labels = []

    # IMPORTANT:
    # No gradients are created.
    with torch.no_grad():

        for start in range(
            0,
            number_of_queries,
            256,
        ):

            batch = X[
                start:start + 256
            ].to(device)

            logits = baseline_model(
                batch
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            predicted_labels.append(
                predictions.cpu()
            )

    predicted_labels = torch.cat(
        predicted_labels,
        dim=0,
    ).long()

    return (
        X,
        predicted_labels,
    )


# ============================================================
# Manual Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AMCShield Surrogate Model Test")
    print("=" * 70)

    model = SurrogateModel(
        num_classes=24
    )

    x = torch.randn(
        4,
        2,
        1024,
    )

    with torch.no_grad():

        output = model(x)

    print(
        f"Input shape  : {tuple(x.shape)}"
    )

    print(
        f"Output shape : {tuple(output.shape)}"
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters   : {parameters:,}"
    )

    assert output.shape == (
        4,
        24,
    )

    print()
    print("✓ Input shape verified")
    print("✓ Output shape verified")
    print("✓ 24 classes verified")
    print("✓ Surrogate architecture test passed")

    print("=" * 70)