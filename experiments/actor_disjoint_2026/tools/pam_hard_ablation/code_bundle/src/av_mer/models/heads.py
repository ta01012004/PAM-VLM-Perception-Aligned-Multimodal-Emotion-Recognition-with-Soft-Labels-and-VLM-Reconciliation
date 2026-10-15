"""Independent hard-label classifier heads for each model branch."""

from __future__ import annotations

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """Map pooled ``[B,H]`` features to six-class logits."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0 or num_classes <= 1:
            raise ValueError("Classifier input_dim must be positive and num_classes > 1.")
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected classifier features [B,{self.input_dim}], "
                f"received {tuple(features.shape)}."
            )
        return self.network(features)

