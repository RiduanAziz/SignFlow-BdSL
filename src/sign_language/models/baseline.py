from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Simple MLP baseline that flattens sequence features before classification."""

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim: int = 128,
        num_classes: int = 5,
        seq_len: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        return self.net(x)
