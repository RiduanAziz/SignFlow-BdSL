from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """LSTM classifier for temporal sequences of landmark features."""

    def __init__(
        self,
        input_dim: int = 42,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        bidirectional: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        direction = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * direction, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is None and mask is not None:
            lengths = mask.long().sum(dim=1)
        if lengths is not None:
            lengths = lengths.clamp(min=1, max=x.shape[1]).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
            direction = 2 if self.lstm.bidirectional else 1
            last_hidden = hidden[-direction:].transpose(0, 1).reshape(x.shape[0], -1)
        else:
            outputs, _ = self.lstm(x)
            last_hidden = outputs[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.classifier(last_hidden)
