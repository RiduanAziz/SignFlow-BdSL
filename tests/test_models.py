from __future__ import annotations

import torch

from sign_language.models import (
    GRUClassifier,
    LSTMClassifier,
    MLPClassifier,
    TemporalTransformerClassifier,
)


def test_model_forward_passes() -> None:
    batch = torch.randn(2, 16, 42)

    mlp = MLPClassifier(input_dim=42, seq_len=16, num_classes=5)
    lstm = LSTMClassifier(input_dim=42, hidden_dim=16, num_layers=1, num_classes=5)
    gru = GRUClassifier(input_dim=42, hidden_dim=16, num_layers=1, num_classes=5)
    transformer = TemporalTransformerClassifier(input_dim=42, embedding_dim=16, num_heads=4, num_layers=1, ff_dim=32, num_classes=5, max_sequence_length=16)

    assert mlp(batch).shape == (2, 5)
    assert lstm(batch).shape == (2, 5)
    assert gru(batch).shape == (2, 5)
    assert transformer(batch).shape == (2, 5)
