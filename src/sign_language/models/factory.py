from __future__ import annotations

import torch

from sign_language.config import ExperimentConfig

from .baseline import MLPClassifier
from .fusion import RobustHolisticFusionClassifier
from .gru import GRUClassifier
from .lstm import LSTMClassifier
from .transformer import TemporalTransformerClassifier


def build_model(config: ExperimentConfig) -> torch.nn.Module:
    """Build the configured model using one canonical model factory."""
    params = config.model
    name = str(params.name).lower().strip()

    if name in {"mlp", "baseline"}:
        return MLPClassifier(
            input_dim=params.input_dim,
            seq_len=params.sequence_length,
            num_classes=params.num_classes,
        )
    if name == "lstm":
        return LSTMClassifier(
            input_dim=params.input_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            num_classes=params.num_classes,
            bidirectional=params.bidirectional,
            dropout=params.dropout,
        )
    if name == "gru":
        return GRUClassifier(
            input_dim=params.input_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            num_classes=params.num_classes,
            bidirectional=params.bidirectional,
            dropout=params.dropout,
        )
    if name in {"robust_holistic_fusion", "fusion", "proposed"}:
        return RobustHolisticFusionClassifier(
            input_dim=params.input_dim,
            num_classes=params.num_classes,
            modality_dim=getattr(params, "embedding_dim", 64),
            hidden_dim=getattr(params, "hidden_dim", 64),
            num_layers=getattr(params, "num_layers", 1),
            dropout=params.dropout,
        )
    if name == "transformer":
        return TemporalTransformerClassifier(
            input_dim=params.input_dim,
            embedding_dim=params.embedding_dim,
            num_heads=params.num_heads,
            num_layers=params.num_layers,
            ff_dim=params.ff_dim,
            num_classes=params.num_classes,
            dropout=params.dropout,
            max_sequence_length=params.max_sequence_length,
        )

    # Default to transformer if unspecified
    return TemporalTransformerClassifier(
        input_dim=params.input_dim,
        embedding_dim=params.embedding_dim,
        num_heads=params.num_heads,
        num_layers=params.num_layers,
        ff_dim=params.ff_dim,
        num_classes=params.num_classes,
        dropout=params.dropout,
        max_sequence_length=params.max_sequence_length,
    )
