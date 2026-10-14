from __future__ import annotations

import torch
from torch import nn


class TemporalTransformerClassifier(nn.Module):
    """A compact temporal transformer encoder for landmark sequences."""

    def __init__(
        self,
        input_dim: int = 42,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.1,
        max_sequence_length: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, embedding_dim))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, seq_len, _ = x.shape
        x = self.input_proj(x)

        # Handle positional embeddings safely up to seq_len
        if seq_len > self.positional_embedding.size(1):
            pos_emb = torch.nn.functional.interpolate(
                self.positional_embedding.transpose(1, 2), size=seq_len, mode="linear"
            ).transpose(1, 2)
        else:
            pos_emb = self.positional_embedding[:, :seq_len, :]

        x = x + pos_emb
        x = self.dropout(x)

        temporal_mask = None
        padding_mask = None
        if mask is not None:
            if mask.dim() == 3:
                # Collapse feature mask [B, T, D] to temporal valid mask [B, T]
                temporal_mask = (mask.abs().sum(dim=-1) > 1e-5).float()
            elif mask.dim() == 2:
                temporal_mask = mask.float()
            padding_mask = ~(temporal_mask.bool())
            # Ensure not all tokens in a row are masked
            all_masked = padding_mask.all(dim=-1)
            if all_masked.any():
                padding_mask[all_masked, 0] = False

        x = self.encoder(x, src_key_padding_mask=padding_mask)

        if temporal_mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = temporal_mask.to(dtype=x.dtype).unsqueeze(-1)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        return self.classifier(pooled)
