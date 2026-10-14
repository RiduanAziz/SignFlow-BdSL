from __future__ import annotations

import torch

from sign_language.models import MultimodalFusionClassifier


def test_multimodal_fusion_classifier_combines_spatial_and_motion_features() -> None:
    spatial = torch.randn(2, 16, 42)
    motion = torch.randn(2, 16, 42)

    model = MultimodalFusionClassifier(
        spatial_dim=42,
        motion_dim=42,
        hidden_dim=32,
        num_classes=5,
    )

    logits = model(spatial, motion)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
