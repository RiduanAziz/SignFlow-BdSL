from __future__ import annotations

import numpy as np

from sign_language.landmarks import LandmarkExtractor, normalize_landmarks


def test_landmark_extractor_shapes() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    points = LandmarkExtractor(feature_dim=42).extract(frame)
    assert points.shape == (42,)
    assert np.isfinite(points).all()


def test_structured_normalization_is_translation_and_scale_invariant() -> None:
    points = np.array([[[1.0, 2.0, 0.0], [3.0, 2.0, 0.0]]], dtype=np.float32)
    shifted_scaled = points * 4.0 + np.array([10.0, -2.0, 3.0], dtype=np.float32)

    normalized = normalize_landmarks(points)
    transformed = normalize_landmarks(shifted_scaled)

    assert np.allclose(normalized, transformed, atol=1e-5)
