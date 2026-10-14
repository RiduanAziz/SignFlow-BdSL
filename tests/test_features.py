from __future__ import annotations

import numpy as np

from sign_language.features import normalize_landmarks


def test_landmark_normalization_is_translation_invariant() -> None:
    landmarks = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    shifted = landmarks + np.array([10.0, 20.0], dtype=np.float32)

    normalized = normalize_landmarks(landmarks)
    shifted_normalized = normalize_landmarks(shifted)

    assert np.allclose(normalized, shifted_normalized, atol=1e-5)


def test_landmark_normalization_is_scale_invariant() -> None:
    landmarks = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    scaled = landmarks * 3.0

    normalized = normalize_landmarks(landmarks)
    scaled_normalized = normalize_landmarks(scaled)

    assert np.allclose(normalized, scaled_normalized, atol=1e-5)
