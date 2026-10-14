from __future__ import annotations

import numpy as np


def normalize_landmarks(landmarks: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Translate and scale landmark coordinates to a consistent range."""
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0:
        return arr
    center = arr.mean(axis=-1, keepdims=True)
    centered = arr - center
    scale = np.linalg.norm(centered, axis=-1, keepdims=True)
    scale = np.where(scale < eps, 1.0, scale)
    return centered / scale
