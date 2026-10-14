from __future__ import annotations

from typing import Any

import numpy as np


def pad_or_truncate_sequence(
    sequence: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad or truncate a landmark sequence to a fixed time length."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[0] >= target_length:
        return arr[:target_length].copy()

    pad_shape = (target_length - arr.shape[0],) + arr.shape[1:]
    padding = np.full(pad_shape, pad_value, dtype=np.float32)
    return np.concatenate([arr, padding], axis=0)


def pad_or_truncate_with_mask(
    sequence: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad or truncate a sequence and return its valid-frame mask and length."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    valid_length = min(arr.shape[0], target_length)
    padded = pad_or_truncate_sequence(arr, target_length, pad_value=pad_value)
    mask = np.zeros(target_length, dtype=bool)
    mask[:valid_length] = True
    return padded, mask, valid_length


def sequence_to_tensor(
    sample: Any,
    target_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Convert a sample landmark sequence into a fixed-length tensor."""
    arr = np.asarray(sample.landmarks, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return pad_or_truncate_sequence(arr, target_length, pad_value=pad_value)


def temporal_augmentation(
    sequence: np.ndarray,
    frame_drop_prob: float = 0.0,
    jitter_scale: float = 0.0,
) -> np.ndarray:
    """Apply lightweight temporal augmentation for robust training."""
    arr = np.asarray(sequence, dtype=np.float32).copy()
    if frame_drop_prob > 0.0:
        drop_mask = np.random.rand(arr.shape[0]) > frame_drop_prob
        arr = arr[drop_mask]
        if len(arr) == 0:
            return sequence.copy()
    if jitter_scale > 0.0:
        noise = np.random.normal(0.0, jitter_scale, size=arr.shape).astype(np.float32)
        arr = arr + noise
    return arr
