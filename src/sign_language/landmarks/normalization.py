from __future__ import annotations

import numpy as np


def normalize_landmarks(
    landmarks: np.ndarray,
    mode: str = "center_scale",
    eps: float = 1e-6,
    mask: np.ndarray | None = None,
    reference_index: int | None = None,
) -> np.ndarray:
    """Normalize coordinates per frame using visible points and geometric scale.

    Operates on true spatial coordinates (3D or 2D geometry) rather than
    unstructured flattened vectors. Supports 1D [D], 2D [T, D], and 3D [T, N, 3] inputs.
    """
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()

    if mode == "none":
        return arr.copy()

    was_1d = (arr.ndim == 1)
    if was_1d:
        arr = arr.reshape(1, -1)

    # 1. 3D tensor: [T, N, 3]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        coords = arr.copy()
        num_frames = coords.shape[0]
        num_points = coords.shape[1]
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.float32)
            if mask_arr.ndim == 3 and mask_arr.shape == coords.shape:
                pt_valid = (mask_arr > 0.5).any(axis=-1)
            elif mask_arr.ndim == 2 and mask_arr.shape == (num_frames, num_points * 3):
                pt_valid = (mask_arr.reshape(num_frames, num_points, 3) > 0.5).any(axis=-1)
            elif mask_arr.ndim == 2 and mask_arr.shape == (num_frames, num_points):
                pt_valid = mask_arr > 0.5
            else:
                pt_valid = np.ones((num_frames, num_points), dtype=bool)
        else:
            pt_valid = (np.abs(coords).sum(axis=-1) > 1e-6)

        for t in range(num_frames):
            visible = pt_valid[t]
            if not visible.any():
                coords[t] = 0.0
                continue
            if reference_index is not None and reference_index < num_points and visible[reference_index]:
                center = coords[t, reference_index]
            else:
                center = coords[t, visible].mean(axis=0)
            centered = coords[t] - center
            scale = np.sqrt(np.mean(np.square(centered[visible])))
            if scale >= eps:
                coords[t] = centered / scale
            else:
                coords[t] = centered
            coords[t, ~visible] = 0.0
        return coords

    # 2. Flattened 3D coordinate sequence [T, N * 3]
    if arr.ndim == 2 and arr.shape[-1] % 3 == 0 and arr.shape[-1] >= 6:
        num_frames = arr.shape[0]
        num_points = arr.shape[1] // 3
        coords = arr.reshape(num_frames, num_points, 3).copy()

        # Parse mask if provided
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=np.float32)
            if was_1d and mask_arr.ndim == 1:
                mask_arr = mask_arr.reshape(1, -1)

            if mask_arr.shape == arr.shape:
                pt_valid = (mask_arr.reshape(num_frames, num_points, 3) > 0.5).any(axis=-1)
            elif mask_arr.shape == (num_frames, num_points):
                pt_valid = mask_arr > 0.5
            elif mask_arr.ndim == 3 and mask_arr.shape == (num_frames, num_points, 3):
                pt_valid = (mask_arr > 0.5).any(axis=-1)
            else:
                pt_valid = np.ones((num_frames, num_points), dtype=bool)
        else:
            # Derive validity from non-zero coordinates
            pt_valid = (np.abs(coords).sum(axis=-1) > 1e-6)

        for t in range(num_frames):
            visible = pt_valid[t]
            if not visible.any():
                coords[t] = 0.0
                continue

            if reference_index is not None and reference_index < num_points and visible[reference_index]:
                center = coords[t, reference_index]
            else:
                center = coords[t, visible].mean(axis=0)

            centered = coords[t] - center
            scale = np.sqrt(np.mean(np.square(centered[visible])))
            if scale >= eps:
                coords[t] = centered / scale
            else:
                coords[t] = centered
            coords[t, ~visible] = 0.0

        res = coords.reshape(num_frames, -1)
        return res[0] if was_1d else res

    # 3. General 2D array of coordinates [N, D] (where rows are points)
    if mode == "center_scale":
        center = arr.mean(axis=0, keepdims=True)
        centered = arr - center
        scale = np.sqrt(np.mean(np.square(centered)))
        if scale < eps:
            scale = 1.0
        res = centered / scale
        return res[0] if was_1d else res

    if mode == "standardize":
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        res = (arr - mean) / std
        return res[0] if was_1d else res

    if mode == "minmax":
        min_val = arr.min(axis=0, keepdims=True)
        max_val = arr.max(axis=0, keepdims=True)
        denom = np.where((max_val - min_val) < eps, 1.0, max_val - min_val)
        res = (arr - min_val) / denom
        return res[0] if was_1d else res

    raise ValueError(f"Unsupported normalization mode: {mode}")
