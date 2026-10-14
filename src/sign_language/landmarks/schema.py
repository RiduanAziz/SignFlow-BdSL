from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION: str = "1.0.0"

# Canonical modality definitions for MediaPipe Holistic
MODALITY_COUNTS: dict[str, int] = {
    "left_hand": 21,
    "right_hand": 21,
    "pose": 33,
    "face": 478,
}

# Modality ordering
MODALITY_ORDER: tuple[str, ...] = ("left_hand", "right_hand", "pose", "face")

# Total point count: 21 + 21 + 33 + 478 = 553
TOTAL_HOLISTIC_POINTS: int = sum(MODALITY_COUNTS.values())
TOTAL_HOLISTIC_COORDINATES: int = TOTAL_HOLISTIC_POINTS * 3  # 1659
TOTAL_DYNAMIC_FEATURES: int = TOTAL_HOLISTIC_COORDINATES * 3  # 4977 (pos, vel, acc)

# Modality offsets (point indices)
MODALITY_POINT_OFFSETS: dict[str, tuple[int, int]] = {
    "left_hand": (0, 21),
    "right_hand": (21, 42),
    "pose": (42, 75),
    "face": (75, 553),
}

# Modality offsets in flattened 3D coordinates [N * 3]
MODALITY_COORD_OFFSETS: dict[str, tuple[int, int]] = {
    "left_hand": (0, 63),
    "right_hand": (63, 126),
    "pose": (126, 225),
    "face": (225, 1659),
}


@dataclass(frozen=True)
class LandmarkObservation:
    """Flattened landmark values and a same-shaped mask for one frame."""

    landmarks: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        if self.landmarks.shape != self.mask.shape:
            raise ValueError(f"landmarks {self.landmarks.shape} and mask {self.mask.shape} must have the same shape")
        if self.landmarks.ndim != 1:
            raise ValueError("landmarks and mask must be one-dimensional")


@dataclass(frozen=True)
class StructuredLandmarkFrame:
    """Structured 3D landmark coordinates and point-level visibility mask for one frame."""

    coordinates: np.ndarray  # Shape: [N, 3]
    point_mask: np.ndarray   # Shape: [N] (1.0 = observed, 0.0 = missing)
    timestamp_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 2 or self.coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [N, 3]")
        if self.point_mask.ndim != 1 or self.point_mask.shape[0] != self.coordinates.shape[0]:
            raise ValueError("point_mask must have shape [N] matching coordinates [N, 3]")

    def get_modality(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract slice for a given modality (left_hand, right_hand, pose, face)."""
        start, end = MODALITY_POINT_OFFSETS[name]
        return self.coordinates[start:end], self.point_mask[start:end]


@dataclass(frozen=True)
class LandmarkSequence:
    """Sequence of landmark features with temporal and visibility masks."""

    landmarks: np.ndarray     # Shape: [T, F]
    mask: np.ndarray          # Shape: [T, F]
    sequence_mask: np.ndarray | None = None  # Shape: [T] boolean/float (1 = valid frame, 0 = padding)
    timestamps_ms: np.ndarray | None = None  # Shape: [T]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.landmarks.shape != self.mask.shape:
            raise ValueError("landmarks and mask must have the same shape")
        if self.landmarks.ndim != 2:
            raise ValueError("landmarks and mask must have shape [T, F]")


def compute_landmark_bounding_box(
    coordinates: np.ndarray,
    point_mask: np.ndarray,
    padding: float = 0.05,
) -> dict[str, float] | None:
    """Compute normalized bounding box (xmin, ymin, xmax, ymax, width, height) from valid landmark points.

    Coordinates must be [N, 3] or [N, 2] with normalized spatial range [0.0, 1.0].
    Returns None if no valid points exist or if box dimensions are degenerate.
    """
    coords = np.asarray(coordinates, dtype=np.float32)
    mask = np.asarray(point_mask, dtype=np.float32)

    valid_indices = np.where(mask > 0.5)[0]
    if len(valid_indices) == 0:
        return None

    pts = coords[valid_indices]
    raw_xmin = float(np.min(pts[:, 0]))
    raw_xmax = float(np.max(pts[:, 0]))
    raw_ymin = float(np.min(pts[:, 1]))
    raw_ymax = float(np.max(pts[:, 1]))

    w = max(0.0, raw_xmax - raw_xmin)
    h = max(0.0, raw_ymax - raw_ymin)

    if padding > 0.0:
        pad_x = max(0.005, w * padding)
        pad_y = max(0.005, h * padding)
    else:
        pad_x = 0.0
        pad_y = 0.0

    xmin = max(0.0, min(1.0, raw_xmin - pad_x))
    xmax = max(0.0, min(1.0, raw_xmax + pad_x))
    ymin = max(0.0, min(1.0, raw_ymin - pad_y))
    ymax = max(0.0, min(1.0, raw_ymax + pad_y))

    final_w = xmax - xmin
    final_h = ymax - ymin

    if final_w < 1e-4 or final_h < 1e-4:
        return None

    return {
        "xmin": float(round(xmin, 4)),
        "ymin": float(round(ymin, 4)),
        "xmax": float(round(xmax, 4)),
        "ymax": float(round(ymax, 4)),
        "width": float(round(final_w, 4)),
        "height": float(round(final_h, 4)),
    }

