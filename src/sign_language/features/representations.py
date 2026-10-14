from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureSpecification:
    name: str
    modalities: tuple[str, ...]
    feature_dim: int


_POINTS_PER_MODALITY = {"hands": 42, "pose": 33, "face": 478}
_FEATURES = {
    "hands": FeatureSpecification("hands", ("hands",), 126),
    "hands_pose": FeatureSpecification("hands_pose", ("hands", "pose"), 225),
    "holistic": FeatureSpecification("holistic", ("hands", "pose", "face"), 1659),
}


def get_feature_specification(name: str) -> FeatureSpecification:
    try:
        return _FEATURES[name]
    except KeyError as error:
        raise ValueError(f"Unsupported feature representation: {name}") from error


def select_feature_representation(sequence: np.ndarray, name: str) -> np.ndarray:
    """Select a documented prefix from the extractor's hands/pose/face ordering."""
    specification = get_feature_specification(name)
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Feature sequences must have shape [T, F]")
    if values.shape[1] < specification.feature_dim:
        raise ValueError(
            f"Representation {name} requires {specification.feature_dim} features, got {values.shape[1]}"
        )
    return values[:, : specification.feature_dim].copy()
