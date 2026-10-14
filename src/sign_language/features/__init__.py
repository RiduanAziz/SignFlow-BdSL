"""Feature normalization and building modules."""

from .normalization import normalize_landmarks
from .representations import (
    FeatureSpecification,
    get_feature_specification,
    select_feature_representation,
)

__all__ = [
    "FeatureSpecification",
    "get_feature_specification",
    "normalize_landmarks",
    "select_feature_representation",
]
