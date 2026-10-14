import numpy as np
import pytest

from sign_language.features import get_feature_specification, select_feature_representation


def test_feature_dimensions_are_explicit() -> None:
    assert get_feature_specification("hands").feature_dim == 126
    assert get_feature_specification("hands_pose").feature_dim == 225
    assert get_feature_specification("holistic").feature_dim == 1659


def test_representation_selection_preserves_time_and_selects_prefix() -> None:
    sequence = np.arange(2 * 1659, dtype=np.float32).reshape(2, 1659)
    selected = select_feature_representation(sequence, "hands_pose")

    assert selected.shape == (2, 225)
    assert np.array_equal(selected, sequence[:, :225])


def test_unknown_representation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported feature representation"):
        get_feature_specification("unknown")
