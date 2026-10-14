import numpy as np
import pytest

from sign_language.robustness import (
    apply_combined_noise,
    apply_spatial_noise,
    apply_temporal_noise,
    recognition_robustness_report,
)


def test_spatial_noise_is_deterministic_and_shape_preserving() -> None:
    sequence = np.ones((8, 12), dtype=np.float32)

    first = apply_spatial_noise(sequence, "coordinate_jitter", 0.2, seed=7)
    second = apply_spatial_noise(sequence, "coordinate_jitter", 0.2, seed=7)

    assert first.shape == sequence.shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, sequence)


def test_temporal_noise_changes_length_without_changing_features() -> None:
    sequence = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)

    dropped = apply_temporal_noise(sequence, "frame_drop", 0.5, seed=7)
    duplicated = apply_temporal_noise(sequence, "frame_duplicate", 0.5, seed=7)

    assert 1 <= dropped.shape[0] < sequence.shape[0]
    assert duplicated.shape[1] == sequence.shape[1]
    assert duplicated.shape[0] > sequence.shape[0]


def test_combined_noise_is_deterministic_and_validates_severity() -> None:
    sequence = np.ones((8, 6), dtype=np.float32)
    first = apply_combined_noise(sequence, "scale", "sequence_truncate", 0.2, 0.25, seed=4)
    second = apply_combined_noise(sequence, "scale", "sequence_truncate", 0.2, 0.25, seed=4)

    assert np.array_equal(first, second)
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_spatial_noise(sequence, "scale", 1.1)


def test_recognition_robustness_reports_metric_degradation() -> None:
    report = recognition_robustness_report(
        targets=[0, 1, 0, 1],
        clean_predictions=[0, 1, 0, 1],
        noisy_predictions=[0, 0, 0, 1],
        class_names=["A", "B"],
        noise_type="frame_drop",
        severity=0.5,
    )

    assert report["clean"]["accuracy"] == 1.0
    assert report["noisy"]["accuracy"] == 0.75
    assert report["accuracy_degradation"] == 0.25
