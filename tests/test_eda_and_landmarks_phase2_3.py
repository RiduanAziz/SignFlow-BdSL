from __future__ import annotations

import numpy as np

from sign_language.data.analysis import summarize_dataset
from sign_language.landmarks.pipeline import LandmarkPipeline


def test_dataset_summary_counts_and_sequence_stats() -> None:
    samples = [
        {
            "sample_id": "S001_HELLO_01",
            "signer_id": "S001",
            "label": "HELLO",
            "landmarks": np.ones((12, 21, 3), dtype=np.float32),
        },
        {
            "sample_id": "S001_HELLO_02",
            "signer_id": "S001",
            "label": "HELLO",
            "landmarks": np.ones((10, 21, 3), dtype=np.float32),
        },
        {
            "sample_id": "S002_THANKS_01",
            "signer_id": "S002",
            "label": "THANKS",
            "landmarks": np.ones((15, 21, 3), dtype=np.float32),
        },
    ]

    report = summarize_dataset(samples)
    assert report["sample_count"] == 3
    assert report["signer_count"] == 2
    assert report["label_count"] == 2
    assert report["min_sequence_length"] == 10
    assert report["max_sequence_length"] == 15
    assert report["average_sequence_length"] == 12.33


def test_landmark_pipeline_extracts_and_normalizes_sequence() -> None:
    pipeline = LandmarkPipeline(feature_dim=42, sequence_length=16)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    result = pipeline.process(frame)
    assert result.shape == (16, 42)
    assert np.isfinite(result).all()

    normalized = pipeline.normalize_sequence(result, include_dynamics=False)
    assert np.isfinite(normalized).all()
    assert normalized.shape == result.shape
