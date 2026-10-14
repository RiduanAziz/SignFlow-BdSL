from __future__ import annotations

import pytest

from sign_language.data import (
    build_manifest,
    generate_dataset_report,
    split_manifest_by_signer,
    validate_signer_assignments,
    validate_signer_independent_split,
)


def test_manifest_and_split_integrity() -> None:
    samples = [
        {
            "sample_id": "S001_HELLO_01",
            "signer_id": "S001",
            "class_id": 0,
            "class_name": "HELLO",
            "video_path": "data/raw/HELLO/S001_01.mp4",
            "num_frames": 32,
            "fps": 30,
            "duration": 1.06,
            "split": "train",
        },
        {
            "sample_id": "S001_THANKS_01",
            "signer_id": "S001",
            "class_id": 1,
            "class_name": "THANKS",
            "video_path": "data/raw/THANKS/S001_01.mp4",
            "num_frames": 28,
            "fps": 30,
            "duration": 0.93,
            "split": "train",
        },
        {
            "sample_id": "S002_HELLO_01",
            "signer_id": "S002",
            "class_id": 0,
            "class_name": "HELLO",
            "video_path": "data/raw/HELLO/S002_01.mp4",
            "num_frames": 30,
            "fps": 30,
            "duration": 1.0,
            "split": "train",
        },
        {
            "sample_id": "S003_HELLO_01",
            "signer_id": "S003",
            "class_id": 0,
            "class_name": "HELLO",
            "video_path": "data/raw/HELLO/S003_01.mp4",
            "num_frames": 35,
            "fps": 30,
            "duration": 1.17,
            "split": "validation",
        },
        {
            "sample_id": "S004_HELLO_01",
            "signer_id": "S004",
            "class_id": 0,
            "class_name": "HELLO",
            "video_path": "data/raw/HELLO/S004_01.mp4",
            "num_frames": 34,
            "fps": 30,
            "duration": 1.13,
            "split": "test",
        },
    ]

    manifest = build_manifest(samples)
    report = generate_dataset_report(manifest)
    assert report["class_count"] == 2
    assert report["sample_count"] == 5
    assert report["signer_count"] == 4
    assert report["train_count"] == 3
    assert report["validation_count"] == 1
    assert report["test_count"] == 1

    train_manifest, val_manifest, test_manifest = split_manifest_by_signer(
        manifest,
        train_signers={"S001", "S002"},
        val_signers={"S003"},
        test_signers={"S004"},
    )

    assert validate_signer_independent_split(train_manifest, val_manifest, test_manifest)


def test_overlapping_signer_assignments_fail() -> None:
    with pytest.raises(ValueError, match="Signer overlap"):
        validate_signer_assignments({"S001", "S002", "S003"}, {"S001"}, {"S001"}, {"S003"})
