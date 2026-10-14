"""Comprehensive unit tests for SignFlow-BdSL (BDSLW131) migration.

Validates canonical vocabulary, dataset manifests, split leakage prevention,
checkpoint contracts, bilingual inference, and CI synthetic fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sign_language.api.inference import RealTimePredictor
from sign_language.config.vocabulary import get_bdsl_vocabulary
from sign_language.robustness import (
    apply_spatial_noise,
    apply_temporal_noise,
    landmark_dropout,
)


def test_bdsl_canonical_vocabulary_integrity() -> None:
    """Validate that the canonical BDSLW131 vocabulary has exactly 131 verified classes."""
    vocab = get_bdsl_vocabulary()
    assert vocab.num_classes == 131, f"Expected 131 classes, got {vocab.num_classes}"

    seen_ids = set()
    seen_labels = set()
    for item in vocab.items:
        cid = item["class_id"]
        dlabel = item["dataset_label"]
        blabel = item["bengali_label"]
        egloss = item["english_gloss"]

        assert 0 <= cid < 131
        assert cid not in seen_ids, f"Duplicate class_id: {cid}"
        seen_ids.add(cid)

        assert dlabel not in seen_labels, f"Duplicate dataset_label: {dlabel}"
        seen_labels.add(dlabel)

        # Ensure non-empty Bengali and English text
        assert len(blabel.strip()) > 0, f"Empty Bengali label for class {cid}"
        assert len(egloss.strip()) > 0, f"Empty English gloss for class {cid}"

    assert len(seen_ids) == 131


def test_bdsl_vocabulary_bidirectional_lookup() -> None:
    """Validate ID and label bidirectional resolution."""
    vocab = get_bdsl_vocabulary()
    item_0 = vocab.get_by_id(0)
    dlabel_0 = item_0["dataset_label"]
    blabel_0 = item_0["bengali_label"]

    assert len(dlabel_0) > 0
    assert len(blabel_0) > 0

    # Lookup by dataset label
    assert vocab.get_id_by_label(dlabel_0) == 0
    # Lookup by Bengali label
    assert vocab.get_id_by_label(blabel_0) == 0

    with pytest.raises(KeyError):
        vocab.get_by_id(999)

    with pytest.raises(KeyError):
        vocab.get_id_by_label("non_existent_label_xyz")


def test_bdsl_manifest_and_splits_no_leakage() -> None:
    """Verify that train, validation, and test splits have zero sample leakage."""
    train_manifest = Path("data/manifests/bdslw131_train_manifest.json")
    val_manifest = Path("data/manifests/bdslw131_val_manifest.json")
    test_manifest = Path("data/manifests/bdslw131_test_manifest.json")

    if not (train_manifest.is_file() and val_manifest.is_file() and test_manifest.is_file()):
        pytest.skip("Manifest files not present in workspace")

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_manifest, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_samples = {s["sample_id"] for s in train_data}
    val_samples = {s["sample_id"] for s in val_data}
    test_samples = {s["sample_id"] for s in test_data}

    # Assert strictly disjoint sets (zero data leakage)
    assert len(train_samples & val_samples) == 0, "Leakage between train and validation sets!"
    assert len(train_samples & test_samples) == 0, "Leakage between train and test sets!"
    assert len(val_samples & test_samples) == 0, "Leakage between validation and test sets!"

    # Assert all 131 classes are represented in train set
    train_classes = {s["label"] for s in train_data}
    assert len(train_classes) == 131, f"Train set only contains {len(train_classes)} classes"


def test_bdsl_checkpoint_contract() -> None:
    """Validate that saved BdSL checkpoint conforms to the 131-class architecture."""
    ckpt_path = Path("checkpoints/bdslw131/robust_holistic_fusion_best.pt")
    if not ckpt_path.is_file():
        pytest.skip("BdSL checkpoint not present")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "class_names" in ckpt
    assert len(ckpt["class_names"]) == 131
    assert ckpt["num_classes"] == 131
    assert ckpt["input_dim"] == 1659

    # Verify output head weights match 131 classes
    state = ckpt["model_state_dict"]
    head_weights = [v for k, v in state.items() if "classifier" in k and "weight" in k]
    assert len(head_weights) > 0
    last_head = head_weights[-1]
    assert last_head.shape[0] == 131, f"Expected 131 output logits, got {last_head.shape[0]}"


def test_realtime_predictor_bilingual_decoding() -> None:
    """Verify that RealTimePredictor decodes 131 classes with Bengali and English labels."""
    ckpt_path = Path("checkpoints/bdslw131/robust_holistic_fusion_best.pt")
    cfg_path = Path("configs/bdslw131/proposed.yaml")
    if not (ckpt_path.is_file() and cfg_path.is_file()):
        pytest.skip("BdSL checkpoint or config not present")

    predictor = RealTimePredictor(model_path=ckpt_path, config_path=cfg_path)
    assert predictor.num_classes == 131
    assert predictor.bdsl_vocab is not None

    # Run inference with zero features
    test_seq = np.zeros((64, 1659), dtype=np.float32)
    res = predictor.predict_label(test_seq, confidence_threshold=0.0)

    assert "predicted_class" in res
    assert 0 <= res["predicted_class"] < 131
    assert "bengali_label" in res
    assert len(res["bengali_label"]) > 0
    assert "english_gloss" in res
    assert len(res["english_gloss"]) > 0
    assert "top_k" in res
    assert len(res["top_k"]) == 5

    for cand in res["top_k"]:
        assert "class_index" in cand
        assert "label" in cand
        assert "bengali_label" in cand
        assert "english_gloss" in cand
        assert "confidence" in cand


def test_synthetic_robustness_ci_fixtures() -> None:
    """Validate perturbation reproducibility and shape preservation using synthetic fixtures."""
    np.random.seed(42)
    seq = np.random.randn(32, 1659).astype(np.float32)

    # 1. Zero severity preservation
    clean_copy = seq.copy()
    jitter_zero = apply_spatial_noise(seq, "coordinate_jitter", severity=0.0, seed=42)
    np.testing.assert_allclose(clean_copy, jitter_zero, atol=1e-5)

    # 2. Shape preservation under landmark dropout
    drop_out = landmark_dropout(seq, severity=0.3, seed=42)
    assert drop_out.shape == seq.shape
    # Sparsity should increase
    assert np.count_nonzero(drop_out == 0) > np.count_nonzero(seq == 0)

    # 3. Temporal frame drop on 2D sequence drops frames along time dimension
    frame_dropped = apply_temporal_noise(seq, "frame_drop", severity=0.2, seed=42)
    assert frame_dropped.shape[1] == seq.shape[1]
    assert frame_dropped.shape[0] < seq.shape[0]
