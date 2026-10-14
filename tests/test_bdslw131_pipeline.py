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


def test_deterministic_split_dataset_canonical_paths_and_disjointness(tmp_path: Path) -> None:
    """End-to-end test verifying canonical split filenames, mutual disjointness, and determinism."""
    from sign_language.data.splitting import split_dataset_stratified, write_manifest_splits

    # 1. Create synthetic manifest records across 5 classes
    classes = ["অপেক্ষা করো_Wait", "অফিস_Office", "আইন_Law", "আজ_Today", "আট_Eight"]
    synthetic_records = []
    sample_idx = 0
    for cls in classes:
        for i in range(20):
            sample_idx += 1
            synthetic_records.append({
                "sample_id": f"synth_{sample_idx:04d}",
                "filepath": f"{cls}/img_{i:03d}.jpg",
                "label": cls,
                "class_id": classes.index(cls),
                "is_valid": 1,
            })

    # 2. Run stratified split with seed 42
    train1, val1, test1 = split_dataset_stratified(
        synthetic_records,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    _ = write_manifest_splits(train1, val1, test1, output_dir=tmp_path / "run1", prefix="bdslw131_")

    # Verify canonical paths
    assert (tmp_path / "run1" / "bdslw131_train_manifest.json").is_file()
    assert (tmp_path / "run1" / "bdslw131_val_manifest.json").is_file()
    assert (tmp_path / "run1" / "bdslw131_test_manifest.json").is_file()

    train_ids = {s["sample_id"] for s in train1}
    val_ids = {s["sample_id"] for s in val1}
    test_ids = {s["sample_id"] for s in test1}

    # Mutual disjointness assertion
    assert len(train_ids & val_ids) == 0, "Train and val sets overlap!"
    assert len(train_ids & test_ids) == 0, "Train and test sets overlap!"
    assert len(val_ids & test_ids) == 0, "Val and test sets overlap!"

    # Full accounting assertion
    all_split_ids = train_ids | val_ids | test_ids
    original_ids = {s["sample_id"] for s in synthetic_records}
    assert all_split_ids == original_ids, "Not all samples accounted for in split output"

    # Class coverage assertion
    assert len({s["label"] for s in train1}) == len(classes)
    assert len({s["label"] for s in val1}) == len(classes)
    assert len({s["label"] for s in test1}) == len(classes)

    # 3. Determinism check: running with same seed yields identical sample sets
    train2, val2, test2 = split_dataset_stratified(
        synthetic_records,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    assert {s["sample_id"] for s in train2} == train_ids
    assert {s["sample_id"] for s in val2} == val_ids
    assert {s["sample_id"] for s in test2} == test_ids


def test_signer_based_split_disjointness_and_canonical_naming(tmp_path: Path) -> None:
    """Verify that signer-based splitting produces canonical filenames and strictly disjoint signers."""
    from sign_language.data.splitting import split_dataset_by_signer, write_manifest_splits

    synthetic_records = []
    signers = ["signer_01", "signer_02", "signer_03", "signer_04", "signer_05", "signer_06"]
    classes = ["অফিস_Office", "আইন_Law"]

    for s_idx, signer in enumerate(signers):
        for c_idx, cls in enumerate(classes):
            for i in range(5):
                synthetic_records.append({
                    "sample_id": f"sample_{s_idx}_{c_idx}_{i}",
                    "filepath": f"{cls}/img_{s_idx}_{i}.jpg",
                    "label": cls,
                    "signer_id": signer,
                    "is_valid": 1,
                })

    train_s, val_s, test_s = split_dataset_by_signer(
        synthetic_records,
        seed=42,
    )
    _ = write_manifest_splits(train_s, val_s, test_s, output_dir=tmp_path / "signer_run", prefix="bdslw131_")

    assert (tmp_path / "signer_run" / "bdslw131_train_manifest.json").is_file()
    assert (tmp_path / "signer_run" / "bdslw131_val_manifest.json").is_file()
    assert (tmp_path / "signer_run" / "bdslw131_test_manifest.json").is_file()

    train_signers = {s["signer_id"] for s in train_s}
    val_signers = {s["signer_id"] for s in val_s}
    test_signers = {s["signer_id"] for s in test_s}

    assert len(train_signers & val_signers) == 0
    assert len(train_signers & test_signers) == 0
    assert len(val_signers & test_signers) == 0


def test_training_fails_closed_on_full_discovery_manifest() -> None:
    """Verify that training fails closed with a clear error if the full discovery manifest is passed."""
    import scripts.train as train_script

    with pytest.raises(ValueError) as excinfo:
        train_script.validate_manifest_split_selection("data/manifests/bdslw131_dataset_manifest.csv")
    assert "Full dataset manifest" in str(excinfo.value)
    assert "cannot be used directly for training" in str(excinfo.value)


def test_training_detects_deliberate_split_leakage() -> None:
    """Verify that training startup detects any sample overlap between train, val, and test manifests."""
    import scripts.train as train_script

    train_samples = [{"sample_id": "sample_001"}, {"sample_id": "sample_002"}]
    val_samples = [{"sample_id": "sample_003"}]
    # Deliberate contamination: test manifest contains sample_002 from train
    test_samples = [{"sample_id": "sample_002"}, {"sample_id": "sample_004"}]

    with pytest.raises(ValueError) as excinfo:
        train_script.validate_split_disjointness(train_samples, val_samples, test_samples)
    assert "Data leakage detected" in str(excinfo.value)
    assert "sample_002" in str(excinfo.value)


def test_inspect_dataset_preserves_canonical_vocabulary_and_transliteration(tmp_path: Path) -> None:
    """Verify that candidate vocabulary generation does not overwrite canonical vocabulary without explicit flag."""
    import scripts.inspect_bdslw131_dataset as inspect_mod

    # Create dummy raw class folder
    raw_dir = tmp_path / "raw"
    c1 = raw_dir / "অপেক্ষা করো_Wait"
    c1.mkdir(parents=True)

    cand_vocab_path = tmp_path / "results" / "candidate_vocab.json"
    canonical_vocab_path = tmp_path / "configs" / "canonical_vocab.json"

    # Pre-populate canonical vocabulary with verified transliteration
    canonical_vocab_path.parent.mkdir(parents=True)
    initial_canonical = [{
        "class_id": 0,
        "dataset_label": "অপেক্ষা করো_Wait",
        "bengali_label": "অপেক্ষা করো",
        "english_gloss": "Wait",
        "transliteration": "opekkha koro",
        "source": "manual_curation",
    }]
    with open(canonical_vocab_path, "w", encoding="utf-8") as f:
        json.dump(initial_canonical, f)

    # 1. Running inspection WITHOUT --update-canonical-vocabulary must NOT modify canonical file
    inspect_mod.inspect_bdslw131(
        raw_dir=raw_dir,
        manifest_output=tmp_path / "manifest.csv",
        report_output=tmp_path / "report.json",
        distribution_output=tmp_path / "dist.csv",
        candidate_vocabulary_output=cand_vocab_path,
        canonical_vocabulary_path=None,  # No update requested
        check_images=False,
    )

    assert cand_vocab_path.is_file()
    with open(cand_vocab_path, "r", encoding="utf-8") as f:
        cand_data = json.load(f)
    # Candidate vocabulary transliteration is empty, not conflated with english_gloss
    assert cand_data[0]["transliteration"] == ""
    assert cand_data[0]["english_gloss"] == "Wait"

    # Canonical vocabulary remains untouched
    with open(canonical_vocab_path, "r", encoding="utf-8") as f:
        curr_canonical = json.load(f)
    assert curr_canonical[0]["transliteration"] == "opekkha koro"

    # 2. Running WITH --update-canonical-vocabulary preserves manually curated transliteration and creates backup
    inspect_mod.inspect_bdslw131(
        raw_dir=raw_dir,
        manifest_output=tmp_path / "manifest.csv",
        report_output=tmp_path / "report.json",
        distribution_output=tmp_path / "dist.csv",
        candidate_vocabulary_output=cand_vocab_path,
        canonical_vocabulary_path=canonical_vocab_path,
        check_images=False,
    )

    assert canonical_vocab_path.with_suffix(".json.bak").is_file()
    with open(canonical_vocab_path, "r", encoding="utf-8") as f:
        updated_canonical = json.load(f)
    # Preserved curated transliteration
    assert updated_canonical[0]["transliteration"] == "opekkha koro"
