from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sign_language.api.inference import RealTimePredictor
from sign_language.config import load_experiment_config
from sign_language.data.dataset import SignSample
from sign_language.data.manifest import DatasetManifestEntry
from sign_language.data.splitting import signer_aware_split, split_manifest_by_signer
from sign_language.models import build_model
from sign_language.robustness import (
    apply_spatial_noise,
    apply_temporal_noise,
)


def test_signer_split_duplicate_sample_rejection() -> None:
    """Ensure duplicate sample IDs in dataset or manifest raise ValueError."""
    samples = [
        SignSample("dup_1", "hello", "S001", np.zeros((8, 10))),
        SignSample("dup_1", "hello", "S002", np.zeros((8, 10))),
        SignSample("unique_3", "world", "S003", np.zeros((8, 10))),
    ]
    with pytest.raises(ValueError, match="Duplicate sample IDs detected"):
        signer_aware_split(samples, train_signers={"S001"}, val_signers={"S002"}, test_signers={"S003"})

    entries = [
        DatasetManifestEntry("dup_entry", "S001", 0, "hello", "path1.mp4", 64, 30.0, 2.13),
        DatasetManifestEntry("dup_entry", "S002", 1, "world", "path2.mp4", 64, 30.0, 2.13),
        DatasetManifestEntry("uniq_entry", "S003", 2, "help", "path3.mp4", 64, 30.0, 2.13),
    ]
    with pytest.raises(ValueError, match="Duplicate sample IDs detected"):
        split_manifest_by_signer(entries, train_signers={"S001"}, val_signers={"S002"}, test_signers={"S003"})


def test_signer_split_overlap_rejection() -> None:
    """Ensure overlapping or unknown signer assignments are rejected."""
    samples = [
        SignSample("s1", "hello", "S001", np.zeros((8, 10))),
        SignSample("s2", "hello", "S002", np.zeros((8, 10))),
        SignSample("s3", "hello", "S003", np.zeros((8, 10))),
    ]
    # S002 assigned to both train and validation
    with pytest.raises(ValueError, match="Signer overlap detected"):
        signer_aware_split(samples, train_signers={"S001", "S002"}, val_signers={"S002"}, test_signers={"S003"})

    # S003 omitted from assignment
    with pytest.raises(ValueError, match="were not assigned to any split"):
        signer_aware_split(samples, train_signers={"S001"}, val_signers={"S002"}, test_signers=set())

    # Unknown signer S999 assigned (with all real signers present)
    with pytest.raises(ValueError, match="Unknown signers assigned"):
        signer_aware_split(samples, train_signers={"S001", "S003"}, val_signers={"S002"}, test_signers={"S999"})


def test_robustness_perturbations_non_mutating_and_severity_zero() -> None:
    """Ensure noise operators do not mutate original arrays and severity 0 is an exact identity."""
    seq = np.ones((16, 63), dtype=np.float32) * 5.0
    mask = np.ones((16, 63), dtype=np.float32)
    seq_orig = seq.copy()
    mask_orig = mask.copy()

    # Severity 0 identity test
    for s_type in ["coordinate_jitter", "translation", "scale", "landmark_dropout"]:
        out_seq, out_mask = apply_spatial_noise(seq, s_type, severity=0.0, seed=42, mask=mask)
        np.testing.assert_allclose(out_seq, seq_orig)
        np.testing.assert_allclose(out_mask, mask_orig)

    for t_type in ["frame_drop", "frame_duplicate", "sequence_truncate"]:
        out_seq, out_mask = apply_temporal_noise(seq, t_type, severity=0.0, seed=42, mask=mask)
        np.testing.assert_allclose(out_seq, seq_orig)
        np.testing.assert_allclose(out_mask, mask_orig)

    # Non-mutation with non-zero severity
    out_drop, out_drop_mask = apply_spatial_noise(seq, "landmark_dropout", severity=0.5, seed=42, mask=mask)
    # Original arrays must remain untouched
    np.testing.assert_allclose(seq, seq_orig)
    np.testing.assert_allclose(mask, mask_orig)
    # Output must have dropped some values and updated mask accordingly
    assert (out_drop == 0.0).any()
    assert (out_drop_mask == 0.0).any()


def test_checkpoint_reconstruction_from_metadata(tmp_path: Path) -> None:
    """Verify that model parameters are restored from saved checkpoint config."""
    config = load_experiment_config("configs/base.yaml")
    config.model.name = "temporal_transformer"
    config.model.num_classes = 12
    config.model.input_dim = 4977
    config.model.num_layers = 3  # Custom non-default layer count

    model = build_model(config)
    ckpt_path = tmp_path / "custom_model.pt"

    payload = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "class_names": [f"word_{i}" for i in range(12)],
        "input_dim": 4977,
        "sequence_length": 64,
        "model_name": "temporal_transformer",
    }
    torch.save(payload, ckpt_path)

    # Instantiate predictor pointing to base.yaml (which has num_layers=2 by default)
    predictor = RealTimePredictor(model_path=str(ckpt_path), config_path="configs/base.yaml")
    assert predictor.num_classes == 12
    assert predictor.class_names == [f"word_{i}" for i in range(12)]
    # Check that model successfully loaded state dict without dimension/layer mismatch
    assert predictor.model is not None


def test_missing_checkpoint_raises_filenotfound() -> None:
    """Confirm RealTimePredictor refuses to initialize with non-existent checkpoints."""
    with pytest.raises(FileNotFoundError, match="Required checkpoint file does not exist"):
        RealTimePredictor(model_path="models/completely_nonexistent_checkpoint.pt")
