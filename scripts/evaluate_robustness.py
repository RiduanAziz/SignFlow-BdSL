from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from sign_language.config import load_experiment_config
from sign_language.data import SignLanguageDataset, build_dataset_from_manifest
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.models import build_model
from sign_language.robustness import (
    apply_spatial_noise,
    apply_temporal_noise,
    recognition_robustness_report,
)
from sign_language.utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hash of a file for provenance tracking."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resolve_test_manifest(
    manifest_arg: str | None,
    config_test_manifest: str | None = None,
) -> Path:
    """Resolve held-out test manifest, failing closed if full dataset manifest is passed."""
    candidate_path: Path | None = None

    if manifest_arg and manifest_arg != "data/manifests/test_manifest.json":
        candidate_path = Path(manifest_arg)
    elif config_test_manifest and Path(config_test_manifest).is_file():
        candidate_path = Path(config_test_manifest)
    elif Path("data/manifests/bdslw131_test_manifest.json").is_file():
        candidate_path = Path("data/manifests/bdslw131_test_manifest.json")
    elif Path("data/manifests/test_manifest.json").is_file():
        candidate_path = Path("data/manifests/test_manifest.json")
    else:
        candidate_path = Path(manifest_arg or "data/manifests/bdslw131_test_manifest.json")

    # Strict check: never evaluate on the full discovery manifest
    cand_str = str(candidate_path).lower().replace("\\", "/")
    if "dataset_manifest" in cand_str or cand_str.endswith(".csv"):
        raise ValueError(
            f"Held-out robustness evaluation requires a dedicated test split manifest (e.g. data/manifests/bdslw131_test_manifest.json), "
            f"but full dataset manifest '{candidate_path}' was supplied. Evaluation on full discovery manifest is prohibited to prevent data leakage."
        )

    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"Test manifest does not exist at {candidate_path}. "
            "Please generate canonical splits using: python scripts/split_dataset.py"
        )

    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model robustness under spatial and temporal disturbances.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config")
    parser.add_argument("--manifest", "--test-manifest", default=None, dest="manifest", help="Path to held-out test manifest")
    parser.add_argument("--landmarks-dir", default=None, help="Path to landmarks directory")
    parser.add_argument("--checkpoint", "--model-path", required=True, dest="checkpoint", help="Path to trained checkpoint (.pt)")
    parser.add_argument("--output", default="results/robustness/report.json", help="Output JSON report path")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--max-samples", "--limit", type=int, default=None, dest="limit", help="Max test samples")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = get_device(args.device)

    # 1. Load Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        class_names = ckpt.get("class_names")
        saved_config = ckpt.get("config", {})
        input_dim = ckpt.get("input_dim", 4977)
        seq_len = ckpt.get("sequence_length", 64)
    else:
        state_dict = ckpt
        class_names = None
        saved_config = {}
        input_dim = 4977
        seq_len = 64

    config = load_experiment_config(args.config)
    if saved_config and isinstance(saved_config, dict):
        if "model" in saved_config and isinstance(saved_config["model"], dict):
            for k, v in saved_config["model"].items():
                setattr(config.model, k, v)
        if "dataset" in saved_config and isinstance(saved_config["dataset"], dict):
            for k, v in saved_config["dataset"].items():
                setattr(config.dataset, k, v)

    if class_names is None:
        class_names = list(config.dataset.labels)

    label_to_index = {name: i for i, name in enumerate(class_names)}
    config.model.num_classes = len(class_names)
    config.model.input_dim = input_dim

    # 2. Build Model
    model = build_model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Raw landmark feature dimension (1659)
    raw_feature_dim = input_dim // 3 if input_dim >= 4977 else input_dim
    pipeline = LandmarkPipeline(feature_dim=raw_feature_dim, sequence_length=seq_len)

    # 3. Resolve Held-Out Test Manifest & Landmarks Directory
    config_test_manifest = getattr(config.dataset, "test_manifest", None)
    resolved_manifest = resolve_test_manifest(args.manifest, config_test_manifest)

    resolved_landmarks_dir = args.landmarks_dir or getattr(config.dataset, "landmarks_dir", "data/landmarks")
    if not Path(resolved_landmarks_dir).is_dir():
        alt_cand = Path("data/landmarks")
        if alt_cand.is_dir():
            resolved_landmarks_dir = str(alt_cand)

    logger.info("Robustness Evaluation Configuration:")
    logger.info("  Checkpoint:    %s", checkpoint_path)
    logger.info("  Test Manifest: %s", resolved_manifest)
    logger.info("  Landmarks Dir: %s", resolved_landmarks_dir)

    # 4. Load Held-Out Test Samples
    test_samples = build_dataset_from_manifest(
        manifest_path=str(resolved_manifest),
        landmarks_dir=str(resolved_landmarks_dir),
        allowed_classes=class_names,
    )
    if args.limit:
        test_samples = test_samples[: args.limit]

    if not test_samples:
        raise ValueError(f"No test samples found in {resolved_manifest} and {resolved_landmarks_dir}")

    logger.info("Evaluating robustness on %d held-out test samples...", len(test_samples))

    test_dataset = SignLanguageDataset(test_samples, label_to_index=label_to_index)
    loaded_samples = [
        {
            "landmarks": item["landmarks"],
            "mask": item.get("mask"),
            "label": item["label"],
            "sample_id": item["sample_id"],
        }
        for item in test_dataset
    ]

    # Evaluate clean test performance
    targets = [s["label"] for s in loaded_samples]
    clean_preds: list[int] = []

    with torch.no_grad():
        for sample in loaded_samples:
            norm_seq, norm_mask = pipeline.normalize_sequence_with_mask(
                sample["landmarks"], mask=sample["mask"], include_dynamics=(input_dim >= 4977)
            )
            t_in = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
            m_in = torch.tensor(norm_mask, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(t_in, mask=m_in)
            clean_preds.append(int(logits.argmax(dim=-1).item()))

    clean_acc = float(np.mean([p == t for p, t in zip(clean_preds, targets)]))
    logger.info("Clean Test Accuracy: %.4f", clean_acc)

    spatial_noises = ["coordinate_jitter", "translation", "scale", "landmark_dropout"]
    temporal_noises = ["frame_drop", "frame_duplicate", "sequence_truncate"]
    severities = [0.0, 0.1, 0.3, 0.5]

    robustness_records: list[dict] = []

    # 4. Evaluate Spatial Disturbances
    for noise_type in spatial_noises:
        for sev in severities:
            noisy_preds: list[int] = []
            with torch.no_grad():
                for sample in loaded_samples:
                    # Apply spatial perturbation to raw coordinates
                    if sample["mask"] is not None:
                        noisy_raw, noisy_mask = apply_spatial_noise(
                            sample["landmarks"], noise_type, severity=sev, seed=42, mask=sample["mask"]
                        )
                    else:
                        noisy_raw = apply_spatial_noise(sample["landmarks"], noise_type, severity=sev, seed=42)
                        noisy_mask = None
                    norm_seq, norm_mask = pipeline.normalize_sequence_with_mask(
                        noisy_raw, mask=noisy_mask, include_dynamics=(input_dim >= 4977)
                    )
                    t_in = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
                    m_in = torch.tensor(norm_mask, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = model(t_in, mask=m_in)
                    noisy_preds.append(int(logits.argmax(dim=-1).item()))

            record = recognition_robustness_report(
                targets=targets,
                clean_predictions=clean_preds,
                noisy_predictions=noisy_preds,
                class_names=class_names,
                noise_type=f"spatial_{noise_type}",
                severity=sev,
            )
            robustness_records.append(record)
            logger.info(
                "Spatial [%s, sev=%.1f] Noisy Acc: %.4f | Acc Degradation: %.4f",
                noise_type,
                sev,
                record["noisy"]["accuracy"],
                record["accuracy_degradation"],
            )

    # 5. Evaluate Temporal Disturbances
    for noise_type in temporal_noises:
        for sev in severities:
            noisy_preds: list[int] = []
            with torch.no_grad():
                for sample in loaded_samples:
                    # Apply temporal perturbation to frame sequence and mask
                    if sample["mask"] is not None:
                        noisy_raw, noisy_mask = apply_temporal_noise(
                            sample["landmarks"], noise_type, severity=sev, seed=42, mask=sample["mask"]
                        )
                    else:
                        noisy_raw = apply_temporal_noise(sample["landmarks"], noise_type, severity=sev, seed=42)
                        noisy_mask = None
                    norm_seq, norm_mask = pipeline.normalize_sequence_with_mask(
                        noisy_raw, mask=noisy_mask, include_dynamics=(input_dim >= 4977)
                    )
                    t_in = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
                    m_in = torch.tensor(norm_mask, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = model(t_in, mask=m_in)
                    noisy_preds.append(int(logits.argmax(dim=-1).item()))

            record = recognition_robustness_report(
                targets=targets,
                clean_predictions=clean_preds,
                noisy_predictions=noisy_preds,
                class_names=class_names,
                noise_type=f"temporal_{noise_type}",
                severity=sev,
            )
            robustness_records.append(record)
            logger.info(
                "Temporal [%s, sev=%.1f] Noisy Acc: %.4f | Acc Degradation: %.4f",
                noise_type,
                sev,
                record["noisy"]["accuracy"],
                record["accuracy_degradation"],
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "provenance": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": compute_file_sha256(checkpoint_path),
            "test_manifest": str(resolved_manifest),
            "manifest_sha256": compute_file_sha256(resolved_manifest),
            "dataset_modality": "still_images (BDSLW131)",
            "methodology_note": (
                "BDSLW131 consists of static sign images. Spatial noise evaluates geometric deformation "
                "tolerance on static landmarks. Temporal perturbations evaluate invariance under repeated-frame sequences."
            ),
        },
        "clean_accuracy": clean_acc,
        "sample_count": len(targets),
        "results": robustness_records,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info("Robustness evaluation finished. Report written to %s", output_path)


if __name__ == "__main__":
    main()
