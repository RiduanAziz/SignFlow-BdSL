from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from sign_language.config import load_experiment_config
from sign_language.data import SignLanguageDataset, build_dataset_from_manifest
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.models import build_model
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
            f"Held-out evaluation requires a dedicated test split manifest (e.g. data/manifests/bdslw131_test_manifest.json), "
            f"but full dataset manifest '{candidate_path}' was supplied. Held-out evaluation on the full discovery manifest is prohibited to prevent data leakage."
        )

    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"Test manifest does not exist at {candidate_path}. "
            "Please generate canonical splits using: python scripts/split_dataset.py"
        )

    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained sign-language model on held-out test data.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", "--model-path", type=str, required=True, dest="checkpoint", help="Path to trained checkpoint (.pt)")
    parser.add_argument("--manifest", "--test-manifest", type=str, default=None, dest="manifest", help="Path to held-out test manifest")
    parser.add_argument("--landmarks-dir", type=str, default=None, help="Path to extracted landmarks directory")
    parser.add_argument("--output", "--output-json", type=str, default="results/evaluation_report.json", dest="output", help="Path to write evaluation report")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for evaluation")
    parser.add_argument("--max-samples", "--limit", type=int, default=None, dest="limit", help="Max test samples to evaluate")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device(args.device)

    # 1. Load Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        class_names = ckpt.get("class_names")
        saved_config = ckpt.get("config", {})
        seq_len = ckpt.get("sequence_length", 64)
        input_dim = ckpt.get("input_dim", 4977)
    else:
        state_dict = ckpt
        class_names = None
        saved_config = {}
        seq_len = 64
        input_dim = 4977

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

    # 2. Build Model & Load Weights
    model = build_model(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Resolve Held-Out Test Manifest & Landmarks Directory
    config_test_manifest = getattr(config.dataset, "test_manifest", None)
    resolved_manifest = resolve_test_manifest(args.manifest, config_test_manifest)

    resolved_landmarks_dir = args.landmarks_dir or getattr(config.dataset, "landmarks_dir", "data/landmarks")
    if not Path(resolved_landmarks_dir).is_dir():
        # Check alternative landmarks location
        alt_cand = Path("data/landmarks")
        if alt_cand.is_dir():
            resolved_landmarks_dir = str(alt_cand)

    logger.info("Evaluation Configuration:")
    logger.info("  Checkpoint:       %s", checkpoint_path)
    logger.info("  Test Manifest:    %s", resolved_manifest)
    logger.info("  Landmarks Dir:    %s", resolved_landmarks_dir)
    logger.info("  Model Classes:    %d", len(class_names))

    # 4. Load Test Samples
    test_samples = build_dataset_from_manifest(
        manifest_path=str(resolved_manifest),
        landmarks_dir=str(resolved_landmarks_dir),
        allowed_classes=class_names,
    )
    if args.limit:
        test_samples = test_samples[: args.limit]

    if not test_samples:
        raise ValueError(
            f"No valid test samples found in manifest {resolved_manifest} and directory {resolved_landmarks_dir}"
        )

    logger.info("Evaluating %d held-out test samples...", len(test_samples))

    raw_feature_dim = input_dim // 3 if input_dim >= 4977 else input_dim
    pipeline = LandmarkPipeline(feature_dim=raw_feature_dim, sequence_length=seq_len)

    test_dataset = SignLanguageDataset(test_samples, label_to_index=label_to_index)

    targets: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []
    all_logits: list[list[float]] = []

    with torch.no_grad():
        for item in test_dataset:
            target_idx = item["label"]
            seq = item["landmarks"]
            mask = item.get("mask")
            norm_seq, norm_mask = pipeline.normalize_sequence_with_mask(
                seq, mask=mask, include_dynamics=(input_dim >= 4977)
            )
            tensor_in = torch.tensor(norm_seq, dtype=torch.float32).unsqueeze(0).to(device)
            mask_in = torch.tensor(norm_mask, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(tensor_in, mask=mask_in)
            logits_np = logits.squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(logits_np))

            targets.append(target_idx)
            predictions.append(pred_idx)
            sample_ids.append(item["sample_id"])
            all_logits.append(logits_np.tolist())

    # 5. Compute Comprehensive Evaluation Metrics
    accuracy = float(accuracy_score(targets, predictions))

    # Top-5 accuracy
    top5_correct = 0
    for lgt, tgt in zip(all_logits, targets):
        top5_indices = np.argsort(lgt)[-5:]
        if tgt in top5_indices:
            top5_correct += 1
    top5_accuracy = float(top5_correct / len(targets)) if targets else 0.0

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )

    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        targets, predictions, labels=list(range(len(class_names))), zero_division=0
    )

    conf_mat = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))

    # Compute provenance hashes
    ckpt_hash = compute_file_sha256(checkpoint_path)
    manifest_hash = compute_file_sha256(resolved_manifest)
    vocab_cand = Path("configs/bdslw131/vocabulary.json")
    vocab_hash = compute_file_sha256(vocab_cand) if vocab_cand.is_file() else ""

    class_wise_failures = [
        {
            "class_id": i,
            "class_name": class_names[i],
            "recall": float(r_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support_per_class[i]),
        }
        for i in range(len(class_names))
        if support_per_class[i] > 0 and (r_per_class[i] < 0.5 or f1_per_class[i] < 0.5)
    ]

    report: dict[str, Any] = {
        "provenance": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": ckpt_hash,
            "test_manifest": str(resolved_manifest),
            "manifest_sha256": manifest_hash,
            "canonical_vocabulary_sha256": vocab_hash,
            "metric_conventions": {
                "macro_f1": "unweighted macro-average across all classes with zero_division=0",
                "top_k": "top-5 accuracy based on unnormalized model logits",
            },
        },
        "total_samples": len(targets),
        "num_classes": len(class_names),
        "class_names": class_names,
        "metrics": {
            "top1_accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(p_weighted),
            "weighted_recall": float(r_weighted),
            "weighted_f1": float(f1_weighted),
        },
        "per_class": {
            class_names[i]: {
                "precision": float(p_per_class[i]),
                "recall": float(r_per_class[i]),
                "f1": float(f1_per_class[i]),
                "support": int(support_per_class[i]),
            }
            for i in range(len(class_names))
        },
        "class_wise_failures": class_wise_failures,
        "confusion_matrix": conf_mat.tolist(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Export per-class metrics CSV
    per_class_csv_path = output_path.parent / f"{output_path.stem}_per_class.csv"
    with open(per_class_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1", "support"])
        for cname in class_names:
            m = report["per_class"][cname]
            writer.writerow([cname, m["precision"], m["recall"], m["f1"], m["support"]])

    # Export confusion matrix CSV
    conf_csv_path = output_path.parent / f"{output_path.stem}_confusion_matrix.csv"
    with open(conf_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class"] + class_names)
        for i, cname in enumerate(class_names):
            writer.writerow([cname] + conf_mat[i].tolist())

    logger.info("Evaluation Complete:")
    logger.info("  Total Samples:  %d", len(targets))
    logger.info("  Top-1 Accuracy: %.4f", accuracy)
    logger.info("  Top-5 Accuracy: %.4f", top5_accuracy)
    logger.info("  Macro-F1:       %.4f", float(f1_macro))
    logger.info("  Weighted-F1:    %.4f", float(f1_weighted))
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
