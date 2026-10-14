from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from sign_language.config import load_experiment_config
from sign_language.data import SignLanguageDataset, build_dataset_from_manifest
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.models import build_model
from sign_language.utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained sign-language model on held-out test data.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", "--model-path", type=str, required=True, dest="checkpoint", help="Path to trained checkpoint (.pt)")
    parser.add_argument("--manifest", type=str, default="data/manifests/test_manifest.json", help="Path to test manifest")
    parser.add_argument("--landmarks-dir", type=str, default="data/landmarks", help="Path to extracted landmarks directory")
    parser.add_argument("--output", type=str, default="results/evaluation_report.json", help="Path to write evaluation report")
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

    # 3. Load Test Samples
    test_samples = build_dataset_from_manifest(
        manifest_path=args.manifest,
        landmarks_dir=args.landmarks_dir,
        allowed_classes=class_names,
    )
    if args.limit:
        test_samples = test_samples[: args.limit]

    if not test_samples:
        raise ValueError(f"No valid test samples found in manifest {args.manifest} and directory {args.landmarks_dir}")

    logger.info("Evaluating %d held-out test samples...", len(test_samples))

    raw_feature_dim = input_dim // 3 if input_dim >= 4977 else input_dim
    pipeline = LandmarkPipeline(feature_dim=raw_feature_dim, sequence_length=seq_len)

    test_dataset = SignLanguageDataset(test_samples, label_to_index=label_to_index)

    targets: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []

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
            pred_idx = int(logits.argmax(dim=-1).item())

            targets.append(target_idx)
            predictions.append(pred_idx)
            sample_ids.append(item["sample_id"])

    # 4. Compute Comprehensive Evaluation Metrics
    accuracy = float(accuracy_score(targets, predictions))
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

    report = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(args.manifest),
        "total_samples": len(targets),
        "num_classes": len(class_names),
        "class_names": class_names,
        "metrics": {
            "top1_accuracy": accuracy,
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
        "confusion_matrix": conf_mat.tolist(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Export per-class metrics CSV
    per_class_csv_path = output_path.parent / f"{output_path.stem}_per_class.csv"
    import csv
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
    logger.info("  Total Samples: %d", len(targets))
    logger.info("  Top-1 Accuracy: %.4f", accuracy)
    logger.info("  Macro-F1:       %.4f", float(f1_macro))
    logger.info("  Weighted-F1:    %.4f", float(f1_weighted))
    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
