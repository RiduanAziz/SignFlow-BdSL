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
from torch.utils.data import DataLoader

from sign_language.config import load_experiment_config
from sign_language.data import SignLanguageDataset, build_dataset_from_manifest
from sign_language.landmarks.pipeline import LandmarkPipeline
from sign_language.landmarks.schema import SCHEMA_VERSION
from sign_language.models import build_model
from sign_language.training.trainer import (
    evaluate_model,
)
from sign_language.utils.device import get_device
from sign_language.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_file_hash(path: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_raw_manifest_entries(manifest_path: Path | str) -> list[dict[str, Any]]:
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_splits_disjointness(
    train_entries: list[dict[str, Any]],
    val_entries: list[dict[str, Any]] | None = None,
    test_entries: list[dict[str, Any]] | None = None,
) -> None:
    """Validate sample ID uniqueness within splits and disjointness across splits."""
    def extract_ids(entries: list[dict[str, Any]]) -> list[str]:
        return [str(e.get("sample_id", "")) for e in entries if e.get("sample_id")]

    train_ids = extract_ids(train_entries)
    if len(train_ids) != len(set(train_ids)):
        from collections import Counter
        dupes = [k for k, v in Counter(train_ids).items() if v > 1]
        raise ValueError(f"Duplicate sample IDs detected in train manifest: {dupes[:10]}")

    train_set = set(train_ids)

    if val_entries is not None:
        val_ids = extract_ids(val_entries)
        if len(val_ids) != len(set(val_ids)):
            from collections import Counter
            dupes = [k for k, v in Counter(val_ids).items() if v > 1]
            raise ValueError(f"Duplicate sample IDs detected in validation manifest: {dupes[:10]}")
        val_set = set(val_ids)

        train_val_overlap = sorted(train_set & val_set)
        if train_val_overlap:
            raise ValueError(
                f"Data leakage detected! {len(train_val_overlap)} sample IDs exist in both "
                f"training and validation splits: {train_val_overlap[:10]}"
            )

    if test_entries is not None:
        test_ids = extract_ids(test_entries)
        if len(test_ids) != len(set(test_ids)):
            from collections import Counter
            dupes = [k for k, v in Counter(test_ids).items() if v > 1]
            raise ValueError(f"Duplicate sample IDs detected in test manifest: {dupes[:10]}")
        test_set = set(test_ids)

        train_test_overlap = sorted(train_set & test_set)
        if train_test_overlap:
            raise ValueError(
                f"Data leakage detected! {len(train_test_overlap)} sample IDs exist in both "
                f"training and test splits: {train_test_overlap[:10]}"
            )

        if val_entries is not None:
            val_test_overlap = sorted(val_set & test_set)
            if val_test_overlap:
                raise ValueError(
                    f"Data leakage detected! {len(val_test_overlap)} sample IDs exist in both "
                    f"validation and test splits: {val_test_overlap[:10]}"
                )


validate_split_disjointness = validate_splits_disjointness


def validate_manifest_split_selection(manifest_path: str | Path | None) -> Path:
    """Validate that the manifest selected for training is not the full discovery manifest."""
    if manifest_path is None:
        raise ValueError("Training manifest path cannot be None.")
    p = Path(manifest_path)
    cand_str = str(p).lower().replace("\\", "/")
    if "dataset_manifest" in cand_str or cand_str.endswith(".csv") or "full_manifest" in cand_str:
        raise ValueError(
            f"Full dataset manifest '{manifest_path}' cannot be used directly for training. "
            "Full discovery manifests contain held-out validation and test samples. "
            "Generate split manifests with 'python scripts/split_dataset.py' and pass "
            "--manifest data/manifests/bdslw131_train_manifest.json"
        )
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sign-language recognition model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--manifest", type=str, default=None, help="Path to training manifest")
    parser.add_argument("--validation-manifest", type=str, default=None, help="Path to validation manifest")
    parser.add_argument("--test-manifest", type=str, default=None, help="Path to test manifest for leakage validation")
    parser.add_argument("--landmarks-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--max-samples", "--limit", type=int, default=None, dest="limit", help="Max samples")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, auto)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--allow-no-validation", action="store_true", help="Allow running training without validation (smoke test mode only)")
    parser.add_argument("--max-missing-ratio", type=float, default=0.05, help="Max allowed missing feature ratio before failing")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without training")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.device is not None:
        config.device = args.device

    # 1. Strict split-aware manifest resolution
    # Detect if user passed full discovery manifest
    if args.manifest:
        validate_manifest_split_selection(args.manifest)

    # Resolve train manifest path
    manifest_candidates = [
        args.manifest,
        getattr(config.dataset, "train_manifest", None),
        "data/manifests/bdslw131_train_manifest.json",
        "data/manifests/train_manifest.json",
    ]
    train_manifest_path = None
    for cand in manifest_candidates:
        if cand and Path(cand).is_file():
            train_manifest_path = Path(cand)
            break

    if train_manifest_path is None:
        # Check if configured manifest is the full dataset manifest
        cfg_m = getattr(config.dataset, "manifest", None)
        if cfg_m and ("dataset_manifest" in cfg_m or "full_manifest" in cfg_m):
            raise FileNotFoundError(
                f"Training manifest not found. The configured dataset.manifest ('{cfg_m}') "
                "is a full discovery manifest and cannot be used for training. "
                "Run 'python scripts/split_dataset.py' to generate 'data/manifests/bdslw131_train_manifest.json'."
            )
        raise FileNotFoundError(
            "Training manifest not found. Checked candidates: "
            f"{[c for c in manifest_candidates if c]}. "
            "Run 'python scripts/split_dataset.py' or provide --manifest."
        )

    # Resolve validation manifest path
    val_candidates = [
        args.validation_manifest,
        getattr(config.dataset, "val_manifest", None),
        "data/manifests/bdslw131_val_manifest.json",
        "data/manifests/val_manifest.json",
        "data/manifests/validation_manifest.json",
    ]
    val_manifest_path = None
    for cand in val_candidates:
        if cand and Path(cand).is_file():
            val_manifest_path = Path(cand)
            break

    if val_manifest_path is None:
        if not args.allow_no_validation and getattr(config.evaluation, "primary_metric", None):
            raise FileNotFoundError(
                "Validation manifest not found. Checked candidates: "
                f"{[c for c in val_candidates if c]}. "
                "Training requires a validation split for checkpoint selection. "
                "Run 'python scripts/split_dataset.py' or pass --allow-no-validation for a smoke test."
            )
        logger.warning("No validation manifest found. Running in unvalidated smoke-test mode.")

    # Resolve test manifest path for leakage verification
    test_candidates = [
        args.test_manifest,
        getattr(config.dataset, "test_manifest", None),
        "data/manifests/bdslw131_test_manifest.json",
        "data/manifests/test_manifest.json",
    ]
    test_manifest_path = None
    for cand in test_candidates:
        if cand and Path(cand).is_file():
            test_manifest_path = Path(cand)
            break

    landmarks_dir = args.landmarks_dir or getattr(config.dataset, "root", "data/landmarks")
    checkpoint_dir_path = Path(args.checkpoint_dir or getattr(config, "checkpoint_dir", "checkpoints"))

    logger.info("Resolved Train Manifest: %s (SHA256: %s)", train_manifest_path, compute_file_hash(train_manifest_path)[:12])
    if val_manifest_path:
        logger.info("Resolved Validation Manifest: %s (SHA256: %s)", val_manifest_path, compute_file_hash(val_manifest_path)[:12])
    if test_manifest_path:
        logger.info("Resolved Test Manifest (leakage check): %s (SHA256: %s)", test_manifest_path, compute_file_hash(test_manifest_path)[:12])

    # 2. Strict ID disjointness & uniqueness validation
    train_raw = load_raw_manifest_entries(train_manifest_path)
    val_raw = load_raw_manifest_entries(val_manifest_path) if val_manifest_path else None
    test_raw = load_raw_manifest_entries(test_manifest_path) if test_manifest_path else None
    validate_splits_disjointness(train_raw, val_raw, test_raw)
    logger.info("Split disjointness verified cleanly: 0%% sample ID leakage detected.")

    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return

    set_seed(config.seed)
    device = get_device(config.device)
    logger.info("Using device: %s", device)

    # 3. Load train samples with feature accounting
    train_samples, train_accounting = build_dataset_from_manifest(
        manifest_path=train_manifest_path,
        landmarks_dir=landmarks_dir,
        allowed_classes=config.dataset.labels,
        max_missing_ratio=args.max_missing_ratio,
        return_accounting=True,
    )
    if args.limit:
        train_samples = train_samples[: args.limit]

    if not train_samples:
        raise ValueError(f"No valid training samples found in {train_manifest_path} and {landmarks_dir}")

    logger.info(
        "Training samples loaded: %d/%d (missing features: %d)",
        len(train_samples),
        train_accounting["allowed_records"],
        train_accounting["missing_count"],
    )

    # Determine vocabulary and label mapping
    if config.dataset.labels:
        class_names = list(config.dataset.labels)
    else:
        class_names = sorted({s.class_name for s in train_samples})

    label_to_index = {name: i for i, name in enumerate(class_names)}
    config.model.num_classes = len(class_names)

    train_dataset = SignLanguageDataset(train_samples, label_to_index=label_to_index)

    # Pipeline for feature normalization and temporal resampling
    raw_feature_dim = config.model.input_dim // 3 if config.model.input_dim >= 4977 else config.model.input_dim
    pipeline = LandmarkPipeline(
        feature_dim=raw_feature_dim,
        sequence_length=config.dataset.sequence_length,
    )

    def collate_with_pipeline(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        features = []
        masks = []
        labels = []
        sample_ids = []
        for item in batch:
            seq = item["landmarks"]
            norm_seq, norm_mask = pipeline.normalize_sequence_with_mask(
                seq, mask=item.get("mask"), include_dynamics=(config.model.input_dim >= 4977)
            )
            features.append(norm_seq)
            masks.append(norm_mask)
            labels.append(item["label"])
            sample_ids.append(item.get("sample_id", ""))

        return {
            "landmarks": torch.tensor(np.stack(features), dtype=torch.float32),
            "mask": torch.tensor(np.stack(masks), dtype=torch.float32),
            "label": torch.tensor(labels, dtype=torch.long),
            "sample_id": sample_ids,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.training.batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_with_pipeline,
    )

    # 4. Load validation samples
    val_loader = None
    if val_manifest_path:
        val_samples, val_accounting = build_dataset_from_manifest(
            manifest_path=val_manifest_path,
            landmarks_dir=landmarks_dir,
            allowed_classes=class_names,
            max_missing_ratio=args.max_missing_ratio,
            return_accounting=True,
        )
        if args.limit:
            val_samples = val_samples[: max(2, args.limit // 4)]
        if val_samples:
            val_dataset = SignLanguageDataset(val_samples, label_to_index=label_to_index)
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(config.training.batch_size, len(val_dataset)),
                shuffle=False,
                collate_fn=collate_with_pipeline,
            )
            logger.info(
                "Validation samples loaded: %d/%d (missing features: %d)",
                len(val_samples),
                val_accounting["allowed_records"],
                val_accounting["missing_count"],
            )

    # 5. Build Model
    model = build_model(config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    primary_metric = getattr(config.evaluation, "primary_metric", "macro_f1")
    is_minimizing = primary_metric in {"loss", "val_loss"}
    best_metric_val = float("inf") if is_minimizing else -float("inf")
    best_epoch = -1

    if args.resume and Path(args.resume).is_file():
        logger.info("Resuming training from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_metric_val = ckpt.get("best_metric_value", -float("inf"))
            best_epoch = ckpt.get("best_epoch", -1)
        else:
            model.load_state_dict(ckpt)

    checkpoint_dir = checkpoint_dir_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"{config.model.name}_best.pt"
    last_ckpt_path = checkpoint_dir / f"{config.model.name}.pt"

    # 6. Training Loop
    logger.info("Beginning training for %d epochs...", config.training.epochs)
    loss_history = []
    val_history = []

    train_manifest_hash = compute_file_hash(train_manifest_path)
    val_manifest_hash = compute_file_hash(val_manifest_path) if val_manifest_path else ""
    test_manifest_hash = compute_file_hash(test_manifest_path) if test_manifest_path else ""

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = batch["landmarks"].to(device)
            masks = batch["mask"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(inputs, mask=masks)
            loss = criterion(logits, targets)
            loss.backward()

            if config.training.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(1, len(train_loader))
        loss_history.append(epoch_loss)

        # Validation step
        val_metrics: dict[str, float] = {}
        if val_loader is not None:
            model.eval()
            val_logits, val_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["landmarks"].to(device)
                    masks = batch["mask"].to(device)
                    targets = batch["label"].to(device)
                    out = model(inputs, mask=masks)
                    val_logits.append(out.cpu())
                    val_targets.append(targets.cpu())
            if val_logits:
                val_metrics = evaluate_model(torch.cat(val_logits), torch.cat(val_targets))
                val_history.append(val_metrics)

        current_metric_val = float(val_metrics.get(primary_metric, val_metrics.get("macro_f1", 0.0))) if val_metrics else 0.0
        logger.info(
            "Epoch %02d/%02d - Train Loss: %.4f | Val Acc: %.4f | Val Macro-F1: %.4f",
            epoch + 1,
            config.training.epochs,
            epoch_loss,
            val_metrics.get("accuracy", 0.0),
            val_metrics.get("macro_f1", 0.0),
        )

        # Build checkpoint metadata payload
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "primary_metric": primary_metric,
            "best_metric_value": best_metric_val,
            "is_validation_selected": val_loader is not None and best_epoch >= 0,
            "train_manifest": str(train_manifest_path),
            "train_manifest_hash": train_manifest_hash,
            "val_manifest": str(val_manifest_path) if val_manifest_path else None,
            "val_manifest_hash": val_manifest_hash,
            "test_manifest": str(test_manifest_path) if test_manifest_path else None,
            "test_manifest_hash": test_manifest_hash,
            "config": config.to_dict(),
            "model_name": config.model.name,
            "input_dim": config.model.input_dim,
            "num_classes": len(class_names),
            "class_names": class_names,
            "label_to_index": label_to_index,
            "sequence_length": config.dataset.sequence_length,
            "feature_representation": config.dataset.feature_representation,
            "schema_version": SCHEMA_VERSION,
            "seed": config.seed,
        }

        # Save last checkpoint
        torch.save(checkpoint_payload, last_ckpt_path)

        # Save best checkpoint
        is_better = (current_metric_val < best_metric_val) if is_minimizing else (current_metric_val > best_metric_val)
        if val_loader is not None and is_better:
            best_metric_val = current_metric_val
            best_epoch = epoch
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["best_metric_value"] = best_metric_val
            checkpoint_payload["is_validation_selected"] = True
            torch.save(checkpoint_payload, best_ckpt_path)
            logger.info(
                "Saved new best model checkpoint to %s (epoch %d, %s=%.4f)",
                best_ckpt_path,
                epoch + 1,
                primary_metric,
                best_metric_val,
            )

    # If no validation set was provided, explicitly mark best checkpoint
    if val_loader is None:
        checkpoint_payload["is_validation_selected"] = False
        checkpoint_payload["selection_metric"] = "none"
        torch.save(checkpoint_payload, best_ckpt_path)
        logger.warning("No validation performed. Best checkpoint marked as unvalidated smoke-test artifact.")

    # 7. Save Experiment Metrics
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = {
        "loss_history": loss_history,
        "validation_history": val_history,
        "primary_metric": primary_metric,
        "best_epoch": best_epoch,
        "best_metric_value": best_metric_val,
        "is_validation_selected": val_loader is not None and best_epoch >= 0,
        "train_manifest": str(train_manifest_path),
        "train_manifest_hash": train_manifest_hash,
        "val_manifest": str(val_manifest_path) if val_manifest_path else None,
        "val_manifest_hash": val_manifest_hash,
        "class_count": len(class_names),
        "classes": class_names,
    }
    metrics_path = results_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    logger.info("Training complete. Results saved to %s", metrics_path)


if __name__ == "__main__":
    main()
