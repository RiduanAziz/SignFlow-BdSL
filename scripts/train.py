from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sign-language recognition model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--validation-manifest", type=str, default=None)
    parser.add_argument("--landmarks-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--max-samples", "--limit", type=int, default=None, dest="limit", help="Max samples")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, auto)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without training")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.device is not None:
        config.device = args.device

    manifest_path = args.manifest or getattr(config.dataset, "manifest", None) or "data/manifests/train_manifest.json"
    val_manifest_path = args.validation_manifest or "data/manifests/validation_manifest.json"
    landmarks_dir = args.landmarks_dir or getattr(config.dataset, "root", "data/landmarks")
    checkpoint_dir_path = Path(args.checkpoint_dir or getattr(config, "checkpoint_dir", "checkpoints"))

    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return

    set_seed(config.seed)
    device = get_device(config.device)
    logger.info("Using device: %s", device)

    # 1. Load train samples
    train_samples = build_dataset_from_manifest(
        manifest_path=manifest_path,
        landmarks_dir=landmarks_dir,
        allowed_classes=config.dataset.labels,
    )
    if args.limit:
        train_samples = train_samples[: args.limit]

    if not train_samples:
        raise ValueError(f"No valid training samples found in {args.manifest} and {args.landmarks_dir}")

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

    # 2. Load validation samples
    val_loader = None
    if Path(val_manifest_path).is_file():
        val_samples = build_dataset_from_manifest(
            manifest_path=val_manifest_path,
            landmarks_dir=landmarks_dir,
            allowed_classes=class_names,
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

    # 3. Build Model
    model = build_model(config)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0
    best_macro_f1 = -1.0

    if args.resume and Path(args.resume).is_file():
        logger.info("Resuming training from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_macro_f1 = ckpt.get("best_macro_f1", -1.0)
        else:
            model.load_state_dict(ckpt)

    checkpoint_dir = checkpoint_dir_path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"{config.model.name}_best.pt"
    last_ckpt_path = checkpoint_dir / f"{config.model.name}.pt"

    # 4. Training Loop
    logger.info("Beginning training for %d epochs...", config.training.epochs)
    loss_history = []
    val_history = []

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
        val_macro_f1 = 0.0
        val_metrics = {}
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
                val_macro_f1 = float(val_metrics.get("macro_f1", 0.0))
                val_history.append(val_metrics)

        logger.info(
            "Epoch %02d/%02d - Train Loss: %.4f | Val Acc: %.4f | Val Macro-F1: %.4f",
            epoch + 1,
            config.training.epochs,
            epoch_loss,
            val_metrics.get("accuracy", 0.0),
            val_macro_f1,
        )

        # Build checkpoint metadata payload
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_macro_f1": max(best_macro_f1, val_macro_f1),
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
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(checkpoint_payload, best_ckpt_path)
            logger.info("Saved new best model checkpoint to %s", best_ckpt_path)

    # If no validation set was provided, also write best checkpoint
    if val_loader is None or best_macro_f1 < 0:
        torch.save(checkpoint_payload, best_ckpt_path)

    # 5. Save Experiment Metrics
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = {
        "loss_history": loss_history,
        "validation_history": val_history,
        "best_macro_f1": best_macro_f1,
        "class_count": len(class_names),
        "classes": class_names,
    }
    metrics_path = results_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    logger.info("Training complete. Results saved to %s", metrics_path)


if __name__ == "__main__":
    main()
