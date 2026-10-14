from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import classification_metrics


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a model state dictionary: {path}")
    model.load_state_dict(state_dict)
    return model.to(device)


def evaluate_checkpoint(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["landmarks"].to(device)
            kwargs = {}
            if "lengths" in batch:
                kwargs["lengths"] = batch["lengths"].to(device)
            if "mask" in batch:
                kwargs["mask"] = batch["mask"].to(device)
            logits = model(inputs, **kwargs)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch["label"].tolist())
    return classification_metrics(targets, predictions, class_names=class_names)
