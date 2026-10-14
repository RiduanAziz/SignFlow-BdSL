from __future__ import annotations

import torch


def get_device(device: str | None = None) -> torch.device:
    """Select a usable device, preferring CUDA when available."""
    if device is None:
        device = "auto"

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
