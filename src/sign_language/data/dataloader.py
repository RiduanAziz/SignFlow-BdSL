from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SignLanguageDataset
from .preprocessing import pad_or_truncate_with_mask


def collate_sequences(batch: list[dict], target_length: int | None = None) -> dict[str, torch.Tensor | list[str]]:
    sequences = [item["landmarks"] for item in batch]
    length = target_length or max(sequence.shape[0] for sequence in sequences)
    padded = []
    masks = []
    lengths = []
    for sequence in sequences:
        values, mask, valid_length = pad_or_truncate_with_mask(sequence, length)
        padded.append(values)
        masks.append(mask)
        lengths.append(valid_length)
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "signer_id": [item["signer_id"] for item in batch],
        "landmarks": torch.from_numpy(np.stack(padded)),
        "mask": torch.from_numpy(np.stack(masks)),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }


def build_dataloader(dataset: SignLanguageDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create a basic DataLoader for landmark samples."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_sequences)
