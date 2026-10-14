from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


@dataclass
class SignSample:
    sample_id: str
    label: str
    signer_id: str
    landmarks: np.ndarray | str | Path
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def class_name(self) -> str:
        return self.label


class SignLanguageDataset(Dataset):
    """Dataset abstraction for sign samples with landmark sequences."""

    def __init__(
        self,
        samples: list[SignSample],
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        self.samples = list(samples)
        labels = sorted({sample.label for sample in self.samples})
        self.label_to_index = label_to_index or {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        mask = None
        if isinstance(sample.landmarks, (str, Path)):
            with np.load(str(sample.landmarks)) as loaded:
                landmarks = loaded["landmarks"].astype(np.float32)
                if "mask" in loaded:
                    mask = loaded["mask"].astype(np.float32)
        else:
            landmarks = np.asarray(sample.landmarks, dtype=np.float32)

        label_index = self.label_to_index[sample.label]
        return {
            "sample_id": sample.sample_id,
            "signer_id": sample.signer_id,
            "label": label_index,
            "label_name": sample.label,
            "landmarks": landmarks,
            "mask": mask,
            "metadata": sample.metadata,
        }

def build_dataset_from_manifest(
    manifest_path: str | Path,
    landmarks_dir: str | Path,
    allowed_classes: list[str] | None = None,
) -> list[SignSample]:
    import csv
    import json
    from pathlib import Path

    manifest_p = Path(manifest_path)
    if manifest_p.suffix.lower() == ".csv":
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = list(csv.DictReader(f))
    else:
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    landmarks_root = Path(landmarks_dir)
    samples = []

    for entry in manifest:
        label = entry.get("class_name") or entry.get("label", "")
        if allowed_classes and label not in allowed_classes:
            continue

        sample_id = entry["sample_id"]
        npz_path = landmarks_root / f"{sample_id}.npz"
        if not npz_path.exists():
            continue

        samples.append(
            SignSample(
                sample_id=sample_id,
                label=label,
                signer_id=entry.get("signer_id", "unknown"),
                landmarks=npz_path,
                metadata={
                    "num_frames": entry.get("num_frames", 1),
                    "fps": entry.get("fps", 0),
                    "split": entry.get("split"),
                },
            )
        )
    return samples


def build_synthetic_dataset(
    labels: list[str] | None = None,
    signers: list[str] | None = None,
    samples_per_class: int = 6,
    sequence_length: int = 16,
    feature_dim: int = 42,
) -> list[SignSample]:
    """Create a small synthetic dataset for testing model training and splitting."""
    labels = labels or ["HELLO", "THANKS", "PLEASE", "YES", "NO"]
    signers = signers or [f"S{i:03d}" for i in range(1, 21)]

    samples: list[SignSample] = []
    for label_idx, label in enumerate(labels):
        for signer_idx, signer in enumerate(signers):
            for session_idx in range(1, samples_per_class + 1):
                base = (label_idx + 1) * 3.0 + signer_idx * 0.7
                sequence = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
                landmarks = np.zeros((sequence_length, feature_dim), dtype=np.float32)
                for t in range(sequence_length):
                    landmarks[t] = base + sequence[t] + np.linspace(0.0, 0.2, feature_dim, dtype=np.float32)
                    landmarks[t] += (signer_idx % 3) * 0.05
                sample = SignSample(
                    sample_id=f"{signer}_{label}_{session_idx:02d}",
                    label=label,
                    signer_id=signer,
                    session_id=f"session_{session_idx:02d}",
                    landmarks=landmarks,
                    metadata={
                        "lighting": "normal",
                        "background": "indoor",
                        "speed": "normal",
                    },
                )
                samples.append(sample)
    return samples
