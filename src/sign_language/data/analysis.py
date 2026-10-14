from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np


def _sequence_length(value: Any) -> int:
    if hasattr(value, "shape"):
        arr = np.asarray(value)
        if arr.ndim >= 2:
            return int(arr.shape[0])
        if arr.ndim == 1:
            return 1
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def summarize_dataset(samples: Iterable[Any]) -> dict[str, float | int | list[str] | dict[str, int]]:
    """Compute sample-level summary statistics for a landmark dataset."""
    rows = list(samples)
    if not rows:
        return {
            "sample_count": 0,
            "signer_count": 0,
            "label_count": 0,
            "min_sequence_length": 0,
            "max_sequence_length": 0,
            "average_sequence_length": 0.0,
            "label_distribution": {},
            "signer_distribution": {},
        }

    label_counter: Counter[str] = Counter()
    signer_counter: Counter[str] = Counter()
    lengths: list[int] = []

    for row in rows:
        label = getattr(row, "label", row.get("label") if isinstance(row, dict) else None)
        signer = getattr(row, "signer_id", row.get("signer_id") if isinstance(row, dict) else None)
        seq = getattr(row, "landmarks", row.get("landmarks") if isinstance(row, dict) else None)

        if label is None:
            label = "unknown"
        if signer is None:
            signer = "unknown"

        label_counter[str(label)] += 1
        signer_counter[str(signer)] += 1
        lengths.append(_sequence_length(seq))

    average_length = round(float(sum(lengths)) / len(lengths), 2) if lengths else 0.0
    return {
        "sample_count": len(rows),
        "signer_count": len(signer_counter),
        "label_count": len(label_counter),
        "min_sequence_length": min(lengths),
        "max_sequence_length": max(lengths),
        "average_sequence_length": average_length,
        "label_distribution": dict(sorted(label_counter.items())),
        "signer_distribution": dict(sorted(signer_counter.items())),
    }
