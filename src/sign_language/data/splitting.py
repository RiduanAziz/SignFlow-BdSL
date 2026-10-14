from __future__ import annotations

import collections
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from .dataset import SignSample


def validate_signer_assignments(
    all_signers: set[str],
    train_signers: set[str],
    val_signers: set[str],
    test_signers: set[str],
) -> None:
    overlaps = {
        "train_validation": train_signers & val_signers,
        "train_test": train_signers & test_signers,
        "validation_test": val_signers & test_signers,
    }
    overlapping = {name: sorted(signers) for name, signers in overlaps.items() if signers}
    if overlapping:
        raise ValueError(f"Signer overlap detected: {overlapping}")

    assigned = train_signers | val_signers | test_signers
    missing = sorted(all_signers - assigned)
    unknown = sorted(assigned - all_signers)
    if missing:
        raise ValueError(f"Signers {missing} were not assigned to any split")
    if unknown:
        raise ValueError(f"Unknown signers assigned to a split: {unknown}")
    if not train_signers or not val_signers or not test_signers:
        raise ValueError("Train, validation, and test splits must all contain signers")


def split_manifest_by_signer(
    manifest: list[Any],
    train_signers: set[str] | list[str] | None = None,
    val_signers: set[str] | list[str] | None = None,
    test_signers: set[str] | list[str] | None = None,
    seed: int = 42,
) -> tuple[list[Any], list[Any], list[Any]]:
    entries = list(manifest)
    if not entries:
        return [], [], []

    def _get(item: Any, key: str) -> str:
        if isinstance(item, dict):
            return str(item.get(key, ""))
        return str(getattr(item, key, ""))

    def _with_split(item: Any, split_val: str) -> Any:
        if isinstance(item, dict):
            copy_item = dict(item)
            copy_item["split"] = split_val
            return copy_item
        return replace(item, split=split_val)

    # Check for duplicate sample IDs
    sample_ids = [_get(item, "sample_id") for item in entries]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [sid for sid, count in collections.Counter(sample_ids).items() if count > 1]
        raise ValueError(f"Duplicate sample IDs detected in manifest: {duplicates[:10]}")

    all_signers = sorted({_get(item, "signer_id") for item in entries})
    train_signers = list(train_signers) if train_signers is not None else []
    val_signers = list(val_signers) if val_signers is not None else []
    test_signers = list(test_signers) if test_signers is not None else []

    if not train_signers and not val_signers and not test_signers:
        if len(all_signers) < 3:
            raise ValueError("At least three unique signers are required for a train/val/test split")
        shuffled_signers = all_signers[:]
        random.Random(seed).shuffle(shuffled_signers)
        train_count = max(1, int(len(shuffled_signers) * 0.7))
        val_count = max(1, int(len(shuffled_signers) * 0.15))
        if train_count + val_count >= len(shuffled_signers):
            train_count = len(shuffled_signers) - 2
            val_count = 1
        train_signers = shuffled_signers[:train_count]
        val_signers = shuffled_signers[train_count : train_count + val_count]
        test_signers = shuffled_signers[train_count + val_count :]

    train_signer_set = set(train_signers)
    val_signer_set = set(val_signers)
    test_signer_set = set(test_signers)
    validate_signer_assignments(
        set(all_signers), train_signer_set, val_signer_set, test_signer_set
    )

    train = [_with_split(item, "train") for item in entries if _get(item, "signer_id") in train_signer_set]
    val = [_with_split(item, "validation") for item in entries if _get(item, "signer_id") in val_signer_set]
    test = [_with_split(item, "test") for item in entries if _get(item, "signer_id") in test_signer_set]

    return train, val, test


def write_manifest_splits(
    train: list[Any],
    validation: list[Any],
    test: list[Any],
    output_dir: str | Path,
    prefix: str = "",
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    pfx = f"{prefix}_" if prefix and not prefix.endswith("_") else (prefix if prefix else "")
    outputs = {
        "train": output_root / f"{pfx}train_manifest.json",
        "validation": output_root / f"{pfx}val_manifest.json",
        "val": output_root / f"{pfx}val_manifest.json",
        "test": output_root / f"{pfx}test_manifest.json",
    }
    for name, entries in (("train", train), ("val", validation), ("test", test)):
        target_path = outputs[name]
        data = [entry.__dict__ if hasattr(entry, "__dict__") and not isinstance(entry, dict) else entry for entry in entries]
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    # For legacy backward compatibility when no prefix is used
    if not pfx:
        legacy_val = output_root / "validation_manifest.json"
        data_val = [entry.__dict__ if hasattr(entry, "__dict__") and not isinstance(entry, dict) else entry for entry in validation]
        with legacy_val.open("w", encoding="utf-8") as handle:
            json.dump(data_val, handle, ensure_ascii=False, indent=2)

    return outputs


def signer_aware_split(
    samples: list[SignSample],
    train_signers: set[str] | list[str] | None = None,
    val_signers: set[str] | list[str] | None = None,
    test_signers: set[str] | list[str] | None = None,
) -> tuple[list[SignSample], list[SignSample], list[SignSample]]:
    """Split a dataset by signer so samples from the same signer stay in only one split."""
    # Check for duplicate sample IDs
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [sid for sid, count in collections.Counter(sample_ids).items() if count > 1]
        raise ValueError(f"Duplicate sample IDs detected in dataset: {duplicates[:10]}")

    all_signers = sorted({sample.signer_id for sample in samples})

    if train_signers is None and val_signers is None and test_signers is None:
        if len(all_signers) < 3:
            raise ValueError("At least three unique signers are required for a train/val/test split")
        train_count = max(1, int(len(all_signers) * 0.7))
        val_count = max(1, int(len(all_signers) * 0.15))
        if train_count + val_count >= len(all_signers):
            train_count = len(all_signers) - 2
            val_count = 1
        train_signers = set(all_signers[:train_count])
        val_signers = set(all_signers[train_count : train_count + val_count])
        test_signers = set(all_signers[train_count + val_count :])
    else:
        train_signers = set(train_signers or [])
        val_signers = set(val_signers or [])
        test_signers = set(test_signers or [])

    validate_signer_assignments(
        set(all_signers), train_signers, val_signers, test_signers
    )

    train = [sample for sample in samples if sample.signer_id in train_signers]
    val = [sample for sample in samples if sample.signer_id in val_signers]
    test = [sample for sample in samples if sample.signer_id in test_signers]
    return train, val, test


def split_manifest_stratified(
    manifest: list[Any],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Deterministically split manifest records stratified by class label."""
    if not manifest:
        return [], [], []

    rng = random.Random(seed)
    class_groups: dict[str, list[Any]] = collections.defaultdict(list)

    for item in manifest:
        lbl = getattr(item, "class_name", None) or getattr(item, "label", None)
        if isinstance(item, dict):
            lbl = item.get("class_name") or item.get("label") or ""
        class_groups[str(lbl)].append(item)

    train: list[Any] = []
    val: list[Any] = []
    test: list[Any] = []

    for _, items in sorted(class_groups.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_train + n_val >= n >= 3:
            n_train = n - 2
            n_val = 1
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train : n_train + n_val])
        test.extend(shuffled[n_train + n_val :])

    return train, val, test


# Canonical aliases
split_dataset_stratified = split_manifest_stratified
split_dataset_by_signer = split_manifest_by_signer
