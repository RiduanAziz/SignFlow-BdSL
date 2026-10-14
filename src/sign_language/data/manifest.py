from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifestEntry:
    sample_id: str
    signer_id: str
    class_id: int
    class_name: str
    video_path: str
    num_frames: int
    fps: float
    duration: float
    split: str | None = None


def _validate_entry(entry: Mapping[str, Any]) -> None:
    required = {
        "sample_id",
        "signer_id",
        "class_id",
        "class_name",
        "video_path",
        "num_frames",
        "fps",
        "duration",
    }
    missing = sorted(required.difference(entry.keys()))
    if missing:
        raise ValueError(f"Manifest entry is missing required keys: {missing}")

    if not entry["sample_id"]:
        raise ValueError("sample_id cannot be empty")
    if not entry["signer_id"]:
        raise ValueError("signer_id cannot be empty")
    if not entry["class_name"]:
        raise ValueError("class_name cannot be empty")
    if entry["num_frames"] <= 0:
        raise ValueError("num_frames must be positive")
    if entry["fps"] <= 0:
        raise ValueError("fps must be positive")
    if entry["duration"] <= 0:
        raise ValueError("duration must be positive")


def build_manifest(entries: Iterable[Mapping[str, Any]]) -> list[DatasetManifestEntry]:
    manifest: list[DatasetManifestEntry] = []
    for entry in entries:
        _validate_entry(entry)
        manifest.append(
            DatasetManifestEntry(
                sample_id=str(entry["sample_id"]),
                signer_id=str(entry["signer_id"]),
                class_id=int(entry["class_id"]),
                class_name=str(entry["class_name"]),
                video_path=str(entry["video_path"]),
                num_frames=int(entry["num_frames"]),
                fps=float(entry["fps"]),
                duration=float(entry["duration"]),
                split=str(entry.get("split")) if entry.get("split") is not None else None,
            )
        )
    return manifest


def generate_dataset_report(manifest: Iterable[DatasetManifestEntry]) -> dict[str, Any]:
    entries = list(manifest)
    class_counts = Counter(item.class_name for item in entries)
    signer_counts = Counter(item.signer_id for item in entries)
    split_names = ["train", "validation", "test"]
    split_counts = {name: 0 for name in split_names}
    split_names_seen = set()

    for item in entries:
        split_key = (item.split or "unassigned").lower()
        if split_key in split_counts:
            split_counts[split_key] += 1
            split_names_seen.add(split_key)
        elif split_key != "unassigned":
            split_counts.setdefault(split_key, 0)
            split_counts[split_key] += 1
            split_names_seen.add(split_key)

    return {
        "sample_count": len(entries),
        "class_count": len(class_counts),
        "signer_count": len(signer_counts),
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "classes": sorted(class_counts.keys()),
        "class_distribution": dict(sorted(class_counts.items())),
        "signers": sorted(signer_counts.keys()),
        "split_distribution": {
            key: split_counts[key]
            for key in sorted(split_counts.keys())
            if key in split_names_seen or key in split_names
        },
        "total_frames": sum(item.num_frames for item in entries),
        "average_frames_per_sample": (
            round(sum(item.num_frames for item in entries) / len(entries), 2) if entries else 0.0
        ),
    }


def validate_signer_independent_split(
    train_manifest: Iterable[DatasetManifestEntry],
    val_manifest: Iterable[DatasetManifestEntry],
    test_manifest: Iterable[DatasetManifestEntry],
) -> bool:
    train_signers = {item.signer_id for item in train_manifest}
    val_signers = {item.signer_id for item in val_manifest}
    test_signers = {item.signer_id for item in test_manifest}

    all_disjoint = (
        train_signers.isdisjoint(val_signers)
        and train_signers.isdisjoint(test_signers)
        and val_signers.isdisjoint(test_signers)
    )
    non_empty = bool(train_signers) and bool(val_signers) and bool(test_signers)
    return all_disjoint and non_empty


def write_manifest_json(manifest: Iterable[DatasetManifestEntry], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(item) for item in manifest]
    import json

    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return output
