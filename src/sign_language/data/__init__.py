"""Canonical dataset abstractions, manifests, and split utilities."""

from .analysis import summarize_dataset
from .dataloader import build_dataloader, collate_sequences
from .dataset import (
    SignLanguageDataset,
    SignSample,
    build_dataset_from_manifest,
    build_synthetic_dataset,
)
from .ingestion import (
    DatasetValidationError,
    DatasetValidationReport,
    build_video_manifest,
    discover_video_paths,
    read_metadata,
    validate_metadata_rows,
)
from .manifest import (
    DatasetManifestEntry,
    build_manifest,
    generate_dataset_report,
    validate_signer_independent_split,
    write_manifest_json,
)
from .preprocessing import (
    pad_or_truncate_sequence,
    pad_or_truncate_with_mask,
    sequence_to_tensor,
    temporal_augmentation,
)
from .splitting import (
    signer_aware_split,
    split_manifest_by_signer,
    split_manifest_stratified,
    validate_signer_assignments,
    write_manifest_splits,
)

__all__ = [
    "DatasetManifestEntry",
    "DatasetValidationError",
    "DatasetValidationReport",
    "SignLanguageDataset",
    "SignSample",
    "build_dataloader",
    "build_dataset_from_manifest",
    "build_manifest",
    "build_synthetic_dataset",
    "build_video_manifest",
    "collate_sequences",
    "discover_video_paths",
    "generate_dataset_report",
    "pad_or_truncate_sequence",
    "pad_or_truncate_with_mask",
    "read_metadata",
    "sequence_to_tensor",
    "signer_aware_split",
    "split_manifest_by_signer",
    "split_manifest_stratified",
    "summarize_dataset",
    "temporal_augmentation",
    "validate_metadata_rows",
    "validate_signer_assignments",
    "validate_signer_independent_split",
    "write_manifest_json",
    "write_manifest_splits",
]
