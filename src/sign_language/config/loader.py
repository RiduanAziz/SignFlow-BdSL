from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATASET_LABELS = ["HELLO", "THANKS", "PLEASE", "YES", "NO"]
DEFAULT_SEQUENCE_LENGTH = 16


@dataclass
class DatasetConfig:
    root: str = "data"
    labels: list[str] = field(default_factory=lambda: list(DEFAULT_DATASET_LABELS))
    signers: list[str] = field(default_factory=list)
    samples_per_class: int | None = 6
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    feature_representation: str = "hands"
    manifest: str | None = None
    train_manifest: str | None = None
    val_manifest: str | None = None
    test_manifest: str | None = None
    raw_root: str | None = None
    landmarks_dir: str | None = None
    num_classes: int | None = None


@dataclass
class ModelConfig:
    name: str = "mlp"
    input_dim: int = 42
    hidden_dim: int = 128
    num_classes: int = 5
    sequence_length: int = 16
    num_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.1
    embedding_dim: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    max_sequence_length: int = 32


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float | None = 1.0
    num_workers: int = 0


@dataclass
class EvaluationConfig:
    validation_split: str = "validation"
    test_split: str = "test"
    primary_metric: str = "macro_f1"


@dataclass
class ExperimentConfig:
    project_name: str = "sign-language-recognition"
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def num_classes(self) -> int:
        return len(self.dataset.labels) or self.model.num_classes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(data: dict[str, Any]) -> ExperimentConfig:
    dataset_data = dict(data.get("dataset", {}))
    model_data = dict(data.get("model", {}))
    training_data = dict(data.get("training", {}))
    evaluation_data = dict(data.get("evaluation", {}))

    dataset_labels = dataset_data.get("labels")
    if dataset_labels is None or not dataset_labels:
        dataset_data["labels"] = list(DEFAULT_DATASET_LABELS)
        dataset_labels = dataset_data["labels"]

    if "num_classes" not in model_data and dataset_labels:
        model_data["num_classes"] = len(dataset_labels)
    if dataset_data.get("sequence_length") is None:
        dataset_data["sequence_length"] = DEFAULT_SEQUENCE_LENGTH
    if "sequence_length" not in model_data:
        model_data["sequence_length"] = dataset_data["sequence_length"]
    if "sequence_length" in model_data and "max_sequence_length" not in model_data:
        model_data["max_sequence_length"] = model_data["sequence_length"]

    config = ExperimentConfig(
        project_name=str(data.get("project_name", ExperimentConfig.project_name)),
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "auto")),
        checkpoint_dir=str(data.get("checkpoint_dir", "checkpoints")),
        results_dir=str(data.get("results_dir", "results")),
        dataset=DatasetConfig(**dataset_data),
        model=ModelConfig(**model_data),
        training=TrainingConfig(**training_data),
        evaluation=EvaluationConfig(**evaluation_data),
    )
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    valid_models = {
        "baseline",
        "mlp",
        "lstm",
        "gru",
        "transformer",
        "temporal_transformer",
        "robust_holistic_fusion",
        "fusion",
        "proposed",
    }
    if config.model.name not in valid_models:
        raise ValueError(f"Unsupported model name: {config.model.name}")
    if config.model.input_dim <= 0 or config.model.num_classes <= 0:
        raise ValueError("model.input_dim and model.num_classes must be positive")
    if config.dataset.labels and len(config.dataset.labels) != config.model.num_classes:
        raise ValueError("dataset.labels and model.num_classes must agree")
    if config.dataset.sequence_length is not None and config.dataset.sequence_length <= 0:
        raise ValueError("dataset.sequence_length must be positive when provided")
    if config.model.sequence_length <= 0:
        raise ValueError("model.sequence_length must be positive")
    if config.training.epochs <= 0 or config.training.batch_size <= 0:
        raise ValueError("training.epochs and training.batch_size must be positive")
    if config.training.learning_rate <= 0 or config.training.weight_decay < 0:
        raise ValueError("training learning rate must be positive and weight decay non-negative")


def load_experiment_config(
    experiment_path: str | Path = "configs/base.yaml",
    base_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> ExperimentConfig:
    experiment_path = Path(experiment_path)
    root = (
        experiment_path.resolve().parents[2]
        if experiment_path.parent.name in {"experiments", "robustness", "bdslw131"}
        else Path.cwd()
    )
    base_path = Path(base_path) if base_path is not None else root / "configs/base.yaml"
    sibling_dataset = experiment_path.parent / "dataset.yaml"
    if dataset_path is None and sibling_dataset.is_file():
        dataset_path = sibling_dataset
    else:
        dataset_path = Path(dataset_path) if dataset_path is not None else root / "configs/dataset.yaml"

    merged: dict[str, Any] = {}
    paths_to_load = []
    for path in (base_path, dataset_path, experiment_path):
        resolved = path.resolve()
        if resolved not in [p.resolve() for p in paths_to_load]:
            paths_to_load.append(path)

    for path in paths_to_load:
        if path.exists():
            merged = _deep_merge(merged, _load_yaml(path))
    return _build_config(merged)
