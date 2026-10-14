from pathlib import Path

import pytest

from sign_language.config import load_experiment_config

ROOT = Path(__file__).parents[1]


def test_experiment_config_composes_dataset_and_model_settings() -> None:
    config = load_experiment_config(ROOT / "configs/experiments/lstm.yaml")

    assert config.model.name == "lstm"
    assert config.model.input_dim == 4977
    assert config.model.num_classes == len(config.dataset.labels) == 100
    assert config.dataset.sequence_length == 64


def test_yaml_model_override_changes_runtime_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "model:\n  name: gru\n  input_dim: 7\n  num_classes: 2\n"
        "dataset:\n  labels: [A, B]\n  sequence_length: 9\n",
        encoding="utf-8",
    )

    config = load_experiment_config(
        config_path,
        base_path=ROOT / "configs/base.yaml",
        dataset_path=ROOT / "configs/dataset.yaml",
    )

    assert config.model.name == "gru"
    assert config.model.input_dim == 7
    assert config.model.num_classes == 2
    assert config.dataset.sequence_length == 9


def test_config_rejects_label_class_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("model:\n  num_classes: 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="labels and model.num_classes"):
        load_experiment_config(
            config_path,
            base_path=ROOT / "configs/base.yaml",
            dataset_path=ROOT / "configs/dataset.yaml",
        )
