from __future__ import annotations

import torch

from sign_language.training import compute_classification_metrics, evaluate_model


def test_evaluation_metrics_are_computed_for_logits() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.2, 2.0],
            [1.8, 0.1],
            [0.3, 1.7],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1])

    metrics = compute_classification_metrics(logits, targets)
    assert metrics["accuracy"] > 0.9
    assert metrics["macro_f1"] > 0.9
    assert "loss" in metrics


def test_model_evaluation_runs_on_dataloader_output() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.1, 2.0],
        ]
    )
    targets = torch.tensor([0, 1])

    metrics = evaluate_model(logits, targets)
    assert metrics["accuracy"] == 1.0
    assert metrics["num_samples"] == 2
