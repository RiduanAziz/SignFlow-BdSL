import torch

from sign_language.evaluation import classification_metrics, evaluate_checkpoint


def test_classification_metrics_include_per_class_and_confusion_matrix() -> None:
    report = classification_metrics([0, 1, 0], [0, 1, 1], ["A", "B"])

    assert report["accuracy"] == 2 / 3
    assert report["macro_f1"] >= 0.0
    assert report["confusion_matrix"] == [[1, 1], [0, 1]]
    assert set(report["per_class"]) == {"A", "B"}


class FixedModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return inputs[:, 0, :]


def test_evaluation_engine_uses_model_predictions() -> None:
    inputs = torch.tensor([[[2.0, 0.0]], [[0.0, 3.0]]])
    targets = torch.tensor([0, 1])

    class DictLoader:
        def __iter__(self):
            yield {"landmarks": inputs, "label": targets}

    report = evaluate_checkpoint(FixedModel(), DictLoader(), torch.device("cpu"), ["A", "B"])
    assert report["accuracy"] == 1.0
