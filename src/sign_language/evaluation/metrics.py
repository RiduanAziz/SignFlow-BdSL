from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    class_names: Sequence[str] | None = None,
) -> dict[str, object]:
    target_values = np.asarray(list(targets), dtype=np.int64)
    prediction_values = np.asarray(list(predictions), dtype=np.int64)
    if target_values.shape != prediction_values.shape:
        raise ValueError("targets and predictions must have the same shape")
    if target_values.size == 0:
        raise ValueError("Cannot evaluate an empty prediction set")

    labels = list(range(len(class_names))) if class_names is not None else sorted(
        set(target_values.tolist()) | set(prediction_values.tolist())
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        target_values,
        prediction_values,
        labels=labels,
        zero_division=0,
    )
    per_class = {}
    for index, label in enumerate(labels):
        name = class_names[label] if class_names is not None else str(label)
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    weighted = precision_recall_fscore_support(
        target_values,
        prediction_values,
        average="weighted",
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        target_values,
        prediction_values,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(target_values, prediction_values)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(target_values, prediction_values, labels=labels).tolist(),
        "num_samples": int(target_values.size),
    }
