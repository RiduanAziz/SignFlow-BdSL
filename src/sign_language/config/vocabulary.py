"""Canonical vocabulary loader and mapping manager for BDSLW131."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_VOCAB_PATH = Path("configs/bdslw131/vocabulary.json")


class BdSLVocabulary:
    """Encapsulates the 131 canonical BDSLW131 class mappings."""

    def __init__(self, vocab_path: str | Path | None = None) -> None:
        self.path = Path(vocab_path or DEFAULT_VOCAB_PATH).resolve()
        if not self.path.is_file():
            # Fallback relative to project root
            cand = Path(__file__).resolve().parents[3] / "configs" / "bdslw131" / "vocabulary.json"
            if cand.is_file():
                self.path = cand
            else:
                raise FileNotFoundError(f"Canonical BDSLW131 vocabulary not found at {self.path}")

        with open(self.path, "r", encoding="utf-8") as f:
            self._items: list[dict[str, Any]] = json.load(f)

        self._id_to_item: dict[int, dict[str, Any]] = {}
        self._label_to_id: dict[str, int] = {}
        self._bengali_to_id: dict[str, int] = {}
        self._english_to_id: dict[str, int] = {}

        for item in self._items:
            cid = int(item["class_id"])
            dlabel = str(item["dataset_label"])
            blabel = str(item["bengali_label"])
            egloss = str(item["english_gloss"])

            self._id_to_item[cid] = item
            self._label_to_id[dlabel] = cid
            self._bengali_to_id[blabel] = cid
            self._english_to_id[egloss.lower()] = cid

    @property
    def num_classes(self) -> int:
        return len(self._items)

    @property
    def items(self) -> list[dict[str, Any]]:
        return list(self._items)

    def get_by_id(self, class_id: int) -> dict[str, Any]:
        if class_id not in self._id_to_item:
            raise KeyError(f"Invalid class ID: {class_id}. Max allowed: {self.num_classes - 1}")
        return self._id_to_item[class_id]

    def get_id_by_label(self, label: str) -> int:
        if label in self._label_to_id:
            return self._label_to_id[label]
        if label in self._bengali_to_id:
            return self._bengali_to_id[label]
        lbl_lower = label.lower()
        if lbl_lower in self._english_to_id:
            return self._english_to_id[lbl_lower]
        raise KeyError(f"Unknown BDSLW131 label: '{label}'")

    def dataset_labels(self) -> list[str]:
        return [self._id_to_item[i]["dataset_label"] for i in range(self.num_classes)]

    def bengali_labels(self) -> list[str]:
        return [self._id_to_item[i]["bengali_label"] for i in range(self.num_classes)]

    def english_glosses(self) -> list[str]:
        return [self._id_to_item[i]["english_gloss"] for i in range(self.num_classes)]

    def to_dict(self) -> list[dict[str, Any]]:
        return self.items


_cached_vocab: BdSLVocabulary | None = None


def get_bdsl_vocabulary(vocab_path: str | Path | None = None) -> BdSLVocabulary:
    global _cached_vocab
    if _cached_vocab is None or vocab_path is not None:
        _cached_vocab = BdSLVocabulary(vocab_path)
    return _cached_vocab
