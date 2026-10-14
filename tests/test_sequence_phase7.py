import numpy as np
import torch

from sign_language.data import SignLanguageDataset, build_dataloader, pad_or_truncate_with_mask
from sign_language.data.dataset import SignSample
from sign_language.models import GRUClassifier, LSTMClassifier, TemporalTransformerClassifier


def test_padding_returns_true_length_and_mask() -> None:
    sequence = np.ones((3, 2), dtype=np.float32)
    padded, mask, length = pad_or_truncate_with_mask(sequence, target_length=5)

    assert padded.shape == (5, 2)
    assert mask.tolist() == [True, True, True, False, False]
    assert length == 3


def test_dataloader_collates_variable_lengths() -> None:
    samples = [
        SignSample("one", "A", "S1", np.ones((2, 3), dtype=np.float32)),
        SignSample("two", "B", "S2", np.ones((4, 3), dtype=np.float32)),
    ]
    batch = next(iter(build_dataloader(SignLanguageDataset(samples), batch_size=2, shuffle=False)))

    assert batch["landmarks"].shape == (2, 4, 3)
    assert batch["lengths"].tolist() == [2, 4]
    assert batch["mask"].tolist() == [[True, True, False, False], [True, True, True, True]]


def test_temporal_models_accept_lengths_and_masks() -> None:
    values = torch.randn(2, 4, 3)
    lengths = torch.tensor([2, 4])
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

    assert LSTMClassifier(input_dim=3, hidden_dim=4, num_layers=1, num_classes=2)(values, lengths=lengths).shape == (2, 2)
    assert GRUClassifier(input_dim=3, hidden_dim=4, num_layers=1, num_classes=2)(values, mask=mask).shape == (2, 2)
    assert TemporalTransformerClassifier(input_dim=3, embedding_dim=4, num_heads=2, num_layers=1, ff_dim=8, num_classes=2, max_sequence_length=4)(values, mask=mask).shape == (2, 2)
