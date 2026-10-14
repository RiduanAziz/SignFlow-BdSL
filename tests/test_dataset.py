from __future__ import annotations

from sign_language.data import (
    SignLanguageDataset,
    build_synthetic_dataset,
    signer_aware_split,
)


def test_dataset_shapes_and_splits() -> None:
    signers = [f"S{i:03d}" for i in range(1, 6)]
    samples = build_synthetic_dataset(signers=signers, samples_per_class=2, sequence_length=8, feature_dim=42)

    train, val, test = signer_aware_split(
        samples,
        train_signers={"S001", "S002", "S003"},
        val_signers={"S004"},
        test_signers={"S005"},
    )

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert {sample.signer_id for sample in train}.isdisjoint({sample.signer_id for sample in val})
    assert {sample.signer_id for sample in train}.isdisjoint({sample.signer_id for sample in test})

    dataset = SignLanguageDataset(samples)
    item = dataset[0]
    assert item["landmarks"].shape == (8, 42)
    assert item["label"] in range(len({s.label for s in samples}))
