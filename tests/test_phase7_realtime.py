from __future__ import annotations

import numpy as np

from sign_language.api.inference import RealTimePredictor, create_prediction_payload


def test_prediction_payload_and_predictor_shape() -> None:
    predictor = RealTimePredictor()
    sample = np.ones((64, 4977), dtype=np.float32)

    payload = create_prediction_payload(sample)
    assert payload["sequence_length"] == 64
    assert payload["feature_dim"] == 4977

    logits = predictor.predict(sample)
    assert logits.shape == (predictor.num_classes,)
    assert np.isfinite(logits).all()
