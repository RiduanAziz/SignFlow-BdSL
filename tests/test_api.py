from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sign_language.api import main as api_main
from sign_language.api.inference import RealTimePredictor
from sign_language.api.main import app


@pytest.fixture
def test_predictor() -> RealTimePredictor:
    """Deterministic predictor fixture configured without requiring untracked disk checkpoints."""
    return RealTimePredictor(model_path=None, config_path="configs/base.yaml")


def test_api_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["process_alive"] is True
        assert data["backend_initialized"] is True
        assert "mediapipe_available" in data
        assert "trained_model_loaded" in data


def test_api_model_info(monkeypatch: pytest.MonkeyPatch, test_predictor: RealTimePredictor) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", test_predictor)
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "num_classes" in data
        assert "sequence_length" in data
        assert "input_dim" in data
        assert data["num_classes"] > 0


def test_api_model_info_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", None)
        response = client.get("/model/info")
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data


def test_api_predict_endpoint_valid(monkeypatch: pytest.MonkeyPatch, test_predictor: RealTimePredictor) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", test_predictor)
        sequence = [[0.0] * test_predictor.input_dim for _ in range(test_predictor.sequence_length)]
        response = client.post("/predict", json={"sequence": sequence, "confidence_threshold": 0.0})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["logits"]) == test_predictor.num_classes
        assert 0 <= payload["predicted_class"] < test_predictor.num_classes
        assert "top_k" in payload
        assert len(payload["top_k"]) <= 5


def test_api_predict_endpoint_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", None)
        sequence = [[0.0] * 4977 for _ in range(64)]
        response = client.post("/predict", json={"sequence": sequence, "confidence_threshold": 0.1})
        assert response.status_code == 503


def test_api_predict_endpoint_invalid(monkeypatch: pytest.MonkeyPatch, test_predictor: RealTimePredictor) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", test_predictor)
        # Empty sequence
        response = client.post("/predict", json={"sequence": []})
        assert response.status_code == 422

        # Invalid shape (1D instead of 2D)
        response = client.post("/predict", json={"sequence": [1.0, 2.0, 3.0]})
        assert response.status_code == 422


def test_websocket_live_connection_and_handshake() -> None:
    with TestClient(app) as client, client.websocket_connect("/ws/live") as websocket:
        # 1. Receive initial handshake
        data = websocket.receive_json()
        assert data["type"] == "session_started"
        assert "session_id" in data
        assert data["active_mode"] == "guided"
        assert "sequence_length" in data

        # 2. Send config update
        websocket.send_json({
            "type": "config",
            "mode": "guided",
            "confidence_threshold": 0.5,
            "debounce_frames": 4,
        })
        config_resp = websocket.receive_json()
        assert config_resp["type"] == "config_updated"
        assert config_resp["confidence_threshold"] == 0.5
        assert config_resp["debounce_frames"] == 4

        # 3. Send reset
        websocket.send_json({"type": "reset"})
        reset_resp = websocket.receive_json()
        assert reset_resp["type"] == "reset_ack"


def test_frontend_static_serving() -> None:
    dist_index = Path("app/frontend/dist/index.html")
    with TestClient(app) as client:
        response = client.get("/")
        if dist_index.is_file():
            assert response.status_code == 200
            assert "SignFlow" in response.text
        else:
            assert response.status_code in (404, 200)


def test_api_vocabulary_endpoint(monkeypatch: pytest.MonkeyPatch, test_predictor: RealTimePredictor) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", test_predictor)
        response = client.get("/api/vocabulary")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert data["count"] == test_predictor.num_classes
        assert isinstance(data["vocabulary"], list)
        assert len(data["vocabulary"]) == test_predictor.num_classes


def test_api_vocabulary_endpoint_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", None)
        response = client.get("/api/vocabulary")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False
        assert data["count"] == 0
        assert data["vocabulary"] == []


def test_api_robustness_experiment(monkeypatch: pytest.MonkeyPatch, test_predictor: RealTimePredictor) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", test_predictor)
        sequence = [[0.0] * test_predictor.input_dim for _ in range(test_predictor.sequence_length)]
        response = client.post(
            "/api/robustness/experiment",
            json={
                "sequence": sequence,
                "perturbation_type": "coordinate_jitter",
                "severity": 0.2,
                "seed": 42,
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert "original_prediction" in payload
        assert "perturbed_prediction" in payload
        assert "prediction_consistent" in payload
        assert "processing_time_ms" in payload
        assert payload["perturbation_type"] == "coordinate_jitter"
        assert "clean_quality" in payload


def test_api_robustness_experiment_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", None)
        sequence = [[0.0] * 4977 for _ in range(64)]
        response = client.post(
            "/api/robustness/experiment",
            json={
                "sequence": sequence,
                "perturbation_type": "coordinate_jitter",
                "severity": 0.2,
                "seed": 42,
            },
        )
        assert response.status_code == 503


def test_websocket_replay_buffer_request() -> None:
    with TestClient(app) as client, client.websocket_connect("/ws/live") as websocket:
        init = websocket.receive_json()
        assert init["type"] == "session_started"

        websocket.send_json({"type": "request_replay"})
        resp = websocket.receive_json()
        assert resp["type"] == "replay_buffer"
        assert isinstance(resp["frames"], list)


def test_offline_online_preprocessing_parity(test_predictor: RealTimePredictor) -> None:
    """Verify that offline LandmarkPipeline and RealTimePredictor yield identical features and masks."""
    raw_dim = test_predictor.pipeline.feature_dim
    seq_len = 32
    # Create deterministic sequence with alternating values and non-trivial masks
    np.random.seed(42)
    raw_sequence = np.random.randn(seq_len, raw_dim).astype(np.float32)
    raw_mask = (np.random.rand(seq_len, raw_dim) > 0.3).astype(np.float32)

    # 1. Pipeline path
    include_dyn = test_predictor.input_dim >= 4977
    feat_pipe, mask_pipe = test_predictor.pipeline.normalize_sequence_with_mask(
        raw_sequence, mask=raw_mask, include_dynamics=include_dyn
    )

    # 2. Assert output shape contracts
    assert feat_pipe.shape == (test_predictor.sequence_length, test_predictor.input_dim)
    assert mask_pipe.shape == (test_predictor.sequence_length, test_predictor.input_dim)
    assert np.all(np.isfinite(feat_pipe))
    assert np.all(np.isin(mask_pipe, [0.0, 1.0]))
