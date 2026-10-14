from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sign_language.api import main as api_main
from sign_language.api.main import SessionState, app
from sign_language.landmarks.schema import (
    StructuredLandmarkFrame,
    compute_landmark_bounding_box,
)


def _make_dummy_jpeg() -> str:
    """Generate a 64x64 valid JPEG image encoded as base64 data URI."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[16:48, 16:48] = [200, 180, 160]
    _, buffer = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def test_session_state_min_frames_for_static_classifier() -> None:
    """Verify SessionState eliminates the 10-frame buffer contradiction for static models."""
    state_static = SessionState(seq_len=1)
    assert state_static.min_sequence_frames == 1
    assert state_static.sequence_buffer.maxlen == 1
    assert state_static.mask_buffer.maxlen == 1

    state_temporal = SessionState(seq_len=32)
    assert state_temporal.min_sequence_frames == 10
    assert state_temporal.sequence_buffer.maxlen == 32
    assert state_temporal.mask_buffer.maxlen == 32


def test_bounding_box_computation() -> None:
    """Verify compute_landmark_bounding_box produces clamped, valid boxes."""
    coords = np.array(
        [[0.2, 0.3, 0.0], [0.6, 0.8, 0.0], [0.4, 0.5, 0.0]], dtype=np.float32
    )
    mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    box = compute_landmark_bounding_box(coords, mask, padding=0.0)
    assert box is not None
    assert abs(box["xmin"] - 0.2) < 1e-4
    assert abs(box["ymin"] - 0.3) < 1e-4
    assert abs(box["xmax"] - 0.6) < 1e-4
    assert abs(box["ymax"] - 0.8) < 1e-4
    assert abs(box["width"] - 0.4) < 1e-4
    assert abs(box["height"] - 0.5) < 1e-4

    # All invalid points must yield None
    invalid_mask = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert compute_landmark_bounding_box(coords, invalid_mask) is None
    assert compute_landmark_bounding_box(np.zeros((0, 3)), np.zeros(0)) is None


def test_single_frame_static_inference_reaches_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a motionless valid hand triggers prediction on a sequence_length=1 model."""
    mock_predictor = MagicMock()
    mock_predictor.sequence_length = 1
    mock_predictor.input_dim = 1659
    mock_predictor.feature_dim = 1659
    mock_predictor.num_classes = 131
    mock_predictor.class_names = [f"word_{i}" for i in range(131)]
    mock_predictor.config.model.name = "robust_holistic_fusion"
    mock_predictor.bdsl_vocab = None
    mock_predictor.predict_label.return_value = {
        "predicted_label": "শব্দ_১",
        "bengali_label": "শব্দ_১",
        "english_gloss": "word_1",
        "transliteration": "shobdo_1",
        "predicted_class": 0,
        "confidence": 0.92,
        "meets_threshold": True,
        "top_k": [{"class_index": 0, "label": "শব্দ_১", "confidence": 0.92}],
    }

    coords = np.zeros((553, 3), dtype=np.float32)
    mask = np.zeros(553, dtype=np.float32)
    # Right hand: point indices 21..42
    coords[21:42, :] = 0.5
    mask[21:42] = 1.0

    mock_extractor = MagicMock()
    mock_extractor.representation = "holistic"
    mock_extractor.feature_dim = 1659
    mock_extractor._landmarker = MagicMock()
    mock_extractor.extract_structured_frame.return_value = StructuredLandmarkFrame(
        coordinates=coords,
        point_mask=mask,
        timestamp_ms=1000.0,
    )

    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", mock_predictor)
        monkeypatch.setattr(api_main, "_extractor", mock_extractor)

        with client.websocket_connect("/ws/live") as ws:
            init = ws.receive_json()
            assert init["type"] == "session_started"
            assert init["sequence_length"] == 1

            jpeg_data = _make_dummy_jpeg()
            ws.send_json({"type": "frame", "data": jpeg_data, "frame_id": 101, "timestamp_ms": 1000})

            messages = []
            for _ in range(5):
                msg = ws.receive_json()
                messages.append(msg)
                if msg.get("type") == "frame_processed":
                    break

            msg_types = [m["type"] for m in messages]
            assert "hand_landmarks" in msg_types
            assert "live_prediction" in msg_types
            assert "frame_processed" in msg_types

            # Verify frame_id preservation
            hand_msg = next(m for m in messages if m["type"] == "hand_landmarks")
            assert hand_msg["frame_id"] == 101
            assert hand_msg["right_hand"] is True
            assert hand_msg["right_box"] is not None

            pred_msg = next(m for m in messages if m["type"] == "live_prediction")
            assert pred_msg["frame_id"] == 101
            assert pred_msg["predicted_label"] == "শব্দ_১"
            assert pred_msg["confidence"] == 0.92

            ack_msg = next(m for m in messages if m["type"] == "frame_processed")
            assert ack_msg["frame_id"] == 101
            assert ack_msg["success"] is True


def test_motion_gate_does_not_block_stationary_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify static model predicts even when consecutive frames have zero movement."""
    mock_predictor = MagicMock()
    mock_predictor.sequence_length = 1
    mock_predictor.input_dim = 1659
    mock_predictor.feature_dim = 1659
    mock_predictor.num_classes = 131
    mock_predictor.class_names = ["বই"] * 131
    mock_predictor.config.model.name = "baseline"
    mock_predictor.bdsl_vocab = None
    mock_predictor.predict_label.return_value = {
        "predicted_label": "বই",
        "bengali_label": "বই",
        "english_gloss": "book",
        "transliteration": "boi",
        "predicted_class": 0,
        "confidence": 0.88,
        "meets_threshold": True,
        "top_k": [{"class_index": 0, "label": "বই", "confidence": 0.88}],
    }

    coords = np.zeros((553, 3), dtype=np.float32)
    mask = np.zeros(553, dtype=np.float32)
    # Left hand: point indices 0..21
    coords[0:21, :] = 0.5
    mask[0:21] = 1.0

    mock_extractor = MagicMock()
    mock_extractor.representation = "holistic"
    mock_extractor.feature_dim = 1659
    mock_extractor._landmarker = MagicMock()
    mock_extractor.extract_structured_frame.return_value = StructuredLandmarkFrame(
        coordinates=coords,
        point_mask=mask,
        timestamp_ms=1000.0,
    )

    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", mock_predictor)
        monkeypatch.setattr(api_main, "_extractor", mock_extractor)

        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()  # init

            # Send identical stationary frame twice
            for fid in [1, 2]:
                ws.send_json({"type": "frame", "data": _make_dummy_jpeg(), "frame_id": fid})
                saw_prediction = False
                for _ in range(5):
                    msg = ws.receive_json()
                    if msg.get("type") == "live_prediction":
                        saw_prediction = True
                    if msg.get("type") == "frame_processed":
                        break
                assert saw_prediction, f"Stationary hand frame {fid} must not be blocked by motion gate"


def test_missing_model_reports_unavailable_honestly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that when no trained model is loaded, STATE_G_MODEL_UNAVAILABLE is reported."""
    coords = np.zeros((553, 3), dtype=np.float32)
    mask = np.zeros(553, dtype=np.float32)
    coords[21:42, :] = 0.5
    mask[21:42] = 1.0

    mock_extractor = MagicMock()
    mock_extractor.representation = "holistic"
    mock_extractor.feature_dim = 1659
    mock_extractor._landmarker = MagicMock()
    mock_extractor.extract_structured_frame.return_value = StructuredLandmarkFrame(
        coordinates=coords,
        point_mask=mask,
        timestamp_ms=1000.0,
    )

    with TestClient(app) as client:
        monkeypatch.setattr(api_main, "_predictor", None)
        monkeypatch.setattr(api_main, "_extractor", mock_extractor)

        with client.websocket_connect("/ws/live") as ws:
            init = ws.receive_json()
            assert init["model_ready"] is False

            ws.send_json({"type": "frame", "data": _make_dummy_jpeg(), "frame_id": 5})

            messages = []
            for _ in range(5):
                msg = ws.receive_json()
                messages.append(msg)
                if msg.get("type") == "frame_processed":
                    break

            hand_msg = next(m for m in messages if m["type"] == "hand_landmarks")
            assert hand_msg["state_code"] == "STATE_G_MODEL_UNAVAILABLE"
            assert not any(m["type"] == "live_prediction" for m in messages)


def test_recoverable_bad_frame_does_not_break_websocket() -> None:
    """Verify invalid frame payloads are reported and socket remains usable for subsequent frames."""
    with TestClient(app) as client, client.websocket_connect("/ws/live") as ws:
        ws.receive_json()  # init

        # 1. Send completely invalid frame payload
        ws.send_json({"type": "frame", "data": "not_an_image_data_uri", "frame_id": 99})

        # Wait for frame_processed ack
        for _ in range(3):
            msg1 = ws.receive_json()
            if msg1["type"] == "frame_processed":
                break
        assert msg1["type"] == "frame_processed"
        assert msg1["frame_id"] == 99
        assert msg1["success"] is False

        # 2. Send reset message to verify socket is healthy
        ws.send_json({"type": "reset"})
        msg2 = ws.receive_json()
        assert msg2["type"] == "reset_ack"


def test_real_mediapipe_landmarker_detection_on_image() -> None:
    """End-to-end integration test with real MediaPipe HolisticLandmarker asset if present."""
    task_path = Path("models/mediapipe/holistic_landmarker.task")
    if not task_path.is_file():
        pytest.skip(f"MediaPipe task asset not found at {task_path}")

    from sign_language.landmarks.extractor import LandmarkExtractor

    extractor = LandmarkExtractor(model_asset_path=str(task_path))
    assert extractor._get_landmarker() is not None

    img = np.ones((480, 640, 3), dtype=np.uint8) * 240
    landmarks = extractor.extract_structured_frame(img)
    assert isinstance(landmarks, StructuredLandmarkFrame)
    assert landmarks.coordinates.shape == (553, 3)
    assert landmarks.point_mask.shape == (553,)
    if extractor._landmarker is not None:
        extractor._landmarker.close()
