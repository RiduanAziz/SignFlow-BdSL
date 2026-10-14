from __future__ import annotations

import base64
import collections
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sign_language.api.inference import RealTimePredictor, create_prediction_payload
from sign_language.landmarks.extractor import LandmarkExtractor
from sign_language.landmarks.schema import compute_landmark_bounding_box
from sign_language.robustness import apply_spatial_noise, apply_temporal_noise

logger = logging.getLogger(__name__)

# Global singleton predictor and extractor instances
_predictor: RealTimePredictor | None = None
_extractor: LandmarkExtractor | None = None
_init_error: str | None = None
_extractor_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _extractor, _init_error, _extractor_error
    model_path = os.environ.get("MODEL_PATH")
    config_path = os.environ.get("CONFIG_PATH")
    mediapipe_asset = os.environ.get(
        "MEDIAPIPE_ASSET_PATH", "models/mediapipe/holistic_landmarker.task"
    )

    # Prioritize BdSL checkpoints
    if not model_path:
        candidates = [
            ("checkpoints/bdslw131/robust_holistic_fusion_best.pt", "configs/bdslw131/proposed.yaml"),
            ("checkpoints/bdslw131/baseline_best.pt", "configs/bdslw131/baseline.yaml"),
        ]
        for m_cand, c_cand in candidates:
            if Path(m_cand).is_file():
                model_path = m_cand
                if not config_path and Path(c_cand).is_file():
                    config_path = c_cand
                break

    if not config_path:
        if Path("configs/bdslw131/proposed.yaml").is_file():
            config_path = "configs/bdslw131/proposed.yaml"
        else:
            config_path = "configs/base.yaml"

    # Initialize RealTimePredictor strictly from trained checkpoint
    try:
        if model_path and Path(model_path).is_file():
            _predictor = RealTimePredictor(model_path=model_path, config_path=config_path)
            logger.info("RealTimePredictor successfully initialized with model %s", model_path)
            _init_error = None
        else:
            _predictor = None
            _init_error = (
                f"No trained model checkpoint found at {model_path or 'checkpoints/bdslw131/robust_holistic_fusion_best.pt'}. "
                "Recognition disabled until valid checkpoint is supplied."
            )
            logger.warning(_init_error)
    except Exception as exc:
        _predictor = None
        _init_error = str(exc)
        logger.warning("Failed to initialize predictor at startup: %s", exc)

    # Initialize LandmarkExtractor eagerly
    task_path = Path(mediapipe_asset)
    if task_path.is_file():
        try:
            resolved_task = task_path.resolve()
            logger.info("Initializing MediaPipe task asset from: %s", resolved_task)
            rep = _predictor.config.dataset.feature_representation if _predictor else "holistic"
            extractor_inst = LandmarkExtractor(
                model_asset_path=resolved_task,
                representation=rep,
                running_mode="image",
            )
            # Eagerly confirm the landmarker is created and valid
            extractor_inst._get_landmarker()
            _extractor = extractor_inst
            _extractor_error = None
            logger.info("MediaPipe HolisticLandmarker successfully initialized from %s", resolved_task)
        except Exception as exc:
            _extractor = None
            _extractor_error = f"Failed to initialize MediaPipe HolisticLandmarker: {exc}"
            logger.warning("Failed to initialize LandmarkExtractor: %s", exc)
    else:
        _extractor = None
        _extractor_error = f"MediaPipe asset not found at {task_path.resolve()}"
        logger.info("LandmarkExtractor disabled: %s", _extractor_error)

    yield

    # Clean up on shutdown
    if _extractor is not None:
        try:
            _extractor.close()
        except Exception as exc:
            logger.debug("Error closing extractor on shutdown: %s", exc)
        finally:
            _extractor = None


app = FastAPI(
    title="SignFlow-BdSL — Live Bangla Sign Language Recognition API",
    description="Backend API and WebSocket service for real-time Bangla Sign Language (BDSLW131) word recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

# Safer CORS configuration: restricted development origins and optional CORS_ORIGINS env var
cors_env = os.environ.get("CORS_ORIGINS")
if cors_env:
    allowed_origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_predictor() -> RealTimePredictor:
    global _predictor, _init_error
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Predictor service unavailable: {_init_error or 'No trained model checkpoint loaded.'}",
        )
    return _predictor


def get_extractor() -> LandmarkExtractor | None:
    global _extractor
    return _extractor


class PredictionRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="2D sequence of landmark features [T, F]")
    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")


@app.get("/health")
def health() -> dict[str, Any]:
    mediapipe_asset_path = os.environ.get("MEDIAPIPE_ASSET_PATH", "models/mediapipe/holistic_landmarker.task")
    asset_present = Path(mediapipe_asset_path).is_file()
    mediapipe_initialized = (
        _extractor is not None
        and getattr(_extractor, "_landmarker", None) is not None
        and _extractor_error is None
    )

    model_loaded = (
        _predictor is not None
        and getattr(_predictor, "model", None) is not None
        and _init_error is None
    )

    checkpoint_compatible = False
    if model_loaded and _predictor is not None:
        checkpoint_compatible = (
            getattr(_predictor, "num_classes", 0) > 0
            and getattr(_predictor, "input_dim", 0) > 0
        )

    inference_ready = mediapipe_initialized and model_loaded and checkpoint_compatible

    if inference_ready:
        overall_status = "ok"
    elif model_loaded or mediapipe_initialized:
        overall_status = "degraded"
    else:
        overall_status = "error"

    model_info_dict = None
    if _predictor is not None and model_loaded:
        model_info_dict = {
            "name": getattr(_predictor.config.model, "name", "unknown"),
            "architecture": _predictor.model.__class__.__name__,
            "num_classes": _predictor.num_classes,
            "input_dim": _predictor.input_dim,
            "sequence_length": _predictor.sequence_length,
            "vocabulary_version": "BDSLW131-v1",
        }

    return {
        "status": overall_status,
        "process_alive": True,
        "backend_initialized": True,
        "websocket_available": True,
        "mediapipe_asset_present": asset_present,
        "mediapipe_initialized": mediapipe_initialized,
        "mediapipe_available": mediapipe_initialized,
        "trained_model_loaded": model_loaded,
        "checkpoint_compatible": checkpoint_compatible,
        "inference_ready": inference_ready,
        "checkpoint_path": getattr(_predictor, "checkpoint_path", None) if _predictor else None,
        "model_info": model_info_dict,
        "last_extraction_error": _extractor_error,
        "last_model_error": _init_error,
        "error": _init_error,
        "mediapipe_error": _extractor_error,
    }


@app.get("/model/info")
def model_info() -> dict[str, Any]:
    predictor = get_predictor()
    return {
        "name": predictor.config.model.name,
        "architecture": predictor.model.__class__.__name__ if hasattr(predictor, "model") else "unknown",
        "status": "ready" if _init_error is None else "error",
        "checkpoint_path": getattr(predictor, "checkpoint_path", None),
        "num_classes": predictor.num_classes,
        "input_dim": predictor.input_dim,
        "sequence_length": predictor.sequence_length,
        "device": str(predictor.device),
        "feature_representation": predictor.config.dataset.feature_representation,
        "class_names_sample": predictor.class_names[:10] if predictor.class_names else [],
        "total_classes": len(predictor.class_names),
        "error": _init_error,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not request.sequence:
        raise HTTPException(status_code=422, detail="Sequence must not be empty.")

    try:
        sequence = np.asarray(request.sequence, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid numeric sequence data: {exc}")

    if sequence.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 2D array [time_steps, features], got {sequence.shape}",
        )

    if not np.all(np.isfinite(sequence)):
        raise HTTPException(status_code=422, detail="Input sequence contains non-finite values (NaN or Inf).")

    predictor = get_predictor()

    try:
        prediction_result = predictor.predict_with_logits(
            sequence, confidence_threshold=request.confidence_threshold
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    payload = create_prediction_payload(sequence)

    return {
        "sequence_length": payload["sequence_length"],
        "feature_dim": payload["feature_dim"],
        "logits": prediction_result["logits"],
        "predicted_class": prediction_result["predicted_class"],
        "predicted_label": prediction_result["predicted_label"],
        "bengali_label": prediction_result.get("bengali_label", prediction_result["predicted_label"]),
        "english_gloss": prediction_result.get("english_gloss", prediction_result["predicted_label"]),
        "transliteration": prediction_result.get("transliteration", ""),
        "confidence": prediction_result["confidence"],
        "meets_threshold": prediction_result["meets_threshold"],
        "top_k": prediction_result.get("top_k", []),
    }


class RobustnessExperimentRequest(BaseModel):
    sequence: list[list[float]] = Field(..., description="Sequence of landmark features [T, F]")
    perturbation_type: str = Field(
        ...,
        description="Perturbation type (coordinate_jitter, translation, scale, landmark_dropout, frame_drop, frame_duplicate, sequence_truncate)",
    )
    severity: float = Field(0.2, ge=0.0, le=1.0, description="Perturbation severity")
    seed: int = Field(42, description="Random seed")


@app.get("/api/vocabulary")
def get_vocabulary() -> dict[str, Any]:
    """Returns the true vocabulary of the loaded model checkpoint for Guided Practice Mode."""
    predictor = _predictor
    if predictor is None:
        return {
            "available": False,
            "error": _init_error or "Model checkpoint not loaded.",
            "vocabulary": [],
            "items": [],
            "count": 0,
            "checkpoint": None,
        }

    items = []
    if getattr(predictor, "bdsl_vocab", None) is not None:
        items = predictor.bdsl_vocab.items
    else:
        for idx, name in enumerate(predictor.class_names):
            items.append({
                "class_id": idx,
                "dataset_label": name,
                "bengali_label": name,
                "english_gloss": name,
                "transliteration": "",
            })

    return {
        "available": True,
        "vocabulary": predictor.class_names,
        "items": items,
        "count": len(predictor.class_names),
        "checkpoint": getattr(predictor, "checkpoint_path", None),
    }


@app.post("/api/robustness/experiment")
def run_robustness_experiment(request: RobustnessExperimentRequest) -> dict[str, Any]:
    """Thesis-aligned robustness analysis comparing clean vs perturbed inference."""
    predictor = get_predictor()
    try:
        seq_clean = np.asarray(request.sequence, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid numeric sequence data: {exc}")

    if seq_clean.ndim != 2:
        raise HTTPException(
            status_code=422,
            detail=f"Expected 2D array [T, F], got shape {seq_clean.shape}",
        )

    t0 = time.perf_counter()
    clean_pred = predictor.predict_label(seq_clean, confidence_threshold=0.0)

    noise_type = request.perturbation_type
    severity = float(request.severity)
    seed = int(request.seed)

    spatial_types = {"coordinate_jitter", "translation", "scale", "landmark_dropout"}
    temporal_types = {"frame_drop", "frame_duplicate", "sequence_truncate"}

    try:
        if noise_type in spatial_types:
            perturbed_seq = apply_spatial_noise(seq_clean, noise_type, severity=severity, seed=seed)
        elif noise_type in temporal_types:
            result = apply_temporal_noise(seq_clean, noise_type, severity=severity, seed=seed)
            perturbed_seq = result[0] if isinstance(result, tuple) else result
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported perturbation '{noise_type}'. Must be one of {sorted(spatial_types | temporal_types)}",
            )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Perturbation execution error: {exc}")

    try:
        perturbed_pred = predictor.predict_label(perturbed_seq, confidence_threshold=0.0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Perturbed prediction execution error: {exc}")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    is_consistent = bool(clean_pred["predicted_label"] == perturbed_pred["predicted_label"])
    clean_sparsity = float(1.0 - (np.count_nonzero(seq_clean) / max(1, seq_clean.size)))
    perturbed_sparsity = float(1.0 - (np.count_nonzero(perturbed_seq) / max(1, perturbed_seq.size)))

    return {
        "original_prediction": clean_pred["predicted_label"],
        "original_confidence": float(clean_pred["confidence"]),
        "perturbed_prediction": perturbed_pred["predicted_label"],
        "perturbed_confidence": float(perturbed_pred["confidence"]),
        "prediction_consistent": is_consistent,
        "prediction_changed": not is_consistent,
        "perturbation_type": noise_type,
        "severity": severity,
        "processing_time_ms": round(elapsed_ms, 2),
        "clean_quality": {
            "sequence_length": int(seq_clean.shape[0]),
            "feature_dim": int(seq_clean.shape[1]),
            "sparsity_ratio": round(clean_sparsity, 3),
        },
        "perturbed_quality": {
            "sequence_length": int(perturbed_seq.shape[0]),
            "feature_dim": int(perturbed_seq.shape[1]),
            "sparsity_ratio": round(perturbed_sparsity, 3),
        },
        "note": "Prediction consistency is reported for this live sequence. Ground-truth accuracy is evaluated on labelled offline benchmark datasets.",
    }



class SessionState:
    def __init__(self, seq_len: int, min_valid_frames: int | None = None):
        import collections
        import time
        self.seq_len = max(1, seq_len)
        self.mode = "guided"
        self.confidence_threshold = 0.28
        self.debounce_frames = 2
        self.pause_threshold_sec = 0.5

        # Eliminate sequence-length contradiction:
        # Static models (seq_len == 1) require exactly 1 frame.
        if self.seq_len == 1:
            self.min_sequence_frames = 1
        else:
            self.min_sequence_frames = min(
                self.seq_len,
                max(1, min_valid_frames if min_valid_frames is not None else min(10, self.seq_len)),
            )

        self.sequence_buffer: collections.deque = collections.deque(maxlen=self.seq_len)
        self.mask_buffer: collections.deque = collections.deque(maxlen=self.seq_len)
        self.candidate_history: collections.deque = collections.deque(maxlen=self.debounce_frames)
        self.recent_replay_frames: collections.deque = collections.deque(maxlen=64)
        self.wrist_history: collections.deque = collections.deque(maxlen=5)
        self.speed_history: collections.deque = collections.deque(maxlen=10)

        self.last_brightness = None
        self.last_blur_score = None

        self.current_candidate = None
        self.current_confidence = 0.0
        self.last_committed_word = None

        # Continuous mode specific
        self.continuous_candidate_history = collections.deque(maxlen=5)
        self.continuous_cooldown_frames = 0
        self.continuous_last_emitted = None

        self.state = "WAITING"
        self.last_hand_activity_time = time.time()
        self.frame_counter = 0
        self.processed_frames_count = 0
        self.extraction_error_count = 0
        self.start_time = time.time()

    def update_config(self, payload: dict):
        import collections
        if "mode" in payload and payload["mode"] in {"guided", "continuous"}:
            self.mode = payload["mode"]
        if "confidence_threshold" in payload:
            self.confidence_threshold = float(payload["confidence_threshold"])
        if "debounce_frames" in payload:
            self.debounce_frames = max(1, int(payload["debounce_frames"]))
            self.candidate_history = collections.deque(
                list(self.candidate_history), maxlen=self.debounce_frames
            )
        if "pause_threshold_sec" in payload:
            self.pause_threshold_sec = float(payload["pause_threshold_sec"])
        if "min_sequence_frames" in payload:
            req_frames = int(payload["min_sequence_frames"])
            self.min_sequence_frames = min(self.seq_len, max(1, req_frames))

    def full_reset(self):
        self.sequence_buffer.clear()
        self.mask_buffer.clear()
        self.candidate_history.clear()
        self.wrist_history.clear()
        self.speed_history.clear()
        self.continuous_candidate_history.clear()
        self.continuous_cooldown_frames = 0
        self.continuous_last_emitted = None
        self.current_candidate = None
        self.current_confidence = 0.0
        self.last_committed_word = None
        self.state = "WAITING"

    def transition_to_committed(self, word: str):
        self.last_committed_word = word
        self.state = "COMMITTED"
        self.current_candidate = None
        self.sequence_buffer.clear()
        self.mask_buffer.clear()
        self.candidate_history.clear()
        self.speed_history.clear()
        self.wrist_history.clear()

    def handle_neutral_pause(self):
        if self.state in {"COMMITTED", "CAPTURING", "CANDIDATE"}:
            self.state = "WAITING"
            self.sequence_buffer.clear()
            self.mask_buffer.clear()
            self.wrist_history.clear()
            self.speed_history.clear()
            self.continuous_last_emitted = None # Allow repeating words after pause

    def append_frame(self, obs_landmarks, obs_mask):
        self.sequence_buffer.append(obs_landmarks)
        self.mask_buffer.append(obs_mask)


def _resolve_word_bilingual(word: str | None, predictor: RealTimePredictor | None) -> tuple[str | None, str | None]:
    if not word or predictor is None:
        return word, word
    vocab = getattr(predictor, "bdsl_vocab", None)
    if vocab is None or not hasattr(vocab, "get_id_by_label"):
        return str(word), str(word)
    try:
        cid = vocab.get_id_by_label(word)
        item = vocab.get_by_id(cid)
        if isinstance(item, dict):
            b_val = item.get("bengali_label")
            e_val = item.get("english_gloss")
            return str(b_val) if b_val is not None else str(word), str(e_val) if e_val is not None else str(word)
        return str(word), str(word)
    except Exception:
        return str(word), str(word)


@app.websocket("/ws/live")
async def websocket_live_recognition(websocket: WebSocket) -> None:
    """Persistent bidirectional WebSocket connection for real-time BdSL recognition."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("WebSocket client connected. Session ID: %s", session_id)

    predictor = _predictor
    extractor = _extractor

    model_ready = predictor is not None and getattr(predictor, "model", None) is not None
    extractor_ready = extractor is not None and getattr(extractor, "_landmarker", None) is not None

    seq_len = predictor.sequence_length if predictor else 1
    raw_dim = (
        predictor.input_dim // 3
        if (predictor and predictor.input_dim >= 4977)
        else (predictor.input_dim if predictor else 1659)
    )

    s = SessionState(seq_len)

    await websocket.send_json({
        "type": "session_started",
        "session_id": session_id,
        "model_ready": model_ready,
        "model_name": predictor.config.model.name if predictor else None,
        "num_classes": predictor.num_classes if predictor else 0,
        "sequence_length": seq_len,
        "min_sequence_frames": s.min_sequence_frames,
        "feature_dim": raw_dim,
        "mediapipe_ready": extractor_ready,
        "active_mode": "guided",
        "supported_modes": ["guided", "continuous"],
        "is_bdsl": True,
        "vocabulary_version": "BDSLW131-v1",
    })

    try:
        while True:
            message = await websocket.receive()
            now = time.time()

            # Handle disconnect message cleanly before trying to read message content
            if message.get("type") == "websocket.disconnect":
                logger.info("Client disconnect received for session %s", session_id)
                break

            frame_id = s.frame_counter + 1
            ts_ms = int(now * 1000)
            frame = None

            if message.get("text"):
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON format."})
                    continue
                msg_type = payload.get("type", "")

                if msg_type == "request_replay":
                    await websocket.send_json({
                        "type": "replay_buffer",
                        "frames": list(s.recent_replay_frames),
                    })
                    continue
                elif msg_type == "config":
                    s.update_config(payload)
                    await websocket.send_json({
                        "type": "config_updated",
                        "mode": s.mode,
                        "confidence_threshold": s.confidence_threshold,
                        "debounce_frames": s.debounce_frames,
                        "pause_threshold_sec": s.pause_threshold_sec,
                        "min_sequence_frames": s.min_sequence_frames,
                    })
                    continue
                elif msg_type == "confirm_word":
                    if s.current_candidate:
                        token_id = str(uuid.uuid4())[:8]
                        b_word, e_word = _resolve_word_bilingual(s.current_candidate, predictor)
                        await websocket.send_json({
                            "type": "word_committed",
                            "token_id": token_id,
                            "word": s.current_candidate,
                            "bengali_word": b_word,
                            "english_gloss": e_word,
                            "confidence": s.current_confidence,
                            "timestamp": time.time(),
                            "mode": s.mode,
                            "source": "manual_confirm",
                        })
                        s.transition_to_committed(s.current_candidate)
                    continue
                elif msg_type == "reject_candidate":
                    s.current_candidate = None
                    s.current_confidence = 0.0
                    s.candidate_history.clear()
                    s.sequence_buffer.clear()
                    s.mask_buffer.clear()
                    s.state = "WAITING"
                    await websocket.send_json({
                        "type": "word_candidate",
                        "word": None,
                        "confidence": 0.0,
                        "stability_count": 0,
                        "target_count": s.debounce_frames,
                        "progress": 0.0,
                    })
                    continue
                elif msg_type == "reset":
                    s.full_reset()
                    await websocket.send_json({"type": "reset_ack"})
                    continue
                elif msg_type == "frame":
                    frame_id = payload.get("frame_id", s.frame_counter + 1)
                    ts_ms = payload.get("timestamp_ms", int(now * 1000))
                    frame_data = payload.get("data", "")
                    if not frame_data:
                        await websocket.send_json({
                            "type": "frame_processed",
                            "frame_id": frame_id,
                            "timestamp_ms": ts_ms,
                            "processing_ms": 0.0,
                            "success": False,
                        })
                        continue
                    if "," in frame_data:
                        frame_data = frame_data.split(",", 1)[1]
                    try:
                        img_bytes = base64.b64decode(frame_data)
                        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    except Exception as exc:
                        await websocket.send_json({
                            "type": "warning",
                            "frame_id": frame_id,
                            "message": f"Frame decode error: {exc}",
                        })
                        await websocket.send_json({
                            "type": "frame_processed",
                            "frame_id": frame_id,
                            "timestamp_ms": ts_ms,
                            "processing_ms": 0.0,
                            "success": False,
                        })
                        continue
                    if frame is None:
                        await websocket.send_json({
                            "type": "frame_processed",
                            "frame_id": frame_id,
                            "timestamp_ms": ts_ms,
                            "processing_ms": 0.0,
                            "success": False,
                        })
                        continue
                else:
                    continue
            elif message.get("bytes"):
                frame_id = s.frame_counter + 1
                ts_ms = int(now * 1000)
                try:
                    img_arr = np.frombuffer(message["bytes"], dtype=np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                except Exception as exc:
                    await websocket.send_json({
                        "type": "warning",
                        "frame_id": frame_id,
                        "message": f"Binary decode error: {exc}",
                    })
                    await websocket.send_json({
                        "type": "frame_processed",
                        "frame_id": frame_id,
                        "timestamp_ms": ts_ms,
                        "processing_ms": 0.0,
                        "success": False,
                    })
                    continue
                if frame is None:
                    await websocket.send_json({
                        "type": "frame_processed",
                        "frame_id": frame_id,
                        "timestamp_ms": ts_ms,
                        "processing_ms": 0.0,
                        "success": False,
                    })
                    continue
            else:
                continue

            s.frame_counter += 1
            t0 = time.perf_counter()

            if extractor is None:
                await websocket.send_json({
                    "type": "landmarks_status",
                    "frame_id": frame_id,
                    "hands_detected": False,
                    "left_hand": False,
                    "right_hand": False,
                    "pose": False,
                    "face": False,
                    "valid_points": 0,
                    "error": "MediaPipe unavailable.",
                    "timestamp_ms": ts_ms,
                })
                await websocket.send_json({
                    "type": "frame_processed",
                    "frame_id": frame_id,
                    "timestamp_ms": ts_ms,
                    "processing_ms": 0.0,
                    "success": False,
                })
                continue

            try:
                structured = extractor.extract_structured_frame(frame, timestamp_ms=int(ts_ms))
            except Exception as exc:
                s.extraction_error_count += 1
                logger.warning("MediaPipe extraction error on frame %s: %s", frame_id, exc)
                await websocket.send_json({
                    "type": "extraction_error",
                    "frame_id": frame_id,
                    "message": f"Landmark extraction failure: {exc}",
                    "timestamp_ms": ts_ms,
                })
                await websocket.send_json({
                    "type": "frame_processed",
                    "frame_id": frame_id,
                    "timestamp_ms": ts_ms,
                    "processing_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "success": False,
                })
                continue

            extraction_ms = (time.perf_counter() - t0) * 1000.0

            selected_modalities = ["left_hand", "right_hand"]
            if extractor.representation in {"hands_pose", "holistic"}:
                selected_modalities.append("pose")
            if extractor.representation == "holistic":
                selected_modalities.append("face")

            mod_values, mod_masks = [], []
            for m_name in selected_modalities:
                c_mod, m_mod = structured.get_modality(m_name)
                mod_values.append(c_mod.reshape(-1))
                mod_masks.append(np.repeat(m_mod, 3).astype(np.float32))

            obs_landmarks = np.concatenate(mod_values).astype(np.float32)
            obs_mask = np.concatenate(mod_masks).astype(np.float32)
            if obs_landmarks.shape[0] > extractor.feature_dim:
                obs_landmarks = obs_landmarks[: extractor.feature_dim]
                obs_mask = obs_mask[: extractor.feature_dim]
            elif obs_landmarks.shape[0] < extractor.feature_dim:
                obs_landmarks = np.pad(obs_landmarks, (0, extractor.feature_dim - obs_landmarks.shape[0]))
                obs_mask = np.pad(obs_mask, (0, extractor.feature_dim - obs_mask.shape[0]))

            coords, mask = structured.coordinates, structured.point_mask
            left_hand = bool(mask[:21].sum() > 0)
            right_hand = bool(mask[21:42].sum() > 0)
            pose = bool(mask[42:75].sum() > 0)
            face = bool(mask[75:].sum() > 0)
            hands_present = left_hand or right_hand
            valid_points = int(mask.sum())

            # Real bounding boxes computed from valid landmark coordinates
            left_box = compute_landmark_bounding_box(coords[:21], mask[:21])
            right_box = compute_landmark_bounding_box(coords[21:42], mask[21:42])

            left_pts = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]), "valid": bool(mask[i] > 0)} for i in range(21)]
            right_pts = [{"x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]), "valid": bool(mask[i] > 0)} for i in range(21, 42)]

            # Quality Coach
            coach_score = 100
            if not hands_present:
                coach_score -= 60
            buf_completeness = min(1.0, len(s.sequence_buffer) / max(1, s.min_sequence_frames))

            state_code = "STATE_F_UNCERTAIN"
            if predictor is None:
                state_code = "STATE_G_MODEL_UNAVAILABLE"
            elif not hands_present:
                state_code = "STATE_B_NO_HAND"
            elif len(s.sequence_buffer) < s.min_sequence_frames:
                state_code = "STATE_C_COLLECTING"
            elif s.state == "COMMITTED":
                state_code = "STATE_E_COMMITTED"
            elif s.current_candidate and s.current_candidate not in {"Unknown (Low Confidence)", "Model Unavailable"}:
                state_code = "STATE_D_CANDIDATE"

            await websocket.send_json({
                "type": "hand_landmarks",
                "frame_id": frame_id,
                "left_landmarks": left_pts,
                "right_landmarks": right_pts,
                "left_box": left_box,
                "right_box": right_box,
                "hands_detected": hands_present,
                "left_hand": left_hand,
                "right_hand": right_hand,
                "pose": pose,
                "face": face,
                "valid_points": valid_points,
                "quality": {"score": max(5, coach_score), "buffer_completeness": buf_completeness},
                "state_code": state_code,
                "timestamp_ms": ts_ms,
            })

            s.recent_replay_frames.append({
                "frame_idx": s.frame_counter, "timestamp_ms": ts_ms,
                "left_landmarks": left_pts, "right_landmarks": right_pts,
                "hands_detected": hands_present, "valid_points": valid_points,
                "state_code": state_code, "candidate": s.current_candidate,
                "confidence": s.current_confidence, "features": obs_landmarks.tolist(),
            })

            # Hand activity tracking
            if hands_present:
                s.last_hand_activity_time = now
                if s.state == "WAITING":
                    s.state = "CAPTURING"
                s.append_frame(obs_landmarks, obs_mask)
            else:
                if (now - s.last_hand_activity_time) > s.pause_threshold_sec:
                    s.handle_neutral_pause()

            inference_ms = 0.0

            # Only run inference if hands are present and model is loaded
            if hands_present and predictor is not None:
                if len(s.sequence_buffer) >= s.min_sequence_frames:
                    # Cooldown logic for continuous mode
                    if s.mode == "continuous" and s.continuous_cooldown_frames > 0:
                        s.continuous_cooldown_frames -= 1
                    else:
                        seq_array = np.stack(list(s.sequence_buffer))
                        mask_array = np.stack(list(s.mask_buffer))

                        t1 = time.perf_counter()
                        try:
                            pred_res = predictor.predict_label(
                                seq_array,
                                mask=mask_array,
                                confidence_threshold=s.confidence_threshold,
                                calibrate=True,
                                calibration_alpha=0.85,
                            )
                        except Exception as exc:
                            logger.warning("Predictor inference error on frame %s: %s", frame_id, exc)
                            pred_res = None

                        if pred_res is not None:
                            inference_ms = (time.perf_counter() - t1) * 1000.0
                            pred_label = pred_res["predicted_label"]
                            conf = pred_res["confidence"]
                            meets_threshold = pred_res["meets_threshold"]

                            bengali_label = pred_res.get("bengali_label", pred_label)
                            english_gloss = pred_res.get("english_gloss", pred_label)
                            translit = pred_res.get("transliteration", "")

                            await websocket.send_json({
                                "type": "live_prediction",
                                "frame_id": frame_id,
                                "predicted_label": pred_label,
                                "bengali_label": bengali_label,
                                "english_gloss": english_gloss,
                                "transliteration": translit,
                                "predicted_class": pred_res.get("predicted_class"),
                                "confidence": conf,
                                "meets_threshold": meets_threshold,
                                "top_k": pred_res.get("top_k", []),
                                "state": s.state,
                                "state_code": state_code,
                                "is_provisional": True,
                                "latency_ms": round(extraction_ms + inference_ms, 1),
                            })

                            if s.mode == "guided":
                                if meets_threshold and pred_label not in {"Unknown (Low Confidence)", "Model Unavailable"}:
                                    s.candidate_history.append(pred_label)
                                    most_common, count = collections.Counter(s.candidate_history).most_common(1)[0]
                                    s.current_candidate = most_common
                                    s.current_confidence = conf
                                    progress = min(1.0, count / s.debounce_frames)
                                    b_cand, e_cand = _resolve_word_bilingual(s.current_candidate, predictor)
                                    await websocket.send_json({
                                        "type": "word_candidate",
                                        "frame_id": frame_id,
                                        "word": s.current_candidate,
                                        "bengali_word": b_cand,
                                        "english_gloss": e_cand,
                                        "confidence": s.current_confidence,
                                        "stability_count": count,
                                        "target_count": s.debounce_frames,
                                        "progress": progress,
                                    })
                                    if count >= s.debounce_frames and s.state != "COMMITTED":
                                        is_new_word = s.current_candidate != s.last_committed_word
                                        is_new_gesture = (now - s.last_hand_activity_time) < 0.2
                                        if is_new_word or is_new_gesture:
                                            b_comm, e_comm = _resolve_word_bilingual(s.current_candidate, predictor)
                                            await websocket.send_json({
                                                "type": "word_committed",
                                                "frame_id": frame_id,
                                                "token_id": str(uuid.uuid4())[:8],
                                                "word": s.current_candidate,
                                                "bengali_word": b_comm,
                                                "english_gloss": e_comm,
                                                "confidence": s.current_confidence,
                                                "timestamp": time.time(),
                                                "mode": "guided",
                                            })
                                            s.transition_to_committed(s.current_candidate)
                                else:
                                    if s.state != "COMMITTED":
                                        await websocket.send_json({
                                            "type": "word_candidate",
                                            "frame_id": frame_id,
                                            "word": None,
                                            "bengali_word": None,
                                            "english_gloss": None,
                                            "confidence": conf,
                                            "stability_count": 0,
                                            "target_count": s.debounce_frames,
                                            "progress": 0.0,
                                        })
                            elif s.mode == "continuous":
                                if meets_threshold and pred_label not in {"Unknown (Low Confidence)", "Model Unavailable"}:
                                    s.continuous_candidate_history.append(pred_label)
                                    if len(s.continuous_candidate_history) == s.continuous_candidate_history.maxlen:
                                        most_common, count = collections.Counter(s.continuous_candidate_history).most_common(1)[0]
                                        if count >= 4 and most_common != s.continuous_last_emitted:
                                            b_comm, e_comm = _resolve_word_bilingual(most_common, predictor)
                                            await websocket.send_json({
                                                "type": "word_committed",
                                                "frame_id": frame_id,
                                                "token_id": str(uuid.uuid4())[:8],
                                                "word": most_common,
                                                "bengali_word": b_comm,
                                                "english_gloss": e_comm,
                                                "confidence": conf,
                                                "timestamp": time.time(),
                                                "mode": "continuous",
                                                "source": "auto_continuous",
                                            })
                                            s.continuous_last_emitted = most_common
                                            s.continuous_cooldown_frames = 15
                                            s.continuous_candidate_history.clear()
                                            s.sequence_buffer.clear()
                                            s.mask_buffer.clear()

            s.processed_frames_count += 1
            total_processing_ms = (time.perf_counter() - t0) * 1000.0

            # Send final frame acknowledgement for this frame
            await websocket.send_json({
                "type": "frame_processed",
                "frame_id": frame_id,
                "timestamp_ms": ts_ms,
                "processing_ms": round(total_processing_ms, 2),
                "success": True,
            })

            if s.processed_frames_count % 5 == 0:
                elapsed = max(1e-3, now - s.start_time)
                await websocket.send_json({
                    "type": "performance",
                    "frame_id": frame_id,
                    "fps": round(s.processed_frames_count / elapsed, 1),
                    "extraction_ms": round(extraction_ms, 1),
                    "inference_ms": round(inference_ms, 1),
                    "total_latency_ms": round(extraction_ms + inference_ms, 1),
                    "frame_count": s.frame_counter,
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected cleanly. Session ID: %s", session_id)
    except Exception as exc:
        logger.warning("Unexpected error in WebSocket loop: %s", exc)

# Mount static production build of frontend if present
_dist_dir = Path("app/frontend/dist")
if _dist_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist_dir), html=True), name="frontend")
