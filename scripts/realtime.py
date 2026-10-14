from __future__ import annotations

import argparse
import collections
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from sign_language.api.inference import RealTimePredictor
from sign_language.landmarks.extractor import LandmarkExtractor
from sign_language.landmarks.pipeline import LandmarkPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Bangla sign language word recognition from webcam or video file.")
    parser.add_argument("--video-source", default="0", help="Camera index (e.g. 0) or path to video file")
    parser.add_argument(
        "--model-path",
        default="checkpoints/bdslw131/robust_holistic_fusion_best.pt",
        help="Path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument("--config", default="configs/bdslw131/proposed.yaml", help="Path to config file")
    parser.add_argument(
        "--model-asset-path",
        default="models/mediapipe/holistic_landmarker.task",
        help="Path to MediaPipe HolisticLandmarker .task asset",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.4, help="Minimum confidence threshold")
    parser.add_argument("--debounce-frames", type=int, default=5, help="Debounce filter length")
    args = parser.parse_args()

    # Fallback to alternative checkpoint if default path doesn't exist
    model_path = args.model_path
    config_path = args.config
    if not Path(model_path).is_file():
        candidates = [
            ("checkpoints/bdslw131/robust_holistic_fusion_best.pt", "configs/bdslw131/proposed.yaml"),
            ("checkpoints/bdslw131/baseline_best.pt", "configs/bdslw131/baseline.yaml"),
        ]
        for m_cand, c_cand in candidates:
            if Path(m_cand).is_file():
                model_path = m_cand
                if not Path(config_path).is_file() and Path(c_cand).is_file():
                    config_path = c_cand
                break

    predictor = RealTimePredictor(model_path=model_path, config_path=config_path)
    extractor = LandmarkExtractor(
        model_asset_path=args.model_asset_path,
        representation=predictor.config.dataset.feature_representation,
        running_mode="video",
    )
    pipeline = LandmarkPipeline(
        feature_dim=predictor.input_dim // 3 if predictor.input_dim >= 4977 else predictor.input_dim,
        sequence_length=predictor.sequence_length,
        extractor=extractor,
    )

    source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Could not open video source: %s", args.video_source)
        return

    logger.info("Starting real-time recognition loop. Press 'q' to quit.")
    sequence_buffer: collections.deque = collections.deque(maxlen=pipeline.sequence_length)
    mask_buffer: collections.deque = collections.deque(maxlen=pipeline.sequence_length)
    recent_predictions: collections.deque = collections.deque(maxlen=args.debounce_frames)

    frame_count = 0
    start_time = time.time()
    last_pred_label = "Waiting for sign..."
    last_confidence = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            ts_ms = int((current_time - start_time) * 1000)

            t0 = time.perf_counter()
            obs = extractor.extract_observation(frame, timestamp_ms=ts_ms)
            extraction_ms = (time.perf_counter() - t0) * 1000.0

            sequence_buffer.append(obs.landmarks)
            mask_buffer.append(obs.mask)

            inference_ms = 0.0
            if len(sequence_buffer) == pipeline.sequence_length:
                seq_array = np.stack(list(sequence_buffer))
                mask_array = np.stack(list(mask_buffer))
                t1 = time.perf_counter()
                pred_res = predictor.predict_label(
                    seq_array,
                    mask=mask_array,
                    confidence_threshold=args.confidence_threshold,
                )
                inference_ms = (time.perf_counter() - t1) * 1000.0

                if pred_res["meets_threshold"]:
                    recent_predictions.append(pred_res["predicted_label"])
                    # Debounce: majority vote
                    most_common = collections.Counter(recent_predictions).most_common(1)[0][0]
                    last_pred_label = most_common
                    last_confidence = pred_res["confidence"]
                else:
                    last_pred_label = "Low Confidence"
                    last_confidence = pred_res["confidence"]

            frame_count += 1
            fps = frame_count / max(1e-3, time.time() - start_time)

            # Draw overlay on frame
            color = (0, 255, 0) if last_confidence >= args.confidence_threshold else (0, 165, 255)
            cv2.putText(
                frame,
                f"Prediction: {last_pred_label} ({last_confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Latency: extract={extraction_ms:.1f}ms infer={inference_ms:.1f}ms",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )

            cv2.imshow("SignFlow-BdSL Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        logger.info("Resources released. Average FPS: %.2f", frame_count / max(1e-3, time.time() - start_time))


if __name__ == "__main__":
    main()
