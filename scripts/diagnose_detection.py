#!/usr/bin/env python3
"""Standalone MediaPipe Hand & Holistic Landmark Detection Diagnostic Utility.

Verifies and benchmarks MediaPipe detector initialization, hand landmark extraction,
bounding box generation, and latency independently of the word classification model.

Usage:
    python scripts/diagnose_detection.py --source 0
    python scripts/diagnose_detection.py --source path/to/image.jpg
    python scripts/diagnose_detection.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure repository root is in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sign_language.landmarks.extractor import LandmarkExtractor  # noqa: E402
from sign_language.landmarks.schema import compute_landmark_bounding_box  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose_detection")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (9, 10), (10, 11), (11, 12),           # Middle
    (13, 14), (14, 15), (15, 16),          # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm knuckles
]


def draw_hand_skeleton(
    image: np.ndarray,
    coords: np.ndarray,
    mask: np.ndarray,
    start_idx: int,
    color: tuple[int, int, int],
    label: str,
    box: dict[str, float] | None,
) -> None:
    """Render 21-point hand skeleton and bounding box on image in-place."""
    h, w = image.shape[:2]
    hand_coords = coords[start_idx : start_idx + 21]
    hand_mask = mask[start_idx : start_idx + 21]

    # Draw bones
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        if hand_mask[p1_idx] > 0.5 and hand_mask[p2_idx] > 0.5:
            pt1 = (int(hand_coords[p1_idx, 0] * w), int(hand_coords[p1_idx, 1] * h))
            pt2 = (int(hand_coords[p2_idx, 0] * w), int(hand_coords[p2_idx, 1] * h))
            cv2.line(image, pt1, pt2, color, 2, cv2.LINE_AA)

    # Draw nodes
    for i in range(21):
        if hand_mask[i] > 0.5:
            pt = (int(hand_coords[i, 0] * w), int(hand_coords[i, 1] * h))
            cv2.circle(image, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, pt, 2, color, -1, cv2.LINE_AA)

    # Draw bounding box
    if box is not None:
        bx1 = int(box["xmin"] * w)
        by1 = int(box["ymin"] * h)
        bx2 = int(box["xmax"] * w)
        by2 = int(box["ymax"] * h)
        cv2.rectangle(image, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(
            image,
            f"{label}: ({box['width']:.2f}x{box['height']:.2f})",
            (bx1, max(18, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def run_diagnostics(args: argparse.Namespace) -> int:
    task_path = Path(args.task_path)
    if not task_path.is_file():
        logger.error("MediaPipe task asset NOT found: %s", task_path.resolve())
        logger.error(
            "Please ensure the task asset is downloaded to models/mediapipe/holistic_landmarker.task"
        )
        return 1

    logger.info("Initializing MediaPipe HolisticLandmarker...")
    logger.info("Task asset path: %s (size: %d bytes)", task_path.resolve(), task_path.stat().st_size)
    logger.info("Representation: %s, Mode: %s", args.representation, args.mode)

    try:
        extractor = LandmarkExtractor(
            model_asset_path=task_path,
            representation=args.representation,
            running_mode=args.mode,
        )
        # Test landmarker creation
        extractor._get_landmarker()
        logger.info("MediaPipe HolisticLandmarker successfully initialized!")
    except Exception as exc:
        logger.error("Failed to initialize MediaPipe landmarker: %s", exc)
        return 1

    source_str = str(args.source)
    is_image = False
    source_path = Path(source_str)

    if source_path.is_file() and source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        is_image = True

    try:
        if is_image:
            logger.info("Processing single image: %s", source_path)
            frame = cv2.imread(str(source_path))
            if frame is None:
                logger.error("Could not read image file: %s", source_path)
                return 1

            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            structured = extractor.extract_structured_frame(frame, timestamp_ms=0)
            extraction_ms = (time.perf_counter() - t0) * 1000.0

            coords = structured.coordinates
            mask = structured.point_mask

            lh_count = int(mask[:21].sum())
            rh_count = int(mask[21:42].sum())
            pose_count = int(mask[42:75].sum())
            face_count = int(mask[75:].sum())
            total_valid = int(mask.sum())

            left_box = compute_landmark_bounding_box(coords[:21], mask[:21])
            right_box = compute_landmark_bounding_box(coords[21:42], mask[21:42])

            print("\n" + "=" * 60)
            print("DIAGNOSTIC DETECTION RESULTS (Single Image)")
            print("=" * 60)
            print(f"Source:              {source_path}")
            print(f"Resolution:          {w}x{h}")
            print(f"Extraction Latency:  {extraction_ms:.2f} ms")
            print(f"Left Hand Detected:  {'YES' if lh_count > 0 else 'NO'} ({lh_count}/21 points)")
            print(f"Right Hand Detected: {'YES' if rh_count > 0 else 'NO'} ({rh_count}/21 points)")
            print(f"Pose Points:         {pose_count}/33")
            print(f"Face Points:         {face_count}/478")
            print(f"Total Valid Points:  {total_valid}/553")
            print(f"Left Box:            {left_box}")
            print(f"Right Box:           {right_box}")
            print("=" * 60 + "\n")

            if not args.no_gui:
                draw_hand_skeleton(frame, coords, mask, 0, (255, 128, 0), "L-Hand", left_box)
                draw_hand_skeleton(frame, coords, mask, 21, (0, 200, 255), "R-Hand", right_box)
                cv2.putText(
                    frame,
                    f"Latency: {extraction_ms:.1f}ms | Points: {total_valid}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("SignFlow Detection Diagnostic", frame)
                logger.info("Displaying image preview. Press any key to close...")
                cv2.waitKey(0)

        else:
            # Video or Webcam stream
            cam_source = int(source_str) if source_str.isdigit() else source_str
            cap = cv2.VideoCapture(cam_source)
            if not cap.isOpened():
                logger.error("Could not open video/camera source: %s", source_str)
                return 1

            logger.info("Opened video source: %s. Starting stream diagnostics (press 'q' to quit)...", source_str)
            frame_count = 0
            hand_detected_count = 0
            latencies = []
            start_wall = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                h, w = frame.shape[:2]
                ts_ms = int((time.time() - start_wall) * 1000)

                t0 = time.perf_counter()
                structured = extractor.extract_structured_frame(frame, timestamp_ms=ts_ms)
                extraction_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(extraction_ms)

                coords = structured.coordinates
                mask = structured.point_mask

                lh_count = int(mask[:21].sum())
                rh_count = int(mask[21:42].sum())
                pose_count = int(mask[42:75].sum())
                face_count = int(mask[75:].sum())
                total_valid = int(mask.sum())
                has_hand = (lh_count > 0 or rh_count > 0)
                if has_hand:
                    hand_detected_count += 1

                left_box = compute_landmark_bounding_box(coords[:21], mask[:21])
                right_box = compute_landmark_bounding_box(coords[21:42], mask[21:42])

                elapsed = max(1e-3, time.time() - start_wall)
                current_fps = frame_count / elapsed

                if frame_count % 10 == 0 or frame_count == 1:
                    logger.info(
                        "Frame %d | Res: %dx%d | FPS: %.1f | Extract: %.1fms | Hands: L=%d/21 R=%d/21 | Total pts: %d",
                        frame_count, w, h, current_fps, extraction_ms, lh_count, rh_count, total_valid
                    )

                if not args.no_gui:
                    draw_hand_skeleton(frame, coords, mask, 0, (255, 128, 0), "L-Hand", left_box)
                    draw_hand_skeleton(frame, coords, mask, 21, (0, 200, 255), "R-Hand", right_box)
                    status_text = f"FPS: {current_fps:.1f} | Extract: {extraction_ms:.1f}ms | Hands: {'YES' if has_hand else 'NO'}"
                    cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if has_hand else (0, 165, 255), 2)
                    cv2.imshow("SignFlow Detection Diagnostic", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.max_frames and frame_count >= args.max_frames:
                    break

            cap.release()
            total_elapsed = max(1e-3, time.time() - start_wall)
            avg_fps = frame_count / total_elapsed
            med_latency = float(np.median(latencies)) if latencies else 0.0
            p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
            det_rate = (hand_detected_count / max(1, frame_count)) * 100.0

            print("\n" + "=" * 60)
            print("STREAM DIAGNOSTIC SUMMARY")
            print("=" * 60)
            print(f"Total Frames Processed:   {frame_count}")
            print(f"Hand Detection Rate:      {det_rate:.1f}% ({hand_detected_count}/{frame_count} frames)")
            print(f"Average Processed FPS:    {avg_fps:.1f}")
            print(f"Median Extraction Time:   {med_latency:.1f} ms")
            print(f"95th-Percentile Latency:  {p95_latency:.1f} ms")
            print("=" * 60 + "\n")

    finally:
        extractor.close()
        cv2.destroyAllWindows()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose and benchmark MediaPipe hand and holistic detection independently of word classification."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Input source: camera index (e.g. 0), image file path, or video file path (default: '0')",
    )
    parser.add_argument(
        "--task-path",
        default="models/mediapipe/holistic_landmarker.task",
        help="Path to MediaPipe HolisticLandmarker .task file",
    )
    parser.add_argument(
        "--representation",
        default="holistic",
        choices=["holistic", "hands", "hands_pose"],
        help="Landmark representation (default: holistic)",
    )
    parser.add_argument(
        "--mode",
        default="image",
        choices=["image", "video"],
        help="MediaPipe running mode (default: image)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process for video/camera sources",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV preview window (headless / automated test mode)",
    )
    args = parser.parse_args()
    sys.exit(run_diagnostics(args))


if __name__ == "__main__":
    main()
