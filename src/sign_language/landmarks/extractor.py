from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np

from .schema import (
    MODALITY_COUNTS,
    MODALITY_ORDER,
    TOTAL_HOLISTIC_POINTS,
    LandmarkObservation,
    StructuredLandmarkFrame,
)

logger = logging.getLogger(__name__)


class LandmarkExtractor:
    """Extract holistic landmarks using MediaPipe Tasks HolisticLandmarker."""

    _MODALITY_COUNTS: ClassVar[dict[str, int]] = MODALITY_COUNTS

    def __init__(
        self,
        feature_dim: int | None = None,
        model_asset_path: str | Path | None = None,
        representation: str = "holistic",
        running_mode: str = "image",  # "image" or "video"
    ) -> None:
        if representation not in {"hands", "hands_pose", "holistic"}:
            raise ValueError(f"Unsupported landmark representation: {representation}")
        if running_mode not in {"image", "video"}:
            raise ValueError(f"Unsupported running mode: {running_mode}")

        self.model_asset_path = Path(model_asset_path) if model_asset_path is not None else None
        self.representation = representation
        self.running_mode = running_mode
        self.feature_dim = int(feature_dim or self._feature_dim(representation))
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self._landmarker = None

    @classmethod
    def _feature_dim(cls, representation: str) -> int:
        point_count = cls._MODALITY_COUNTS["left_hand"] + cls._MODALITY_COUNTS["right_hand"]
        if representation in {"hands_pose", "holistic"}:
            point_count += cls._MODALITY_COUNTS["pose"]
        if representation == "holistic":
            point_count += cls._MODALITY_COUNTS["face"]
        return point_count * 3

    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        if self.model_asset_path is None:
            raise RuntimeError("A MediaPipe HolisticLandmarker .task asset is required for extraction")
        if not self.model_asset_path.is_file():
            raise FileNotFoundError(f"MediaPipe model asset does not exist: {self.model_asset_path}")

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        mode = (
            vision.RunningMode.VIDEO
            if self.running_mode == "video"
            else vision.RunningMode.IMAGE
        )

        options = vision.HolisticLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_asset_path)),
            running_mode=mode,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(options)
        return self._landmarker

    def close(self) -> None:
        """Release MediaPipe landmarker resources explicitly."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception as exc:
                logger.debug("Error closing landmarker: %s", exc)
            finally:
                self._landmarker = None

    def __enter__(self) -> LandmarkExtractor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
        """Ensure input frame is a contiguous 3-channel RGB uint8 array."""
        arr = np.asarray(frame)
        if arr.size == 0:
            raise ValueError("Empty frame provided to landmark extractor")
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            # OpenCV captures in BGR by default, convert to RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"Unexpected frame shape: {arr.shape}")
        return np.ascontiguousarray(arr, dtype=np.uint8)

    @staticmethod
    def _points(values: list | None, expected_count: int) -> tuple[np.ndarray, np.ndarray]:
        """Extract (N, 3) coordinates and (N, 3) mask from landmark points."""
        coordinates = np.zeros((expected_count, 3), dtype=np.float32)
        mask = np.zeros((expected_count, 3), dtype=np.float32)
        if values:
            for index, point in enumerate(values[:expected_count]):
                coordinates[index] = (point.x, point.y, point.z)
                mask[index] = 1.0
        return coordinates, mask

    def extract_structured_frame(
        self, frame: np.ndarray, timestamp_ms: int = 0
    ) -> StructuredLandmarkFrame:
        """Extract structured 3D coordinates [553, 3] and point mask [553]."""
        rgb_frame = self._ensure_rgb(frame)

        if self.model_asset_path is None:
            return StructuredLandmarkFrame(
                coordinates=np.zeros((TOTAL_HOLISTIC_POINTS, 3), dtype=np.float32),
                point_mask=np.zeros(TOTAL_HOLISTIC_POINTS, dtype=np.float32),
                timestamp_ms=float(timestamp_ms),
            )

        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        landmarker = self._get_landmarker()

        if self.running_mode == "video":
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))
        else:
            result = landmarker.detect(mp_image)

        modalities = {
            "left_hand": getattr(result, "left_hand_landmarks", None),
            "right_hand": getattr(result, "right_hand_landmarks", None),
            "pose": getattr(result, "pose_landmarks", None),
            "face": getattr(result, "face_landmarks", None),
        }

        all_coords: list[np.ndarray] = []
        all_masks: list[np.ndarray] = []

        for name in MODALITY_ORDER:
            coords, mask_3d = self._points(modalities[name], self._MODALITY_COUNTS[name])
            all_coords.append(coords)
            all_masks.append(mask_3d[:, 0])  # [N] point mask

        coords_arr = np.concatenate(all_coords, axis=0)  # [553, 3]
        mask_arr = np.concatenate(all_masks, axis=0)      # [553]

        return StructuredLandmarkFrame(
            coordinates=coords_arr,
            point_mask=mask_arr,
            timestamp_ms=float(timestamp_ms),
        )

    def extract_observation(
        self, frame: np.ndarray, timestamp_ms: int = 0
    ) -> LandmarkObservation:
        """Extract flattened observation matching configured representation and feature_dim."""
        if frame is None or np.asarray(frame).size == 0:
            return LandmarkObservation(
                np.zeros(self.feature_dim, dtype=np.float32),
                np.zeros(self.feature_dim, dtype=np.float32),
            )

        structured = self.extract_structured_frame(frame, timestamp_ms=timestamp_ms)

        selected_modalities = ["left_hand", "right_hand"]
        if self.representation in {"hands_pose", "holistic"}:
            selected_modalities.append("pose")
        if self.representation == "holistic":
            selected_modalities.append("face")

        values: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        for name in selected_modalities:
            coords, pt_mask = structured.get_modality(name)
            values.append(coords.reshape(-1))
            masks.append(np.repeat(pt_mask, 3))

        landmarks = np.concatenate(values)
        mask = np.concatenate(masks)

        if landmarks.shape[0] != self.feature_dim:
            if landmarks.shape[0] > self.feature_dim:
                landmarks = landmarks[: self.feature_dim]
                mask = mask[: self.feature_dim]
            else:
                padding = self.feature_dim - landmarks.shape[0]
                landmarks = np.pad(landmarks, (0, padding))
                mask = np.pad(mask, (0, padding))

        return LandmarkObservation(landmarks.astype(np.float32), mask.astype(np.float32))

    def extract(self, frame: np.ndarray, timestamp_ms: int = 0) -> np.ndarray:
        return self.extract_observation(frame, timestamp_ms=timestamp_ms).landmarks
