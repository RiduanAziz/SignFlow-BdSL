from __future__ import annotations

import cv2
import numpy as np

from .extractor import LandmarkExtractor
from .normalization import normalize_landmarks
from .schema import LandmarkSequence


def resample_sequence(
    sequence: np.ndarray,
    target_length: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly resample a temporal sequence [T, D] to target_length [L, D].

    Preserves the start, middle, and end of dynamic signs without front-truncation.
    """
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Sequence must have shape [T, D], got {arr.shape}")

    t_in, d = arr.shape
    if t_in == 0:
        empty = np.zeros((target_length, d), dtype=np.float32)
        return empty, empty

    if t_in == target_length:
        resampled_mask = mask if mask is not None else np.ones_like(arr, dtype=np.float32)
        return arr.copy(), resampled_mask.copy()

    # Uniform linear interpolation along the time dimension
    orig_times = np.linspace(0.0, 1.0, num=t_in, endpoint=True)
    target_times = np.linspace(0.0, 1.0, num=target_length, endpoint=True)

    resampled = np.zeros((target_length, d), dtype=np.float32)
    for dim in range(d):
        resampled[:, dim] = np.interp(target_times, orig_times, arr[:, dim])

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float32)
        resampled_mask = np.zeros((target_length, d), dtype=np.float32)
        for dim in range(d):
            # Nearest or thresholded interpolation for boolean/float masks
            resampled_mask[:, dim] = np.interp(target_times, orig_times, mask_arr[:, dim])
        resampled_mask = np.where(resampled_mask >= 0.5, 1.0, 0.0).astype(np.float32)
    else:
        resampled_mask = np.ones((target_length, d), dtype=np.float32)

    return resampled, resampled_mask


def compute_temporal_derivatives(
    sequence: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute position, velocity, and acceleration with correct boundary and mask handling."""
    pos = np.asarray(sequence, dtype=np.float32)
    t, _ = pos.shape

    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)

    if t > 1:
        vel[1:] = pos[1:] - pos[:-1]
    if t > 2:
        acc[2:] = vel[2:] - vel[1:-1]

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.float32)
        vel_mask = np.zeros_like(mask_arr)
        if t > 1:
            vel_mask[1:] = mask_arr[1:] * mask_arr[:-1]
        vel = vel * vel_mask

        acc_mask = np.zeros_like(mask_arr)
        if t > 2:
            acc_mask[2:] = vel_mask[2:] * vel_mask[1:-1]
        acc = acc * acc_mask
    else:
        vel_mask = np.ones_like(vel)
        acc_mask = np.ones_like(acc)

    return pos, vel, acc


class LandmarkPipeline:
    """Build a deterministic landmark sequence and feature representations."""

    def __init__(
        self,
        feature_dim: int = 1659,
        sequence_length: int = 64,
        extractor: LandmarkExtractor | None = None,
    ) -> None:
        self.extractor = extractor or LandmarkExtractor(feature_dim=feature_dim)
        self.feature_dim = int(feature_dim or self.extractor.feature_dim)
        self.sequence_length = int(sequence_length)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame into landmark features repeated across sequence_length."""
        arr = np.asarray(frame)
        if arr.size == 0:
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        observation = self.extractor.extract_observation(arr)
        if observation.landmarks.shape[0] != self.feature_dim:
            raise ValueError(
                f"Extractor produced {observation.landmarks.shape[0]} features; expected {self.feature_dim}"
            )
        return observation.landmarks.reshape(1, -1).repeat(self.sequence_length, axis=0)

    def process_video(self, video_path: str) -> LandmarkSequence:
        """Read real video frames sequentially with monotonic timestamps."""
        capture = cv2.VideoCapture(str(video_path))
        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        timestamps_ms: list[float] = []

        try:
            if not capture.isOpened():
                raise ValueError(f"Unable to open video: {video_path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            frame_idx = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                ts_ms = frame_idx * (1000.0 / fps)
                obs = self.extractor.extract_observation(frame, timestamp_ms=int(ts_ms))
                frames.append(obs.landmarks)
                masks.append(obs.mask)
                timestamps_ms.append(ts_ms)
                frame_idx += 1
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"Video contains no readable frames: {video_path}")

        raw_landmarks = np.stack(frames)
        raw_masks = np.stack(masks)
        raw_timestamps = np.array(timestamps_ms, dtype=np.float32)

        return LandmarkSequence(
            landmarks=raw_landmarks,
            mask=raw_masks,
            sequence_mask=np.ones(len(frames), dtype=np.float32),
            timestamps_ms=raw_timestamps,
        )

    def normalize_sequence_with_mask(
        self,
        sequence: np.ndarray,
        mask: np.ndarray | None = None,
        include_dynamics: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize geometric landmarks, resample uniformly, and compute dynamics.

        Returns:
            Tuple of (features, feature_mask) with matching shapes [T, D].
        """
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected raw landmark feature_dim={self.feature_dim}, got {arr.shape[-1]}"
            )

        # 1. Normalize geometric coordinates per frame
        normalized = normalize_landmarks(arr, mask=mask)

        # 2. Resample uniformly to target sequence length
        resampled_pos, resampled_mask = resample_sequence(
            normalized, target_length=self.sequence_length, mask=mask
        )

        # 3. Compute temporal dynamics if requested
        if include_dynamics:
            pos, vel, acc = compute_temporal_derivatives(resampled_pos, mask=resampled_mask)
            t = resampled_pos.shape[0]
            vel_mask = np.zeros_like(resampled_mask)
            acc_mask = np.zeros_like(resampled_mask)
            if t > 1:
                vel_mask[1:] = resampled_mask[1:] * resampled_mask[:-1]
            if t > 2:
                acc_mask[2:] = vel_mask[2:] * vel_mask[1:-1]
            feat = np.concatenate([pos, vel, acc], axis=-1)
            feat_mask = np.concatenate([resampled_mask, vel_mask, acc_mask], axis=-1)
            return feat, feat_mask

        return resampled_pos, resampled_mask

    def normalize_sequence(
        self,
        sequence: np.ndarray,
        mask: np.ndarray | None = None,
        include_dynamics: bool = True,
    ) -> np.ndarray:
        """Normalize geometric landmarks, resample uniformly, and compute dynamics."""
        features, _ = self.normalize_sequence_with_mask(
            sequence, mask=mask, include_dynamics=include_dynamics
        )
        return features
