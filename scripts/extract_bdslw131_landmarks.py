"""Extract MediaPipe holistic landmarks from BDSLW131 images into normalized .npz assets."""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sign_language.landmarks.extractor import LandmarkExtractor
from sign_language.landmarks.normalization import normalize_landmarks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_landmarks_for_manifest(
    manifest_path: Path,
    raw_dir: Path,
    output_dir: Path,
    model_asset_path: Path,
    representation: str = "holistic",
    max_samples: int | None = None,
    overwrite: bool = False,
    batch_size: int = 50,
) -> dict[str, Any]:
    """Process manifest records, extract holistic landmarks, normalize, and save .npz."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not model_asset_path.is_file():
        raise FileNotFoundError(f"MediaPipe task asset not found: {model_asset_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest rows
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if max_samples:
        all_rows = all_rows[:max_samples]

    total_records = len(all_rows)
    logger.info("Starting landmark extraction for %d records -> %s", total_records, output_dir)

    # Initialize extractor with image running mode
    extractor = LandmarkExtractor(
        model_asset_path=model_asset_path,
        representation=representation,
        running_mode="image",
    )

    extracted_count = 0
    skipped_count = 0
    failed_count = 0
    start_time = time.time()

    for idx, row in enumerate(all_rows):
        sample_id = row["sample_id"]
        rel_path = row["filepath"]
        class_id = int(row["class_id"])
        label = row["label"]

        out_path = output_dir / f"{sample_id}.npz"
        if out_path.is_file() and not overwrite:
            skipped_count += 1
            continue

        full_img_path = (raw_dir / rel_path).resolve()
        if not full_img_path.is_file():
            logger.warning("Image not found: %s", full_img_path)
            failed_count += 1
            continue

        try:
            # Unicode-safe image reading for Windows
            img_bytes = np.fromfile(str(full_img_path), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Failed to decode image: %s", full_img_path)
                failed_count += 1
                continue

            # Extract structured frame [553, 3] and mask [553]
            frame_struct = extractor.extract_structured_frame(img)
            raw_coords = frame_struct.coordinates  # [553, 3]
            point_mask = frame_struct.point_mask    # [553]

            # Flatten to coordinate-level mask [1659]
            coord_mask = np.repeat(point_mask, 3)

            # Normalize coordinates anatomically
            norm_coords = normalize_landmarks(
                raw_coords.reshape(1, -1),
                mode="center_scale",
                mask=coord_mask.reshape(1, -1),
            ).flatten()

            # Save clean compressed npz
            np.savez_compressed(
                out_path,
                landmarks=norm_coords.astype(np.float32),
                mask=coord_mask.astype(np.float32),
                raw_landmarks=raw_coords.flatten().astype(np.float32),
                class_id=class_id,
                label=label,
                sample_id=sample_id,
            )
            extracted_count += 1

        except Exception as exc:
            logger.warning("Error processing %s: %s", sample_id, exc)
            failed_count += 1

        if (idx + 1) % batch_size == 0 or (idx + 1) == total_records:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 0.001)
            logger.info(
                "Progress: %d/%d (%.1f%%) | Extracted: %d, Skipped: %d, Failed: %d | %.1f img/s",
                idx + 1,
                total_records,
                100.0 * (idx + 1) / total_records,
                extracted_count,
                skipped_count,
                failed_count,
                rate,
            )

    extractor.close()
    elapsed_total = time.time() - start_time

    summary = {
        "total_requested": total_records,
        "extracted_count": extracted_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "output_dir": str(output_dir),
        "elapsed_seconds": round(elapsed_total, 2),
    }
    logger.info("Landmark extraction complete: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract holistic landmarks from BDSLW131 images.")
    parser.add_argument(
        "--manifest",
        default="data/manifests/bdslw131_dataset_manifest.csv",
        help="Path to BDSLW131 manifest CSV.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/bangla-sign-dataset",
        help="Raw images directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/landmarks/bdslw131",
        help="Output directory for .npz landmark files.",
    )
    parser.add_argument(
        "--model-asset-path",
        default="models/mediapipe/holistic_landmarker.task",
        help="Path to MediaPipe holistic landmarker task asset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Logging interval batch size.",
    )

    args = parser.parse_args()
    summary = extract_landmarks_for_manifest(
        manifest_path=Path(args.manifest),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        model_asset_path=Path(args.model_asset_path),
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )
    print("\nExtraction Summary:", summary)


if __name__ == "__main__":
    main()
