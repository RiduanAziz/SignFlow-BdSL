"""Dataset discovery, integrity inspection, and manifest generation for BDSLW131."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_raw_dataset_dir(configured_dir: str | None = None) -> Path:
    """Resolve the BDSLW131 raw dataset directory with flexible fallback."""
    candidates = []
    if configured_dir:
        candidates.append(Path(configured_dir))
    if os.environ.get("BDSL_RAW_DIR"):
        candidates.append(Path(os.environ["BDSL_RAW_DIR"]))
    if os.environ.get("BDSLW131_DIR"):
        candidates.append(Path(os.environ["BDSLW131_DIR"]))
    candidates.extend([
        Path("data/raw/bdslw131"),
        Path("data/raw/bangla-sign-dataset"),
        Path("data/bdslw131"),
    ])

    for cand in candidates:
        if cand.is_dir() and (cand / "metadata.csv").is_file():
            return cand.resolve()

    # If metadata.csv not directly in root, check if 100+ subdirectories exist
    for cand in candidates:
        if cand.is_dir():
            subdirs = [d for d in cand.iterdir() if d.is_dir()]
            if len(subdirs) >= 50:
                return cand.resolve()

    raise FileNotFoundError(
        f"Could not locate BDSLW131 dataset directory. Checked: {[str(c) for c in candidates]}"
    )


def parse_bilingual_label(folder_name: str) -> tuple[str, str]:
    """Parse `<Bengali>_<English>` folder naming convention."""
    if "_" in folder_name:
        parts = folder_name.split("_", 1)
        bengali = parts[0].strip()
        english = parts[1].strip()
        return bengali, english
    return folder_name.strip(), folder_name.strip()


def inspect_bdslw131(
    raw_dir: Path,
    manifest_output: Path,
    report_output: Path,
    distribution_output: Path,
    vocabulary_output: Path,
    check_images: bool = True,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Inspect and validate BDSLW131 dataset."""
    logger.info("Starting BDSLW131 inspection on: %s", raw_dir)

    metadata_path = raw_dir / "metadata.csv"

    # Step 1: Discover class directories
    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    discovered_class_count = len(class_dirs)
    logger.info("Discovered %d class subdirectories", discovered_class_count)

    # Step 2: Build Canonical Vocabulary (Sorted by folder name for deterministic IDs)
    vocabulary: list[dict[str, Any]] = []
    folder_to_class_id: dict[str, int] = {}

    for class_id, cdir in enumerate(class_dirs):
        folder_name = cdir.name
        bengali, english = parse_bilingual_label(folder_name)
        folder_to_class_id[folder_name] = class_id
        vocabulary.append({
            "class_id": class_id,
            "dataset_label": folder_name,
            "bengali_label": bengali,
            "english_gloss": english,
            "transliteration": english,
            "source": "BDSLW131 official folder naming",
        })

    # Save canonical vocabulary
    vocabulary_output.parent.mkdir(parents=True, exist_ok=True)
    with open(vocabulary_output, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    logger.info("Saved canonical vocabulary (%d classes) to: %s", len(vocabulary), vocabulary_output)

    # Step 3: Parse metadata.csv if present
    official_metadata: list[dict[str, str]] = []
    if metadata_path.is_file():
        with open(metadata_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            official_metadata = list(reader)
        logger.info("Loaded metadata.csv with %d recorded rows", len(official_metadata))

    # Step 4: Discover all image files on disk
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    total_images_discovered = 0
    manifest_rows: list[dict[str, Any]] = []
    class_frequencies: dict[str, int] = Counter()
    image_shapes: set[tuple[int, int, int]] = set()
    corrupt_samples: list[dict[str, str]] = []
    total_bytes = 0

    sample_counter = 0
    for cdir in class_dirs:
        folder_name = cdir.name
        class_id = folder_to_class_id[folder_name]
        bengali, english = parse_bilingual_label(folder_name)

        # Iterate all image files in this class folder
        image_files = sorted([
            f for f in cdir.iterdir()
            if f.is_file() and f.suffix.lower() in supported_extensions
        ])

        for img_path in image_files:
            total_images_discovered += 1
            sample_counter += 1
            if sample_limit and sample_counter > sample_limit:
                break

            file_size = img_path.stat().st_size
            total_bytes += file_size
            sample_id = f"bdsl_{class_id:03d}_{img_path.stem}"
            rel_path = f"{folder_name}/{img_path.name}"

            width, height, channels = 0, 0, 0
            is_valid = True
            error_reason = ""

            # Check image integrity if enabled
            if check_images:
                # Use cv2 with unicode-safe reading
                try:
                    img_bytes = np.fromfile(str(img_path), dtype=np.uint8)
                    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                    if img is None:
                        is_valid = False
                        error_reason = "cv2_imdecode_failed"
                    else:
                        height, width = img.shape[:2]
                        channels = img.shape[2] if img.ndim == 3 else 1
                        image_shapes.add((width, height, channels))
                except Exception as exc:
                    is_valid = False
                    error_reason = str(exc)

            if not is_valid:
                corrupt_samples.append({
                    "sample_id": sample_id,
                    "filepath": rel_path,
                    "reason": error_reason,
                })
            else:
                class_frequencies[folder_name] += 1

            manifest_rows.append({
                "sample_id": sample_id,
                "filepath": rel_path,
                "label": folder_name,
                "bengali_label": bengali,
                "english_gloss": english,
                "class_id": class_id,
                "file_size_bytes": file_size,
                "width": width,
                "height": height,
                "channels": channels,
                "is_valid": int(is_valid),
            })

        if sample_limit and sample_counter > sample_limit:
            break

    # Step 5: Save manifest CSV
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "filepath", "label", "bengali_label", "english_gloss",
        "class_id", "file_size_bytes", "width", "height", "channels", "is_valid"
    ]
    with open(manifest_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    logger.info("Saved dataset manifest (%d records) to: %s", len(manifest_rows), manifest_output)

    # Step 6: Save class distribution CSV
    distribution_output.parent.mkdir(parents=True, exist_ok=True)
    with open(distribution_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "label", "bengali_label", "english_gloss", "sample_count"])
        for item in vocabulary:
            c_id = item["class_id"]
            lbl = item["dataset_label"]
            cnt = class_frequencies.get(lbl, 0)
            writer.writerow([c_id, lbl, item["bengali_label"], item["english_gloss"], cnt])
    logger.info("Saved class distribution to: %s", distribution_output)

    # Step 7: Compile Comprehensive Dataset Health Report
    valid_count = sum(1 for r in manifest_rows if r["is_valid"] == 1)
    counts = list(class_frequencies.values())
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    mean_count = float(np.mean(counts)) if counts else 0.0
    median_count = float(np.median(counts)) if counts else 0.0

    report = {
        "dataset_name": "BDSLW131 (Bangla Sign Language Word Dataset)",
        "source_url": "https://www.kaggle.com/datasets/musaddiquealierfan/bdslw131-bangla-sign-language-word-dataset",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "modality": "still_images",
        "file_format": "JPG",
        "raw_directory": str(raw_dir),
        "total_classes": len(vocabulary),
        "total_images_discovered": total_images_discovered,
        "valid_labeled_samples": valid_count,
        "excluded_corrupt_samples": len(corrupt_samples),
        "corrupt_samples": corrupt_samples,
        "class_balance_summary": {
            "min_samples_per_class": min_count,
            "max_samples_per_class": max_count,
            "mean_samples_per_class": round(mean_count, 2),
            "median_samples_per_class": round(median_count, 2),
        },
        "signer_information": {
            "signer_ids_available": False,
            "note": "Metadata does not supply individual signer identifiers. Stratified splitting by class is enforced to prevent class leakage.",
        },
        "image_resolution_profiles": [
            {"width": w, "height": h, "channels": c} for (w, h, c) in sorted(image_shapes)
        ],
        "total_dataset_size_mb": round(total_bytes / (1024 * 1024), 2),
    }

    report_output.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Saved dataset report to: %s", report_output)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect, validate, and index BDSLW131 dataset.")
    parser.add_argument("--raw-dir", help="Directory containing BDSLW131 raw data.")
    parser.add_argument(
        "--manifest-output",
        default="data/manifests/bdslw131_dataset_manifest.csv",
        help="Path to output manifest CSV.",
    )
    parser.add_argument(
        "--report-output",
        default="results/bdslw131_dataset_report.json",
        help="Path to output dataset report JSON.",
    )
    parser.add_argument(
        "--distribution-output",
        default="results/bdslw131_class_distribution.csv",
        help="Path to output class distribution CSV.",
    )
    parser.add_argument(
        "--vocabulary-output",
        default="configs/bdslw131/vocabulary.json",
        help="Path to output canonical vocabulary JSON.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip reading image contents (fast metadata-only mode).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of images to check for fast dry run.",
    )

    args = parser.parse_args()
    raw_dir = resolve_raw_dataset_dir(args.raw_dir)

    report = inspect_bdslw131(
        raw_dir=raw_dir,
        manifest_output=Path(args.manifest_output),
        report_output=Path(args.report_output),
        distribution_output=Path(args.distribution_output),
        vocabulary_output=Path(args.vocabulary_output),
        check_images=not args.skip_image_check,
        sample_limit=args.sample_limit,
    )

    print("\n================ BDSLW131 DATASET INSPECTION SUMMARY ================")
    print(f"Total Verified Classes: {report['total_classes']}")
    print(f"Total Discovered Images: {report['total_images_discovered']}")
    print(f"Valid Usable Samples: {report['valid_labeled_samples']}")
    print(f"Corrupt / Excluded Samples: {report['excluded_corrupt_samples']}")
    print(f"Mean Samples per Class: {report['class_balance_summary']['mean_samples_per_class']}")
    print(f"Dataset Size on Disk: {report['total_dataset_size_mb']} MB")
    print(f"Manifest Generated: {args.manifest_output}")
    print(f"Canonical Vocabulary: {args.vocabulary_output}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
