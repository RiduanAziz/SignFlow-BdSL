"""Dataset discovery, integrity inspection, and manifest generation for BDSLW131."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 checksum of a file."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for an image (64-bit hex string)."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    decimal_val = 0
    for idx, bit in enumerate(diff.flatten()):
        if bit:
            decimal_val |= 1 << idx
    return f"{decimal_val:016x}"


def hamming_distance(hex1: str, hex2: str) -> int:
    """Compute Hamming distance between two hexadecimal hash strings."""
    return bin(int(hex1, 16) ^ int(hex2, 16)).count("1")


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


def update_canonical_vocabulary_file(
    candidate_vocab: list[dict[str, Any]],
    canonical_path: Path,
) -> dict[str, Any]:
    """Safely update canonical vocabulary file with diffing, validation, and backup."""
    diff_summary: dict[str, Any] = {"additions": [], "removals": [], "modified": []}
    canonical_path = Path(canonical_path)

    if canonical_path.is_file():
        with open(canonical_path, "r", encoding="utf-8") as f:
            existing_vocab = json.load(f)

        existing_by_label = {item["dataset_label"]: item for item in existing_vocab}

        for new_item in candidate_vocab:
            lbl = new_item["dataset_label"]
            if lbl in existing_by_label:
                old_id = existing_by_label[lbl]["class_id"]
                new_id = new_item["class_id"]
                if old_id != new_id:
                    raise ValueError(
                        f"Refusing to update canonical vocabulary: class ID shifted for '{lbl}' "
                        f"(existing ID {old_id} -> candidate ID {new_id}). "
                        "Reassigning class IDs breaks existing model checkpoints and evaluation consistency!"
                    )
                # Preserve manually curated fields (e.g. verified non-empty transliterations)
                old_translit = existing_by_label[lbl].get("transliteration", "")
                if old_translit and not new_item.get("transliteration"):
                    new_item["transliteration"] = old_translit

                if new_item != existing_by_label[lbl]:
                    diff_summary["modified"].append(lbl)
            else:
                diff_summary["additions"].append(lbl)

        for old_item in existing_vocab:
            lbl = old_item["dataset_label"]
            if lbl not in {item["dataset_label"] for item in candidate_vocab}:
                diff_summary["removals"].append(lbl)

        # Create backup before overwriting
        backup_path = canonical_path.with_suffix(".json.bak")
        shutil.copy2(canonical_path, backup_path)
        logger.info("Created backup of canonical vocabulary at %s", backup_path)
    else:
        diff_summary["additions"] = [item["dataset_label"] for item in candidate_vocab]

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(candidate_vocab, f, ensure_ascii=False, indent=2)

    logger.info(
        "Successfully updated canonical vocabulary (%d classes) at %s. Diff: %d additions, %d removals, %d modified",
        len(candidate_vocab),
        canonical_path,
        len(diff_summary["additions"]),
        len(diff_summary["removals"]),
        len(diff_summary["modified"]),
    )
    return diff_summary


def inspect_bdslw131(
    raw_dir: Path,
    manifest_output: Path,
    report_output: Path,
    distribution_output: Path,
    candidate_vocabulary_output: Path,
    canonical_vocabulary_path: Path | None = None,
    check_images: bool = True,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Inspect and validate BDSLW131 dataset with duplicate checks and provenance tracking."""
    logger.info("Starting BDSLW131 inspection on: %s", raw_dir)

    metadata_path = raw_dir / "metadata.csv"

    # Step 1: Discover class directories
    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    discovered_class_count = len(class_dirs)
    logger.info("Discovered %d class subdirectories", discovered_class_count)

    # Step 2: Build Candidate Vocabulary
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
            "transliteration": "",  # Unverified phonetic romanization; never conflate with english_gloss
            "source": "BDSLW131 official folder naming",
        })

    # Save candidate vocabulary artifact
    candidate_vocabulary_output.parent.mkdir(parents=True, exist_ok=True)
    with open(candidate_vocabulary_output, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    logger.info("Saved candidate vocabulary (%d classes) to: %s", len(vocabulary), candidate_vocabulary_output)

    # If canonical vocabulary update is explicitly requested, execute safe update
    vocab_diff: dict[str, Any] = {}
    if canonical_vocabulary_path is not None:
        vocab_diff = update_canonical_vocabulary_file(vocabulary, canonical_vocabulary_path)

    # Step 3: Parse metadata.csv if present
    official_metadata: list[dict[str, str]] = []
    if metadata_path.is_file():
        with open(metadata_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            official_metadata = list(reader)
        logger.info("Loaded metadata.csv with %d recorded rows", len(official_metadata))

    # Step 4: Discover all image files on disk (exact full count first)
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_discovered_files: list[tuple[int, str, Path]] = []
    for cdir in class_dirs:
        folder_name = cdir.name
        cid = folder_to_class_id[folder_name]
        for img_path in sorted(cdir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in supported_extensions:
                all_discovered_files.append((cid, folder_name, img_path))

    total_images_discovered = len(all_discovered_files)
    logger.info("Discovered %d total image files across all classes", total_images_discovered)

    # Apply sample limit to verification loop if configured
    files_to_process = all_discovered_files[:sample_limit] if sample_limit else all_discovered_files

    manifest_rows: list[dict[str, Any]] = []
    class_frequencies: dict[str, int] = Counter()
    image_shapes: set[tuple[int, int, int]] = set()
    corrupt_samples: list[dict[str, str]] = []
    total_bytes = 0

    content_hashes: dict[str, list[dict[str, str]]] = {}
    perceptual_hashes: list[dict[str, Any]] = []

    for class_id, folder_name, img_path in files_to_process:
        bengali, english = parse_bilingual_label(folder_name)
        file_size = img_path.stat().st_size
        total_bytes += file_size
        sample_id = f"bdsl_{class_id:03d}_{img_path.stem}"
        rel_path = f"{folder_name}/{img_path.name}"

        # SHA-256 Content Hash
        sha256_hash = compute_file_sha256(img_path)
        if sha256_hash:
            if sha256_hash not in content_hashes:
                content_hashes[sha256_hash] = []
            content_hashes[sha256_hash].append({"sample_id": sample_id, "label": folder_name, "path": rel_path})

        width, height, channels = 0, 0, 0
        is_valid = True
        error_reason = ""
        dhash_str = ""

        if check_images:
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
                    dhash_str = compute_dhash(img)
                    perceptual_hashes.append({
                        "sample_id": sample_id,
                        "label": folder_name,
                        "dhash": dhash_str,
                    })
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
            "sha256": sha256_hash,
            "is_valid": int(is_valid),
        })

    # Step 5: Duplicate detection analysis
    exact_duplicates = [
        {"sha256": h, "count": len(items), "samples": items}
        for h, items in content_hashes.items()
        if len(items) > 1
    ]

    near_duplicates: list[dict[str, Any]] = []
    # If sample size allows, check near-duplicates with dHash distance <= 2
    if check_images and len(perceptual_hashes) <= 1500:
        for i in range(len(perceptual_hashes)):
            for j in range(i + 1, len(perceptual_hashes)):
                dist = hamming_distance(perceptual_hashes[i]["dhash"], perceptual_hashes[j]["dhash"])
                if dist <= 2:
                    near_duplicates.append({
                        "sample_a": perceptual_hashes[i]["sample_id"],
                        "label_a": perceptual_hashes[i]["label"],
                        "sample_b": perceptual_hashes[j]["sample_id"],
                        "label_b": perceptual_hashes[j]["label"],
                        "hamming_distance": dist,
                    })

    # Step 6: Save manifest CSV
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "filepath", "label", "bengali_label", "english_gloss",
        "class_id", "file_size_bytes", "width", "height", "channels", "sha256", "is_valid"
    ]
    with open(manifest_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    logger.info("Saved dataset manifest (%d records) to: %s", len(manifest_rows), manifest_output)

    manifest_sha256 = compute_file_sha256(manifest_output)

    # Step 7: Save class distribution CSV
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

    # Step 8: Compile Comprehensive Dataset Health Report
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
        "manifest_path": str(manifest_output),
        "manifest_sha256": manifest_sha256,
        "total_classes": len(vocabulary),
        "total_images_discovered": total_images_discovered,
        "samples_processed": len(manifest_rows),
        "sample_limit_applied": sample_limit,
        "valid_labeled_samples": valid_count,
        "excluded_corrupt_samples": len(corrupt_samples),
        "corrupt_samples": corrupt_samples,
        "duplicates_audit": {
            "exact_content_duplicate_groups": len(exact_duplicates),
            "exact_duplicates": exact_duplicates[:20],
            "near_perceptual_duplicates_detected": len(near_duplicates),
            "near_duplicates_sample": near_duplicates[:20],
            "note": "Perceptual near-duplicates computed with dHash (Hamming distance <= 2).",
        },
        "vocabulary_update": vocab_diff,
        "class_balance_summary": {
            "min_samples_per_class": min_count,
            "max_samples_per_class": max_count,
            "mean_samples_per_class": round(mean_count, 2),
            "median_samples_per_class": round(median_count, 2),
        },
        "signer_information": {
            "signer_ids_available": False,
            "note": (
                "Official metadata does not supply individual signer identifiers. "
                "Disjoint sample IDs alone do not guarantee signer-independent evaluation. "
                "Class-stratified splitting is enforced to avoid class leakage."
            ),
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
    parser.add_argument("--raw-dir", "--data-dir", dest="raw_dir", help="Directory containing BDSLW131 raw data.")
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
        "--candidate-vocabulary-output",
        "--vocabulary-output",
        default="results/bdslw131_generated_vocabulary.json",
        dest="vocabulary_output",
        help="Path to output candidate vocabulary JSON (default: results/bdslw131_generated_vocabulary.json).",
    )
    parser.add_argument(
        "--update-canonical-vocabulary",
        type=str,
        nargs="?",
        const="configs/bdslw131/vocabulary.json",
        default=None,
        help="Explicitly update canonical vocabulary file (default target: configs/bdslw131/vocabulary.json).",
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

    canonical_vocab_path = Path(args.update_canonical_vocabulary) if args.update_canonical_vocabulary else None

    report = inspect_bdslw131(
        raw_dir=raw_dir,
        manifest_output=Path(args.manifest_output),
        report_output=Path(args.report_output),
        distribution_output=Path(args.distribution_output),
        candidate_vocabulary_output=Path(args.vocabulary_output),
        canonical_vocabulary_path=canonical_vocab_path,
        check_images=not args.skip_image_check,
        sample_limit=args.sample_limit,
    )

    print("\n================ BDSLW131 DATASET INSPECTION SUMMARY ================")
    print(f"Total Verified Classes: {report['total_classes']}")
    print(f"Total Discovered Images: {report['total_images_discovered']}")
    print(f"Valid Usable Samples: {report['valid_labeled_samples']}")
    print(f"Corrupt / Excluded Samples: {report['excluded_corrupt_samples']}")
    print(f"Exact Duplicate Groups: {report['duplicates_audit']['exact_content_duplicate_groups']}")
    print(f"Mean Samples per Class: {report['class_balance_summary']['mean_samples_per_class']}")
    print(f"Dataset Size on Disk: {report['total_dataset_size_mb']} MB")
    print(f"Manifest Generated: {args.manifest_output}")
    print(f"Candidate Vocabulary: {args.vocabulary_output}")
    if canonical_vocab_path:
        print(f"Canonical Vocabulary Updated: {canonical_vocab_path}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
