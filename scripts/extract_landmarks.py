from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from sign_language.landmarks.extractor import LandmarkExtractor
from sign_language.landmarks.pipeline import LandmarkPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global worker state to reuse landmarker across videos
_worker_pipeline: LandmarkPipeline | None = None


def _init_worker(model_asset_path: str, representation: str) -> None:
    global _worker_pipeline
    extractor = LandmarkExtractor(
        model_asset_path=model_asset_path,
        representation=representation,
        running_mode="video",
    )
    _worker_pipeline = LandmarkPipeline(extractor=extractor)


def _process_video_item(args: tuple[dict, str, str, bool]) -> tuple[bool, str, str]:
    entry, raw_dir_str, output_dir_str, overwrite = args
    sample_id = str(entry.get("sample_id", "")).strip()
    video_path_str = str(entry.get("video_path", "")).strip()

    raw_dir = Path(raw_dir_str)
    output_dir = Path(output_dir_str)
    output_path = output_dir / f"{sample_id}.npz"
    temp_path = output_dir / f"{sample_id}.tmp.npz"

    if output_path.exists() and not overwrite:
        # Check if file is non-empty and readable
        try:
            with np.load(output_path) as data:
                if "landmarks" in data and "mask" in data:
                    return True, sample_id, "already_exists"
        except Exception:
            pass  # corrupt existing, re-extract

    full_video_path = (raw_dir / video_path_str).resolve()
    if not full_video_path.is_file():
        # Try direct path
        full_video_path = Path(video_path_str).resolve()
        if not full_video_path.is_file():
            return False, sample_id, f"video_not_found: {video_path_str}"

    global _worker_pipeline
    if _worker_pipeline is None:
        return False, sample_id, "worker_pipeline_not_initialized"

    try:
        seq = _worker_pipeline.process_video(str(full_video_path))
        # Atomic write
        np.savez_compressed(
            temp_path,
            landmarks=seq.landmarks,
            mask=seq.mask,
            sequence_mask=seq.sequence_mask,
            timestamps_ms=seq.timestamps_ms,
        )
        if temp_path.exists():
            temp_path.replace(output_path)
        return True, sample_id, ""
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        return False, sample_id, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmarks for all videos in a manifest.")
    parser.add_argument("--manifest", required=True, help="Path to input manifest json")
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw videos")
    parser.add_argument("--output-dir", default="data/landmarks", help="Directory to save extracted landmarks")
    parser.add_argument("--model-asset-path", default="models/mediapipe/holistic_landmarker.task")
    parser.add_argument("--representation", default="holistic", choices=["hands", "hands_pose", "holistic"])
    parser.add_argument("--workers", type=int, default=2, help="Number of workers (conservative default: 2)")
    parser.add_argument("--max-samples", "--limit", type=int, default=None, dest="limit", help="Max samples to process")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        parser.error(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.limit:
        manifest = manifest[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_args = [(entry, args.raw_dir, args.output_dir, args.overwrite) for entry in manifest]

    logger.info("Extracting landmarks for %d videos using %d workers...", len(manifest), args.workers)

    success_count = 0
    fail_count = 0
    skipped_count = 0
    failures: list[dict[str, str]] = []

    if args.workers <= 1:
        _init_worker(args.model_asset_path, args.representation)
        for item in task_args:
            ok, sid, msg = _process_video_item(item)
            if ok:
                if msg == "already_exists":
                    skipped_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                failures.append({"sample_id": sid, "error": msg})
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.model_asset_path, args.representation),
        ) as executor:
            for ok, sid, msg in executor.map(_process_video_item, task_args):
                if ok:
                    if msg == "already_exists":
                        skipped_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
                    failures.append({"sample_id": sid, "error": msg})

    logger.info(
        "Extraction completed: %d newly extracted, %d skipped/existing, %d failed out of %d total.",
        success_count,
        skipped_count,
        fail_count,
        len(manifest),
    )
    if failures:
        fail_log = output_dir / "extraction_failures.json"
        with fail_log.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        logger.warning("Recorded %d failures to %s", len(failures), fail_log)


if __name__ == "__main__":
    main()
