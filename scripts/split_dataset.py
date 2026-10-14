from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sign_language.data import (
    build_manifest,
    split_manifest_by_signer,
    split_manifest_stratified,
    validate_signer_independent_split,
    write_manifest_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic split manifests.")
    parser.add_argument("--manifest", default="data/manifests/bdslw131_dataset_manifest.csv", help="Path to input dataset manifest")
    parser.add_argument("--output-dir", default="data/manifests", help="Output directory for split manifests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio (default: 0.15)")
    parser.add_argument("--stratified", action="store_true", help="Force class-stratified splitting.")
    parser.add_argument("--prefix", default=None, help="Optional filename prefix for outputs (defaults to bdslw131 for bdsl manifests).")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        # Fallback to full_manifest.json if present
        cand = Path("data/manifests/full_manifest.json")
        if cand.is_file():
            manifest_path = cand
        else:
            raise FileNotFoundError(f"Manifest file not found: {args.manifest}")

    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8") as f:
            raw_entries = list(csv.DictReader(f))
    else:
        with manifest_path.open("r", encoding="utf-8") as f:
            raw_entries = json.load(f)

    if not raw_entries:
        raise ValueError(f"Input manifest is empty: {manifest_path}")

    # Determine canonical prefix
    if args.prefix is not None:
        prefix = args.prefix
    elif "bdslw131" in manifest_path.name.lower():
        prefix = "bdslw131"
    else:
        prefix = ""
    pfx = f"{prefix}_" if prefix and not prefix.endswith("_") else (prefix if prefix else "")

    # Check if signer IDs are available and diverse
    has_signers = False
    signer_set = set()
    for item in raw_entries:
        sid = item.get("signer_id")
        if sid and sid != "unknown":
            signer_set.add(sid)
    if len(signer_set) >= 3 and not args.stratified:
        has_signers = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if has_signers:
        manifest = build_manifest(raw_entries)
        train, validation, test = split_manifest_by_signer(manifest, seed=args.seed)
        if not validate_signer_independent_split(train, validation, test):
            raise ValueError("Generated manifests are not signer-independent")
        split_type = "signer_independent"
    else:
        # Stratified class split
        train, validation, test = split_manifest_stratified(
            raw_entries,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_type = "stratified_by_class"

    # Enforce strict ID disjointness and uniqueness validation
    def extract_id(entry: Any) -> str:
        if isinstance(entry, dict):
            return str(entry.get("sample_id", ""))
        return str(getattr(entry, "sample_id", ""))

    train_ids = [extract_id(item) for item in train]
    val_ids = [extract_id(item) for item in validation]
    test_ids = [extract_id(item) for item in test]

    # Check for duplicate sample IDs within any split
    for split_name, s_ids in [("train", train_ids), ("validation", val_ids), ("test", test_ids)]:
        if len(s_ids) != len(set(s_ids)):
            from collections import Counter
            dupes = [k for k, v in Counter(s_ids).items() if v > 1]
            raise ValueError(f"Duplicate sample IDs detected in {split_name} split: {dupes[:10]}")

    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    test_id_set = set(test_ids)

    # Check pairwise disjointness
    overlaps = {
        "train_validation": sorted(train_id_set & val_id_set),
        "train_test": sorted(train_id_set & test_id_set),
        "val_test": sorted(val_id_set & test_id_set),
    }
    detected_leaks = {k: v for k, v in overlaps.items() if v}
    if detected_leaks:
        raise ValueError(f"Data leakage detected between generated splits: {detected_leaks}")

    # Write canonical manifests
    outputs = write_manifest_splits(train, validation, test, str(output_dir), prefix=prefix)

    report = {
        "split_type": split_type,
        "seed": args.seed,
        "total_samples": len(raw_entries),
        "train_samples": len(train),
        "val_samples": len(validation),
        "test_samples": len(test),
        "train_ratio": round(len(train) / max(len(raw_entries), 1), 3),
        "val_ratio": round(len(validation) / max(len(raw_entries), 1), 3),
        "test_ratio": round(len(test) / max(len(raw_entries), 1), 3),
        "train_manifest": str(outputs["train"]),
        "val_manifest": str(outputs["val"]),
        "test_manifest": str(outputs["test"]),
        "disjoint_sample_ids": True,
    }
    if has_signers:
        report["split_signers"] = {
            "train": sorted({item.signer_id for item in train}),
            "validation": sorted({item.signer_id for item in validation}),
            "test": sorted({item.signer_id for item in test}),
        }

    report_path = output_dir / f"{pfx}split_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    result = {
        "outputs": {name: str(path) for name, path in outputs.items()},
        "report": str(report_path),
        "disjoint": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
