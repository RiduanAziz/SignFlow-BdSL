from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sign_language.data import (
    build_manifest,
    generate_dataset_report,
    split_manifest_by_signer,
    split_manifest_stratified,
    validate_signer_independent_split,
    write_manifest_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic split manifests.")
    parser.add_argument("--manifest", default="data/manifests/bdslw131_dataset_manifest.csv")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratified", action="store_true", help="Force class-stratified splitting.")
    parser.add_argument("--prefix", default="", help="Optional filename prefix for outputs.")
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
    pfx = f"{args.prefix}_" if args.prefix else ""

    if has_signers:
        manifest = build_manifest(raw_entries)
        train, validation, test = split_manifest_by_signer(manifest, seed=args.seed)
        if not validate_signer_independent_split(train, validation, test):
            raise ValueError("Generated manifests are not signer-independent")
        outputs = write_manifest_splits(train, validation, test, str(output_dir))
        report = generate_dataset_report([*train, *validation, *test])
        report["seed"] = args.seed
        report["split_signers"] = {
            "train": sorted({item.signer_id for item in train}),
            "validation": sorted({item.signer_id for item in validation}),
            "test": sorted({item.signer_id for item in test}),
        }
        report_path = output_dir / f"{pfx}dataset_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        # Stratified class split
        train, validation, test = split_manifest_stratified(raw_entries, seed=args.seed)
        outputs = {
            "train": output_dir / f"{pfx}train_manifest.json",
            "validation": output_dir / f"{pfx}val_manifest.json",
            "test": output_dir / f"{pfx}test_manifest.json",
        }
        for name, data in [("train", train), ("validation", validation), ("test", test)]:
            with outputs[name].open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        report = {
            "split_type": "stratified_by_class",
            "seed": args.seed,
            "total_samples": len(raw_entries),
            "train_samples": len(train),
            "val_samples": len(validation),
            "test_samples": len(test),
            "train_ratio": round(len(train) / max(len(raw_entries), 1), 3),
            "val_ratio": round(len(validation) / max(len(raw_entries), 1), 3),
            "test_ratio": round(len(test) / max(len(raw_entries), 1), 3),
        }
        report_path = output_dir / f"{pfx}split_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"outputs": {name: str(path) for name, path in outputs.items()}, "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
