# SignFlow-BdSL End-to-End Pipeline

# 1. Dataset Discovery & Manifest Generation
python scripts/inspect_bdslw131_dataset.py --data-dir data/raw/bdslw131/

# 2. Stratified Dataset Splitting
python scripts/split_dataset.py --manifest data/manifests/bdslw131_dataset_manifest.csv

# 3. Holistic Landmark Extraction
python scripts/extract_bdslw131_landmarks.py --data-dir data/raw/bdslw131/ --output-dir data/landmarks/bdslw131/

# 4. Train Proposed Model
python scripts/train.py --config configs/bdslw131/proposed.yaml

# 5. Evaluate Proposed Model
python scripts/evaluate.py --config configs/bdslw131/proposed.yaml --checkpoint checkpoints/bdslw131/robust_holistic_fusion_best.pt --output-json results/bdslw131_proposed_evaluation.json

# 6. Evaluate Robustness
python scripts/evaluate_robustness.py --config configs/bdslw131/proposed.yaml --checkpoint checkpoints/bdslw131/robust_holistic_fusion_best.pt --manifest data/manifests/bdslw131_test_manifest.json --output results/bdslw131_robustness_report.json
