# Local Development & Reproduction Guide: SignFlow-BdSL

This guide provides end-to-end instructions for setting up the environment, preparing the BDSLW131 dataset, extracting holistic landmarks, training models, and launching the interactive SignFlow-BdSL application.

---

## 1. Environment Requirements

- **Operating System**: Windows 10/11 or Ubuntu 22.04+ (Windows tested with PowerShell).
- **Python Version**: Python 3.11 to 3.14 (Verified on Python 3.14.7).
- **Node.js**: v18.0.0 or higher.
- **Hardware**: Minimum 8GB RAM. Dedicated NVIDIA GPU recommended for large-batch training, but CPU inference is fully supported.

---

## 2. Workspace Setup

1. **Activate Workspace & Install Package**:
   ```powershell
   cd F:\SignFlow-BdSL
   python -m pip install -e . --no-deps
   ```

2. **Install Frontend Dependencies**:
   ```powershell
   cd app/frontend
   npm.cmd install
   cd ../..
   ```

3. **Verify Dependencies**:
   ```powershell
   python -c "import torch, cv2, mediapipe, fastapi; print('Dependencies verified successfully')"
   ```

---

## 3. Dataset Location & Discovery

The raw BDSLW131 dataset is placed at `data/raw/bangla-sign-dataset/` and aliased via junction to `data/raw/bdslw131/`.

1. **Discover and Inspect Dataset**:
   ```powershell
   python scripts/inspect_bdslw131_dataset.py --data-dir data/raw/bdslw131/
   ```
   Generates:
   - `data/manifests/bdslw131_dataset_manifest.csv` (29,930 rows)
   - `results/bdslw131_dataset_report.json`
   - `results/bdslw131_class_distribution.csv`

2. **Generate Deterministic Class-Stratified Splits**:
   ```powershell
   python scripts/split_dataset.py --manifest data/manifests/bdslw131_dataset_manifest.csv --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
   ```
   Generates:
   - `data/manifests/bdslw131_train_manifest.json` (20,950 samples)
   - `data/manifests/bdslw131_val_manifest.json` (4,431 samples)
   - `data/manifests/bdslw131_test_manifest.json` (4,549 samples)

---

## 4. Preprocessing & Landmark Extraction

Extract 553-point holistic landmarks using the local MediaPipe task file `models/mediapipe/holistic_landmarker.task`:

```powershell
# Extract landmarks across the entire dataset to data/landmarks/bdslw131/
python scripts/extract_bdslw131_landmarks.py --data-dir data/raw/bdslw131/ --output-dir data/landmarks/bdslw131/
```

Extracted `.npz` files contain:
- `landmarks`: Shape `[553, 3]` (1,659 coordinate floats).
- `visibility_mask`: Shape `[553]` (binary visibility flags).
- `point_mask`: Per-point anatomical validity indicators.

---

## 5. Model Training

1. **Baseline MLP Training**:
   ```powershell
   python scripts/train.py --config configs/bdslw131/baseline.yaml --epochs 10 --batch-size 64
   ```
   Checkpoint saved to: `checkpoints/bdslw131/baseline_best.pt`

2. **Proposed Robust Holistic Fusion Training**:
   ```powershell
   python scripts/train.py --config configs/bdslw131/proposed.yaml --epochs 10 --batch-size 64
   ```
   Checkpoint saved to: `checkpoints/bdslw131/robust_holistic_fusion_best.pt`

---

## 6. Model Evaluation & Benchmarking

1. **Run Evaluation on Held-out Test Set**:
   ```powershell
   # Evaluate Baseline
   python scripts/evaluate.py --config configs/bdslw131/baseline.yaml --checkpoint checkpoints/bdslw131/baseline_best.pt --output-json results/bdslw131_baseline_evaluation.json

   # Evaluate Proposed Model
   python scripts/evaluate.py --config configs/bdslw131/proposed.yaml --checkpoint checkpoints/bdslw131/robust_holistic_fusion_best.pt --output-json results/bdslw131_proposed_evaluation.json
   ```

2. **Run Robustness Perturbation Analysis**:
   ```powershell
   python scripts/evaluate_robustness.py --checkpoint checkpoints/bdslw131/robust_holistic_fusion_best.pt --config configs/bdslw131/proposed.yaml --manifest data/manifests/bdslw131_test_manifest.json --output results/bdslw131_robustness_report.json
   ```

---

## 7. Running the Full Application Stack

### Option A: Integrated Production Mode (Recommended)
FastAPI serves both the backend API and the compiled React production frontend:

1. **Build Frontend**:
   ```powershell
   cd app/frontend
   npm.cmd run build
   cd ../..
   ```

2. **Start Server**:
   ```powershell
   python -m uvicorn sign_language.api.main:app --host 127.0.0.1 --port 8000
   ```

3. Open your browser at `http://127.0.0.1:8000/`.

### Option B: Frontend Live Development Mode
1. In Terminal 1:
   ```powershell
   python -m uvicorn sign_language.api.main:app --host 127.0.0.1 --port 8000 --reload
   ```
2. In Terminal 2:
   ```powershell
   cd app/frontend
   npm.cmd run dev
   ```
3. Open your browser at `http://127.0.0.1:5173/`.

---

## 8. Verifying Hand Detection with Standalone Diagnostic

Before running the full web interface, verify that MediaPipe detects your hands independently:

```powershell
# Live webcam preview with hand skeleton and bounding boxes (press Q to exit):
python scripts/diagnose_detection.py --source 0

# Single image test without GUI:
python scripts/diagnose_detection.py --image path/to/sample.jpg --no-gui
```

---

## 9. Running Automated Tests

```powershell
# Run all backend regression and repair tests (79 tests)
python -m pytest tests/ -v

# Run real-time recognition repair regression suite
python -m pytest tests/test_realtime_repair.py -v

# Run frontend Vitest test suite (12 tests)
cd app/frontend
npm.cmd test -- --run

# Compile frontend production bundle & typecheck
npm.cmd run build
```
