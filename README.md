# SignFlow-BdSL: Robust Real-Time Bangla Sign Language Word Recognition Using Holistic Spatiotemporal Features

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.11-00bcd4.svg)](https://developers.google.com/mediapipe)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-05998b.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![BDSLW131](https://img.shields.io/badge/Dataset-BDSLW131-emerald.svg)](https://www.kaggle.com/datasets/musaddiquealierfan/bdslw131-bangla-sign-language-word-dataset)

**SignFlow-BdSL** is a comprehensive research platform and assistive web application for real-time Bangla Sign Language (BdSL) word recognition based on the **BDSLW131** dataset. The platform integrates holistic spatiotemporal landmark feature extraction, anatomical coordinate normalization with explicit missingness masks, bilingual Unicode decoding (Bengali script with English glosses), defensible perturbation robustness analysis, and a local-first interactive communication interface.

---

## 1. Research Overview & Academic Integrity

- **Research Title**: *SignFlow-BdSL: Robust Real-Time Bangla Sign Language Word Recognition Using Holistic Spatiotemporal Features*
- **Target Dataset**: [BDSLW131 Bangla Sign Language Word Dataset](https://www.kaggle.com/datasets/musaddiquealierfan/bdslw131-bangla-sign-language-word-dataset) (Musaddique Ali Erfan, 2021).
- **Dataset Modality**: **29,930 still JPEG images** across **131 distinct word classes**.
- **Supported Recognition Tasks**:
  - **Task A — Static Image Word Classification**: Primary offline training and evaluation task.
  - **Task C — Live Webcam Word Recognition**: Captures live frames via MediaPipe Holistic Landmarker with temporal debouncing and stability hysteresis.
- **Academic Integrity Statement**: BDSLW131 comprises labeled still images, *not* dynamic video sequences. In adherence to academic integrity principles, this codebase makes **no claim of trained dynamic temporal video recognition** on BDSLW131 without video training data, and does not synthesize artificial time series as dynamic evidence.

---

## 2. System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │               Input Stream                   │
                    │   (BDSLW131 Images or Live Webcam Feed)      │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │    MediaPipe Holistic Landmarker (553 Pts)   │
                    │   Left Hand (21) + Right Hand (21)           │
                    │   Upper Pose (33) + Face Subset (478)        │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │     Anatomical Normalization & Masking       │
                    │   - Torso / Wrist relative centering         │
                    │   - Spatial bounding-box scaling             │
                    │   - Explicit 1659-dim validity masks         │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │               Model Family                   │
                    │  • Baseline: Deep Multilayer Perceptron      │
                    │  • Proposed: Robust Holistic Modality Fusion │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │         Bilingual Unicode Decoder            │
                    │   Canonical BDSLW131 Mapping (131 Classes)   │
                    │   Bengali Unicode: "একটা", "বই", "ভাত"...   │
                    │   English Gloss: "A / An", "Book", "Rice"... │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │            Interactive Interface             │
                    │   - FastAPI REST API & Low-Latency WebSocket │
                    │   - SignFlow-BdSL React / TypeScript App     │
                    │   - Speech Synthesis (Bengali / English TTS) │
                    └──────────────────────────────────────────────┘
```

---

## 3. Dataset Characteristics (BDSLW131)

| Metric | Verified Value |
| :--- | :--- |
| **Total Classes** | **131** distinct Bangla words |
| **Total Valid Samples** | **29,930** images |
| **File Format** | `.JPG` / `.JPEG` (RGB, varying resolutions) |
| **Mean Samples / Class** | **228.47** (Min: 180, Max: 254) |
| **Split Strategy** | Class-stratified, deterministic (seed 42) |
| **Train Set** | 20,950 samples (70.0%) |
| **Validation Set** | 4,431 samples (14.8%) |
| **Test Set** | 4,549 samples (15.2%) |
| **Data Leakage** | **0.0%** (strictly disjoint sample IDs) |

---

## 4. Key Components

### A. Canonical Vocabulary (`configs/bdslw131/vocabulary.json`)
The single versioned source of truth for all 131 BDSLW131 classes. Every entry contains:
- `class_id`: Stable index (0 to 130).
- `dataset_label`: Original directory name (e.g., `001_A_an_ekta`).
- `bengali_label`: Verified Bengali Unicode representation (e.g., `একটা`).
- `english_gloss`: Verified English translation (e.g., `A / An`).
- `transliteration`: Phonetic romanization (e.g., `ekta`).

### B. Preprocessing & Holistic Feature Pipeline
- Extracts 553 holistic landmarks using Google MediaPipe `holistic_landmarker.task`.
- Produces normalized spatial coordinate vectors `[1659]` synchronized with explicit binary visibility masks `[1659]`.
- Implements anatomical torso-relative centering preventing distortion of spatial signing cues.
- Never converts missing landmarks into genuine zero-valued coordinates without retaining an explicit missingness mask.

### C. Model Architectures
- **Baseline Model (`BaselineClassifier`)**: Deep feedforward network with Layer Normalization and Dropout.
- **Proposed Model (`RobustHolisticFusion`)**: Independent modality sub-encoders (Left Hand, Right Hand, Pose, Face) with missing-modality validity masking, cross-modal attention/fusion, and a 131-class linear projection head.

### D. Robustness Evaluation (`scripts/evaluate_robustness.py`)
Assesses classification degradation across controlled physically meaningful perturbations:
- Spatial jitter / Gaussian sensor noise.
- Anatomical translation (camera pan).
- Uniform coordinate scaling (camera zoom).
- Random and structured modality dropout (simulating occlusion).

---

## 5. Getting Started & Reproduction

### Prerequisites
- Windows 10/11 or Linux
- Python 3.11+ (Tested on Python 3.14.7)
- Node.js 18+ and npm

### Step 1: Environment Setup
```powershell
# Clone and enter directory
cd F:\SignFlow-BdSL

# Install Python package in editable mode
python -m pip install -e . --no-deps

# Install frontend dependencies
cd app/frontend
npm.cmd install
cd ../..
```

### Step 2: Dataset Discovery & Manifest Generation
```powershell
# Inspect the raw dataset (aliased to data/raw/bdslw131/)
python scripts/inspect_bdslw131_dataset.py --data-dir data/raw/bdslw131/

# Generate deterministic class-stratified splits (70/15/15)
python scripts/split_dataset.py --manifest data/manifests/bdslw131_dataset_manifest.csv
```

### Step 3: Landmark Extraction
```powershell
# Extract 553 holistic landmarks to data/landmarks/bdslw131/*.npz
python scripts/extract_bdslw131_landmarks.py --data-dir data/raw/bdslw131/ --output-dir data/landmarks/bdslw131/
```

### Step 4: Model Training (Smoke Test / Full Training)
```powershell
# Train Proposed Robust Holistic Fusion Model (Smoke run: 2 epochs)
python scripts/train.py --config configs/bdslw131/proposed.yaml --epochs 2

# Train Baseline Model (Smoke run: 2 epochs)
python scripts/train.py --config configs/bdslw131/baseline.yaml --epochs 2
```

### Step 5: Offline Model Evaluation
```powershell
# Evaluate Baseline Model
python scripts/evaluate.py --config configs/bdslw131/baseline.yaml --checkpoint checkpoints/bdslw131/baseline_best.pt --output-json results/bdslw131_baseline_evaluation.json

# Evaluate Proposed Model
python scripts/evaluate.py --config configs/bdslw131/proposed.yaml --checkpoint checkpoints/bdslw131/robust_holistic_fusion_best.pt --output-json results/bdslw131_proposed_evaluation.json
```

### Step 6: Robustness Stress-Testing
```powershell
python scripts/evaluate_robustness.py --checkpoint checkpoints/bdslw131/robust_holistic_fusion_best.pt --config configs/bdslw131/proposed.yaml --manifest data/manifests/bdslw131_test_manifest.json --output results/bdslw131_robustness_report.json
```

---

## 6. Running the SignFlow-BdSL Application

### 1. Launch FastAPI Backend Service
```powershell
python -m uvicorn sign_language.api.main:app --host 127.0.0.1 --port 8000
```
- **Health Check**: [`http://127.0.0.1:8000/health`](http://127.0.0.1:8000/health)
- **Model Info**: [`http://127.0.0.1:8000/model/info`](http://127.0.0.1:8000/model/info)
- **Bilingual Vocabulary**: [`http://127.0.0.1:8000/api/vocabulary`](http://127.0.0.1:8000/api/vocabulary)
- **Interactive Swagger Docs**: [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)
- **Production Web App**: [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) *(serves compiled React frontend)*

### 2. Frontend Development Server (Live Hot-Reload)
```powershell
cd app/frontend
npm.cmd run dev
```
Open [`http://127.0.0.1:5173/`](http://127.0.0.1:5173/) in your web browser.

### 3. MediaPipe Standalone Hand Detection Diagnostic
Verify camera and hand detection independently before launching the full recognizer:
```powershell
# Live webcam preview with hand skeleton and bounding boxes (press Q to exit):
python scripts/diagnose_detection.py --source 0

# Test detection on a single image file without GUI:
python scripts/diagnose_detection.py --image path/to/sample.jpg --no-gui
```

---

## 7. Automated Testing Suite

The repository includes comprehensive test suites across backend ML models, APIs, and the frontend web app:

```powershell
# Run all backend regression and pipeline tests (79 tests)
python -m pytest tests/ -v

# Run real-time recognition repair regression suite
python -m pytest tests/test_realtime_repair.py -v

# Run frontend Vitest test suite (12 tests)
cd app/frontend
npm.cmd test -- --run

# Run frontend production build & TypeScript verification
npm.cmd run build
```

---

## 8. Documentation Index

- [`docs/REALTIME_RECOGNITION_AUDIT.md`](docs/REALTIME_RECOGNITION_AUDIT.md): Baseline audit of live recognition flow, buffer contradictions, and hand detection bugs.
- [`docs/REALTIME_RECOGNITION_FIX_LOG.md`](docs/REALTIME_RECOGNITION_FIX_LOG.md): Comprehensive fix log and verification matrix for Phases 1-13.
- [`docs/BDSL_MIGRATION_AUDIT.md`](docs/BDSL_MIGRATION_AUDIT.md): Initial repository audit, component reusability matrix, and baseline contracts.
- [`docs/BDSL_MIGRATION_PLAN.md`](docs/BDSL_MIGRATION_PLAN.md): 14-phase systematic technical migration plan.
- [`docs/BDSL_DATASET_CARD.md`](docs/BDSL_DATASET_CARD.md): Comprehensive dataset card for BDSLW131 (provenance, class distribution, split design, ethics).
- [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md): Model architectures, inputs, outputs, performance metrics, and intentional limitations.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): Rigorous benchmarking protocol, metric formulas, and comparative evaluation standards.
- [`docs/LOCAL_DEVELOPMENT.md`](docs/LOCAL_DEVELOPMENT.md): Step-by-step developer guide for setup, training, testing, and debugging.
- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md): Real-time progress tracker and milestone validation log.

---

## 9. Citation & Attribution

If you utilize this codebase or the BDSLW131 dataset, please cite:

```bibtex
@misc{signflow_bdsl_2026,
  title={SignFlow-BdSL: Robust Real-Time Bangla Sign Language Word Recognition Using Holistic Spatiotemporal Features},
  author={SignFlow-BdSL Research Team},
  year={2026},
  url={https://github.com/RiduanAziz/SignFlow-BdSL}
}

@dataset{erfan2021bdslw131,
  title={BDSLW131: Bangla Sign Language Word Dataset},
  author={Erfan, Musaddique Ali},
  year={2021},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/musaddiquealierfan/bdslw131-bangla-sign-language-word-dataset}
}
```
