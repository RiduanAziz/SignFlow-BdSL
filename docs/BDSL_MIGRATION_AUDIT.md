# SignFlow-BdSL Migration Audit

**Audit Date**: October 13, 2026  
**Project**: SignFlow-BdSL: Robust Real-Time Bangla Sign Language Word Recognition Using Holistic Spatiotemporal Features  
**Repository**: `RiduanAziz/SignFlow-BdSL` (`f:\SignFlow-BdSL`)  
**Active Git Branch**: `main`  
**Remote URL**: `https://github.com/RiduanAziz/SignFlow-BdSL.git`  
**Isolation Status**: Verified. Editable package `sign-language-recognition` (0.1.0) reinstalled to point exclusively to `F:\SignFlow-BdSL\src`. No linkage or dependency on original ASL workspace.

---

## 1. Environment & Dependency Baseline

| Component | Detected Version | Execution Notes |
|---|---|---|
| **Operating System** | Windows 11 (win32) | Native PowerShell execution environment |
| **Python** | 3.14.7 (`C:\Python314\python.exe`) | Global Python environment with required packages |
| **PyTorch** | 2.14.1+cpu | CPU-only PyTorch build; tensor operations verified |
| **TorchVision** | 0.29.1 | Vision operations verified |
| **OpenCV** | 5.0.0.93 (`opencv-python`, `opencv-contrib-python`) | BGR image decoding and frame manipulation verified |
| **MediaPipe** | 1.0.1 | MediaPipe Tasks Python API; HolisticLandmarker task asset verified |
| **NumPy** | 2.5.3 | Array processing and mathematical operations verified |
| **Scikit-Learn** | 1.9.1 | Metric computations and classification utilities verified |
| **FastAPI / Uvicorn** | FastAPI 0.142.2 / Uvicorn 0.54.0 | ASGI web backend with WebSocket streaming |
| **HTTPX / WebSockets** | HTTPX 0.28.1 / WebSockets 17.2 | Installed during audit for test client and WS streaming |
| **Pytest** | 9.1.1 | Python unit and integration testing engine |
| **Node.js / npm** | Node v24.19.0 / npm 11.17.0 | Frontend build engine (`npm.cmd` required under PowerShell) |
| **MediaPipe Task Asset** | `models/mediapipe/holistic_landmarker.task` | 13.6 MB bundle verified present locally |

---

## 2. Pre-Existing Test Baseline & Known Failures

Prior to making any architectural modifications, the test suites across backend and frontend were executed and recorded:

### Frontend Test Baseline (`app/frontend`)
- **Suite**: Vitest 5.0.3 (`npm.cmd test`)
- **Result**: **10 passed, 0 failed** in 13.92s.
- **Production Build**: `npm.cmd run build` passed cleanly in 15.81s (`dist/index.html` 1.50 kB, CSS 31.44 kB, JS 255.85 kB).

### Backend Test Baseline (`tests/`)
- **Total Test Files**: 20 files
- **Total Tests Collected**: 62 tests
- **Passed**: 59 tests
- **Pre-Existing Failures**: 3 tests (documented below)

### Analysis of Pre-Existing Failures
1. **`tests/test_phase7_realtime.py::test_prediction_payload_and_predictor_shape`**:
   - *Failure*: `ValueError: Feature dimension mismatch: expected 1659, got 4977`.
   - *Root Cause*: Default `RealTimePredictor` input dimension was configured to 1659 (pure positional coordinates: 553 points × 3), but the test passed a dynamic feature sequence with 4977 dimensions (position + velocity + acceleration: 1659 × 3).
2. **`tests/test_api.py::test_offline_online_preprocessing_parity`**:
   - *Failure*: `assert (64, 4977) == (64, 1659)`.
   - *Root Cause*: Same parity assertion assumption: offline pipeline generated dynamic derivatives (4977), while online test predictor expected raw positional dimension (1659).
3. **`tests/test_config_phase2.py::test_experiment_config_composes_dataset_and_model_settings`**:
   - *Failure*: `assert 1659 == 4977` (`assert config.model.input_dim == 4977`).
   - *Root Cause*: `configs/experiments/lstm.yaml` defines `input_dim: 1659` while the unit test explicitly asserted `4977`.

---

## 3. Dataset Discovery Findings (BDSLW131)

- **Local Path**: `data/raw/bangla-sign-dataset/`
- **Total Identified Classes**: **131 distinct folders**
- **Folder Naming Convention**: Bilingual format: `<Bengali_Unicode>_<English_Gloss>` (e.g., `অপেক্ষা করো_Wait`, `ভালোবাসা_Love`, `শূন্য_Zero`).
- **Sample Count**: **29,930 total images** (verified across `metadata.csv` and folder listings).
- **Class Balance**: High balance, average ~228.5 images per class (mode: 230 images).
- **Data Modality**: **STILL IMAGES (JPG)** captured via smartphone cameras.
  > **CRITICAL ARCHITECTURAL FINDING**: The BDSLW131 dataset contains **still images only**, not isolated video clips or dynamic sign recordings. In accordance with Phase 3 non-negotiable requirements, the valid target task is **Static Image Gesture / Word Classification** (Task A). We will not manufacture synthetic videos or fabricate dynamic video signs. Real-time webcam inference in SignFlow will extract holistic landmarks from individual frames (with temporal stability smoothing / debounce), mapping to the 131 static Bengali classes.
- **Signer IDs**: `metadata.csv` provides only `filepath` and `label`. Signer identifiers are not provided by the dataset publishers. In accordance with non-negotiable rules, this limitation will be documented honestly; random stratified splitting by class will be used with strict deduplication checks.

---

## 4. File-by-File Migration Classification

| File / Path | Starting Status | Migration Action | Rationale |
|---|---|---|---|
| **`pyproject.toml`** | ASL description & package name | **Adapt** | Update project name to `signflow-bdsl`, update description, ensure dependencies reflect Python 3.14 / BDSLW131 requirements. |
| **`requirements.txt`** | ASL requirements | **Adapt** | Lock tested versions (httpx, websockets, torch 2.14, opencv 5.0, mediapipe 1.0.1). |
| **`README.md`** | ASL thesis overview | **Replace** | Replace completely with SignFlow-BdSL documentation, BDSLW131 facts, quick start guide, and academic citations. |
| **`Dockerfile` / `docker-compose.yml`** | ASL containerization | **Adapt** | Update labels, maintain multi-stage Node + Python build. |
| **`configs/base.yaml`** | ASL 100 classes, input_dim 1659 | **Adapt** | Update default num_classes to 131, project name to `signflow-bdsl`. |
| **`configs/dataset.yaml`** | 100 WLASL gloss labels | **Replace** | Point to BDSLW131 dataset, 131 Bengali classes. |
| **`configs/experiments/`** | WLASL100 / WLASL20 configs | **Archive / Replace** | Create `configs/bdslw131/` with `dataset.yaml`, `baseline.yaml` (MLP/Linear), `proposed.yaml` (Holistic Landmark Fusion Classifier), `vocabulary.json`. |
| **`models/*.pt` (ASL checkpoints)** | 8 ASL/WLASL model checkpoints | **Archive / Retain without active loading** | Never load ASL checkpoints in Bangla inference; isolate. |
| **`models/mediapipe/holistic_landmarker.task`** | MediaPipe model asset (13.6 MB) | **Retain** | Language-agnostic computer vision asset for extracting holistic landmarks (hands, pose, face). |
| **`src/sign_language/landmarks/extractor.py`** | MediaPipe holistic extractor | **Retain & Adapt** | Supports image running mode; verify RGB conversion and mask generation. |
| **`src/sign_language/landmarks/normalization.py`** | Anatomical normalization | **Retain** | Coordinate-level normalization (scale and translation invariance) applies directly to BdSL landmarks. |
| **`src/sign_language/landmarks/pipeline.py`** | Pipeline with sequence resampling | **Adapt** | Add static landmark extraction mode for single-image processing and image-based dataset building. |
| **`src/sign_language/landmarks/schema.py`** | 553 holistic points / 1659 coords | **Retain** | Language-agnostic landmark topology (21 LH, 21 RH, 33 Pose, 478 Face). |
| **`src/sign_language/data/ingestion.py`** | WLASL JSON video loader | **Adapt / Extend** | Add `BDSLW131ImageIngestion` to discover, validate, and index BDSLW131 image directory structure and `metadata.csv`. |
| **`src/sign_language/data/manifest.py`** | Manifest validation | **Adapt** | Support image manifests with verified Bengali labels and filepaths. |
| **`src/sign_language/data/dataset.py`** | Video sequence dataset | **Adapt** | Support image landmark dataset (`BDSLW131Dataset`) with landmark caching. |
| **`src/sign_language/data/splitting.py`** | Signer-independent splitting | **Adapt** | Support stratified class splitting when signer IDs are absent; retain duplicate sample rejection. |
| **`src/sign_language/data/dataloader.py`** | Variable length collator | **Retain & Adapt** | Retain mask collation for batches. |
| **`src/sign_language/models/baseline.py`** | Landmark MLP baseline | **Adapt** | Configure for 131 BDSLW131 classes and static landmark inputs. |
| **`src/sign_language/models/fusion.py`** | `RobustHolisticFusionClassifier` | **Adapt** | Adapt for BDSLW131 static holistic feature classification with modality gating (hands, pose, face). |
| **`src/sign_language/models/transformer.py` / `lstm.py` / `gru.py`** | Temporal sequence models | **Retain for sequence/webcam mode** | Preserved for live camera sequence buffer processing. |
| **`src/sign_language/models/factory.py`** | Model construction factory | **Adapt** | Support 131 output classes and static vs temporal model instantiation. |
| **`src/sign_language/training/trainer.py`** | PyTorch training loop | **Adapt** | Support BDSLW131 training, class weighting, validation tracking, rich checkpoint saving with vocabulary metadata. |
| **`src/sign_language/evaluation/`** | Evaluation metrics and reports | **Adapt** | Output reports strictly to `results/bdslw131_*` with per-class precision, recall, F1, and confusion matrices. |
| **`src/sign_language/robustness.py`** | Spatial perturbations | **Retain & Adapt** | Perturbations (spatial noise, translation, occlusion) apply directly to BDSLW131 landmarks. |
| **`src/sign_language/api/main.py`** | FastAPI app & WebSocket | **Adapt** | Point to BDSLW131 models, load canonical Bengali vocabulary, render Bengali Unicode in API responses. |
| **`src/sign_language/api/inference.py`** | `RealTimePredictor` | **Adapt** | Fix 1659 vs 4977 dimension handling, load BDSLW131 canonical vocabulary, return Bengali display labels + English glosses. |
| **`app/frontend/src/App.tsx`** | Main frontend layout | **Adapt** | Update branding to SignFlow-BdSL, support Bengali rendering, update settings and default states. |
| **`app/frontend/src/components/*`** | React UI components | **Adapt** | Update text, guidance, practice mode, gesture replay, and transcript to Bengali/English bilingual support. |
| **`app/frontend/src/types/index.ts`** | TypeScript interfaces | **Adapt** | Add `bengali_label`, `english_gloss`, update vocabulary types. |
| **`scripts/inspect_bdslw131_dataset.py`** | Does not exist | **Create** | Full dataset discovery, integrity check, corrupt image detection, report generation. |
| **`scripts/extract_bdslw131_landmarks.py`** | Does not exist | **Create** | MediaPipe holistic landmark extraction for BDSLW131 images to `.npz`. |
| **`scripts/train.py`** | ASL training CLI | **Adapt** | Add `--config configs/bdslw131/...` support for BDSLW131 training runs. |
| **`scripts/evaluate.py`** | ASL evaluation CLI | **Adapt** | Evaluate BDSLW131 checkpoints, export clean evaluation JSON and confusion matrix. |
| **`scripts/realtime.py`** | ASL OpenCV desktop demo | **Adapt** | Display Bengali text on screen using PIL / Unicode overlay. |
| **`tests/test_*.py`** | ASL test suite | **Adapt & Extend** | Update tests to validate Bengali vocabulary, BDSLW131 image ingestion, correct dimension handling. |
