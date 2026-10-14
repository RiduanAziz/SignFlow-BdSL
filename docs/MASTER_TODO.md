# Master TODO

## Overall Status
- [x] Dataset verified (metadata inventory and 11,980 valid extracted holistic landmark sequences)
- [x] Canonical landmark schema defined and tested (553 points: Hands, Pose, Face)
- [x] MediaPipe Tasks extraction verified on real task asset
- [x] Signer-independent splits generated in `data/manifests/`
- [x] Baseline models (MLP, LSTM, GRU, Temporal Transformer) implemented and trained on real data
- [x] Proposed `RobustHolisticFusionClassifier` implemented and trained on real data
- [x] Held-out signer evaluation executed and verified (`scripts/evaluate.py`)
- [x] Multi-noise spatial/temporal robustness evaluation executed and verified (`scripts/evaluate_robustness.py`)
- [x] Real-time inference and webcam pipeline hardened (`scripts/realtime.py`)
- [x] Singleton FastAPI service implemented and tested (`test_api.py`)
- [x] Automated thesis reporting and figure generation implemented (`scripts/generate_report.py`)
- [x] CI workflow, Docker setup, and pyproject dependencies repaired
- [x] Full test suite green (54 tests passing in `pytest -q`, 10 in `vitest`)

---

## Phase Details

### Phase 0 — Foundation
- [x] Python package structure established in `src/sign_language/`
- [x] `pyproject.toml` unified with testing dependencies
- [x] Reference repo review completed in `docs/REFERENCE_REPO_REVIEW.md`
- [x] Known defect audit cataloged in `docs/PROJECT_AUDIT.md`
- [x] CI workflow updated for clean test execution

### Phase 1 — Architecture Consolidation
- [x] Canonical package selected as `src/sign_language/`
- [x] Compatibility wrappers documented
- [x] Import and package layout validated across all scripts

### Phase 2 — Configuration Management
- [x] One authoritative configuration system (`configs/base.yaml`, `configs/experiments/proposed.yaml`)
- [x] Dynamic class count derived from vocabulary mapping
- [x] Dimension and model name validation in `src/sign_language/config/loader.py`

### Phase 3 & 4 — Dataset Manifest & Signer-Independent Splits
- [x] WLASL JSON ingestion repaired: deterministic class ID derivation, no fallback to 0
- [x] Path traversal check in place for safe dataset resolution
- [x] Signer-independent splits generated with zero signer overlap across train/val/test
- [x] Manifests stored under `data/manifests/`

### Phase 5 & 6 — Landmark Schema & MediaPipe Extraction
- [x] Canonical 553-point holistic landmark schema (`src/sign_language/landmarks/schema.py`)
- [x] MediaPipe Tasks `RunningMode.VIDEO` with monotonic timestamps
- [x] OpenCV BGR->RGB conversion enforced
- [x] Extractor worker reuse and atomic `.tmp.npz` writes
- [x] Tested with real `models/mediapipe/holistic_landmarker.task` asset

### Phase 7 & 8 — Normalization, Dynamics & Batches
- [x] Structured 3D coordinate normalization with landmark visibility masks
- [x] Boundary-safe velocity and acceleration computation
- [x] Uniform sequence resampling (replaces front-truncation)
- [x] `SignLanguageDataset` and `collate_sequences` load masks, sequence lengths, and landmarks reliably

### Phase 9 & 10 — Models & Proposed Fusion Architecture
- [x] MLP, LSTM, GRU, Temporal Transformer baselines supported in model factory
- [x] Transformer padding mask polarity fixed (2D boolean mask `[B, T]`)
- [x] Proposed `RobustHolisticFusionClassifier` implemented with modality encoders, bidirectional GRU streams, temporal pooling, presence gating, and multimodal fusion
- [x] Tested in unit tests (`test_phase5_transformer_multimodal.py`, `test_models.py`)

### Phase 11 & 12 — Training & Held-Out Evaluation
- [x] `scripts/train.py` trained both baseline and proposed models on real local `.npz` sequences
- [x] Best checkpoints saved with rich metadata (`models/temporal_transformer_best.pt`, `models/robust_holistic_fusion_best.pt`)
- [x] `scripts/evaluate.py` evaluated models on real held-out test data (`results/evaluation_report.json`)

### Phase 13 & 14 — Robustness Evaluation & Ablations
- [x] Deterministic spatial disturbances (jitter, translation, scale, landmark dropout)
- [x] Deterministic temporal disturbances (frame drop, frame duplicate, sequence truncate) with mask alignment
- [x] Recognition degradation evaluated across severities [0.0, 0.1, 0.3, 0.5] (`results/robustness/report.json`, `results/robustness/proposed_report.json`)

### Phase 15 & 16 — Real-Time Inference & API
- [x] `RealTimePredictor` requires valid checkpoint and raises `FileNotFoundError` on missing models
- [x] BGR->RGB conversion, rolling sequence buffer, debounce smoothing, and FPS profiling in `scripts/realtime.py`
- [x] Singleton `RealTimePredictor` in FastAPI lifespan, truthful `/health` and `/model/info`, input validation on `/predict`
- [x] API test suite passing

### Phase 17 & 18 — Reporting & Documentation
- [x] `scripts/generate_report.py` generates training curves, robustness curves, and markdown research summary
- [x] Research logs updated in `docs/IMPLEMENTATION_LOG.md`
- [x] Project status updated in `docs/PROJECT_STATUS.md`

### Phase 19 & 20 — Acceptance Criteria
- [x] 34 passed unit and integration tests in `pytest -q`
- [x] All 8 script entry points support `--help`
