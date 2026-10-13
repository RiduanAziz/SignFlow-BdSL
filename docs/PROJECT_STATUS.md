# SignFlow-BdSL: Project Status & Migration Tracking

## Repository Metadata
- **Project Title**: SignFlow-BdSL: Robust Real-Time Bangla Sign Language Word Recognition Using Holistic Spatiotemporal Features
- **Repository**: `RiduanAziz/SignFlow-BdSL`
- **Active Git Branch**: `main`
- **Canonical Python Package**: `src/sign_language/` (installed in editable mode)
- **Python Environment**: Python 3.14.7, PyTorch 2.14.1+cpu, MediaPipe 1.0.1, OpenCV 5.0.0.93, Scikit-learn 1.9.1
- **Node Environment**: Node v24.19.0, npm 11.17.0
- **Frontend Test Suite**: 10 / 10 passed (`vitest run`) in `app/frontend`
- **Frontend Production Build**: `tsc && vite build` compiled cleanly to `app/frontend/dist`
- **Backend Test Suite**: 65 / 65 passed (`pytest tests/ -v`), 100% test pass rate
- **Linter Status**: 100% clean (`ruff check src scripts tests` passed with 0 errors)
- **Target Dataset**: BDSLW131 (29,930 still images across 131 classes in `data/raw/bangla-sign-dataset/`, aliased to `data/raw/bdslw131/`)

---

## Migration Phase Tracking

| Phase | Milestone Name | Status | Evidence / Verification |
|---|---|---|---|
| **Phase 0** | Workspace Inspection & Safe Baseline | ✅ COMPLETE | Repository isolated, Git verified, packages installed, tests recorded, audit & plan documents created (`docs/BDSL_MIGRATION_AUDIT.md`, `docs/BDSL_MIGRATION_PLAN.md`). |
| **Phase 1** | BDSLW131 Dataset Inspection & Validation | ✅ COMPLETE | Located and scanned 29,930 JPG images across 131 class directories. Generated `data/manifests/bdslw131_dataset_manifest.csv`, `results/bdslw131_dataset_report.json`, `results/bdslw131_class_distribution.csv`, and `docs/BDSL_DATASET_CARD.md`. Deterministic class-stratified splits (70/15/15) created with 0% data leakage. |
| **Phase 2** | Target Vocabulary & Bengali Labels | ✅ COMPLETE | Single versioned source of truth created at `configs/bdslw131/vocabulary.json` with 131 classes, stable IDs (0-130), verified Bengali script labels, English glosses, and transliterations. Managed via `src/sign_language/config/vocabulary.py`. |
| **Phase 3** | Supported Recognition Task Determination | ✅ COMPLETE | Verified dataset modality as still JPEG images. Primary task defined as Static Image Word Recognition (Task A) + Live Webcam Word Prediction with stability debouncing (Task C). Limitations honestly documented without claiming dynamic video training. |
| **Phase 4** | Preprocessing & Landmark Pipeline | ✅ COMPLETE | Created `scripts/extract_bdslw131_landmarks.py` extracting 553 holistic points (1659 3D normalized coordinates + 1659 visibility masks) using `models/mediapipe/holistic_landmarker.task` to `data/landmarks/bdslw131/*.npz`. Verified on real images. |
| **Phase 5** | BDSLW131 Model Architectures | ✅ COMPLETE | Created `BaselineClassifier` (131 classes) and `RobustHolisticFusion` (131 classes). Added configuration files `configs/bdslw131/baseline.yaml` and `configs/bdslw131/proposed.yaml`. Updated model factory and dimension contracts. |
| **Phase 6** | Training Pipeline & Verification Smoke Run | ✅ COMPLETE | Adapted `scripts/train.py` with manifest loading, stratified evaluation, AdamW optimizer, and checkpoint saving. Executed verified smoke runs generating `checkpoints/bdslw131/baseline_best.pt` and `checkpoints/bdslw131/robust_holistic_fusion_best.pt`. |
| **Phase 7** | Defensible Evaluation & Experiment Reports | ✅ COMPLETE | Updated `scripts/evaluate.py`. Evaluated both checkpoints against test manifest. Exported `results/bdslw131_baseline_evaluation.json`, `results/bdslw131_proposed_evaluation.json`, per-class CSVs, and confusion matrices. |
| **Phase 8** | Robustness Research | ✅ COMPLETE | Adapted `src/sign_language/robustness.py` and executed `scripts/evaluate_robustness.py`. Produced `results/bdslw131_robustness_report.json` benchmarking jitter, translation, scale, and landmark dropout. |
| **Phase 9** | Backend API & WebSocket Migration | ✅ COMPLETE | Updated `src/sign_language/api/main.py` and `src/sign_language/api/inference.py`. Prioritized BdSL checkpoints, added `/api/vocabulary` bilingual response, updated `/predict` with `bengali_label` and `english_gloss`, and integrated bilingual WebSocket payloads for live prediction and committed words. |
| **Phase 10** | SignFlow Frontend Bengali Migration | ✅ COMPLETE | Migrated `app/frontend/`: Header, LiveRecognitionPanel, PracticeMode, TranscriptWorkspace, PostureInspector, HandQualityCoach, HelpModal, and App footer. Displays Bengali Unicode with secondary English glosses. Verified with `vitest` (10/10 passing) and `tsc && vite build` (clean build to `dist/`). |
| **Phase 11** | Test Suite & Continuous Integration | ✅ COMPLETE | Created `tests/test_bdslw131_pipeline.py` (6/6 passing) testing vocabulary integrity, split leakage prevention, checkpoint contract, and synthetic fixtures. Fixed contract dimensions in `tests/test_api.py` (14/14 passing). Total suite passing cleanly. |
| **Phase 12** | Environment, Scripts & Reproducibility | ✅ COMPLETE | Protected large datasets, checkpoints, and caches in `.gitignore`. Verified commands for training, evaluation, inference, and test execution. |
| **Phase 13** | Documentation & Academic Deliverables | ✅ COMPLETE | Updated root `README.md`. Created `docs/MODEL_CARD.md`, `docs/EXPERIMENT_PROTOCOL.md`, and `docs/LOCAL_DEVELOPMENT.md`. Fully preserved attribution and academic integrity. |

---

## Final Verification Summary
- **API Status**: Health check 200 OK (`http://127.0.0.1:8000/health`), trained model loaded (131 classes), MediaPipe Holistic extractor ready.
- **Frontend Status**: Production build generated at `app/frontend/dist`, served directly by FastAPI. Vitest passed 10/10.
- **Data Protection**: `data/raw/`, `data/landmarks/`, `checkpoints/`, `*.npz`, `*.pt` strictly excluded from Git.
- **Academic Integrity**: BDSLW131 explicitly documented as an image dataset; static recognition and live webcam debounced recognition implemented without fabricated dynamic videos.
