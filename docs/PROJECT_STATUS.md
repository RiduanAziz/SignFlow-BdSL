# SignFlow-BdSL: Project Status, Audit Remediation & Verification Log

## Repository Metadata
- **Project Title**: SignFlow-BdSL: Robust Real-Time Bangla Sign Language Word Recognition Using Holistic Spatiotemporal Features
- **Repository**: `RiduanAziz/SignFlow-BdSL`
- **Active Git Branch**: `main`
- **Canonical Python Package**: `src/sign_language/` (installed in editable mode)
- **Python Environment**: Python 3.14.7, PyTorch 2.14.1+cpu, MediaPipe 1.0.1, OpenCV 5.0.0.93, Scikit-learn 1.9.1
- **Node Environment**: Node v24.19.0, npm 11.17.0
- **Backend Test Suite**: **72 / 72 passed** (`python -m pytest tests/ -v`), 100% test pass rate
- **Frontend Test Suite**: **10 / 10 passed** (`npm.cmd test -- --run`) in `app/frontend`
- **Frontend Production Build**: `tsc && vite build` compiled cleanly to `app/frontend/dist`
- **Linter Status**: **100% clean** (`python -m ruff check src scripts tests` passed with 0 errors)
- **Target Dataset**: BDSLW131 (29,930 still images across 131 classes in `data/raw/bangla-sign-dataset/`, aliased to `data/raw/bdslw131/`)

---

## Audit Remediation Status Matrix

| Audit Area | Critical Finding / Defect | Remediation Implemented | Verification Evidence |
|---|---|---|---|
| **P0-A: Manifest Naming** | Mismatched split filenames between scripts and tests | Unified on canonical paths: `bdslw131_dataset_manifest.csv`, `bdslw131_train_manifest.json`, `bdslw131_val_manifest.json`, `bdslw131_test_manifest.json` across all split branches, configs, and consumers. | `scripts/split_dataset.py`, `src/sign_language/data/splitting.py`, `tests/test_bdslw131_pipeline.py` |
| **P0-A: Data Leakage & Fail-Closed Loading** | Risk of full dataset manifest being silently trained on, contaminating validation/test sets | `scripts/train.py` now enforces fail-closed manifest validation: rejects full discovery manifests (`dataset_manifest.csv`), enforces pairwise disjointness (`train_val`, `train_test`, `val_test`), and fails with clear actionable error. | Unit test `test_training_fails_closed_on_full_discovery_manifest` & `test_training_detects_deliberate_split_leakage` PASSED |
| **P0-A: Validation & Checkpoint Selection** | Validation loop bypassed if default path missing, leading to last checkpoint mislabeled as best | Training loop constructs validation loader from canonical `bdslw131_val_manifest.json`, evaluates after each epoch, selects best checkpoint based on `primary_metric` (`macro_f1`), tracks best epoch and metric value, and distinctly labels best vs last checkpoints. | `scripts/train.py` lines 180-450 |
| **P0-A: Evaluation Provenance** | Evaluation scripts defaulted to legacy paths without recording cryptographic provenance | `scripts/evaluate.py` and `scripts/evaluate_robustness.py` resolve test manifest from config/CLI, reject full discovery manifests, and record SHA-256 hashes of checkpoint, manifest, and vocabulary. | `scripts/evaluate.py`, `scripts/evaluate_robustness.py` |
| **P0-B: Leakage Detection Tests** | Tests skipped when expected filenames did not match split outputs | Created deterministic end-to-end tests verifying canonical split paths, mutual disjointness, determinism across repeated seeds, and deliberate split leakage rejection. | `tests/test_bdslw131_pipeline.py` (all 11 tests PASSED) |
| **P1-A: Dataset Integrity & Duplicates** | Sample counting stopped on `--sample-limit`; duplicate images undetected | `scripts/inspect_bdslw131_dataset.py` discovers all files on disk before applying limit, computes SHA-256 content hashes for exact duplicate detection, and computes dHash (64-bit) for perceptual near-duplicates (Hamming distance <= 2). | `scripts/inspect_bdslw131_dataset.py` |
| **P1-B: Canonical Vocabulary Protection** | Inspector assigned English gloss to transliteration; risked overwriting canonical vocabulary | Candidate vocabulary generated separately (`results/bdslw131_generated_vocabulary.json`) with `transliteration: ""`. `--update-canonical-vocabulary` required to update canonical file, which diffs classes, verifies stable IDs, preserves curated transliterations, and writes `.bak` backup. | `test_inspect_dataset_preserves_canonical_vocabulary_and_transliteration` PASSED |
| **P1-C: Feature Integrity & Masks** | Cached features missing finite value check; missing landmarks risking zero-coordinate ambiguity | `SignLanguageDataset` validates finite values (rejects NaN/Inf), asserts mask shape alignment, and tracks feature accounting (`total_records`, `missing_count`, `max_missing_ratio`). Explicit binary masks preserved through normalization. | `src/sign_language/data/dataset.py` |
| **P1-D: Configuration & Architecture Contracts** | `MLPClassifier` factory ignored `hidden_dim: 256` from YAML | Updated `MLPClassifier` and `src/sign_language/models/factory.py` to parameterize `hidden_dim` and `dropout`. Updated `baseline.yaml` and `proposed.yaml` with explicit dataset split blocks. | `src/sign_language/models/baseline.py`, `src/sign_language/models/factory.py` |
| **P2-A: Inference Efficiency** | `/predict` performed duplicate forward pass (`predict_label` + `predict_logits`) | Implemented `predict_with_logits()` in `RealTimePredictor` which performs exactly ONE forward pass and returns both raw logits and prediction metadata. | Regression test `test_single_forward_pass_per_predict_request` PASSED (forward count == 1) |
| **P2-B: API Security & State Safety** | Wildcard `["*"]` CORS; WebSocket state leakage on reject candidate | Restricted CORS default to local development origins (`localhost:5173`, `127.0.0.1:5173`, etc.) with `CORS_ORIGINS` override. Updated `reject_candidate` in WebSocket to clear sequence and mask buffers. | `test_cors_restricted_origins` PASSED |
| **P2-C: Frontend Usability & Unicode** | Bengali labels and state freshness | Bengali script preserved as primary label; English glosses displayed secondary. Replay buffer refreshed on navigation to replay tab. Stale-data and model-offline states explicitly displayed. | `app/frontend/src/App.tsx`, `Header.tsx` |
| **P2-D: Dependencies & Environment** | Inconsistent requirement pins across files | Reconciled `pyproject.toml` and `requirements.txt` to align on Python 3.11+ / tested Python 3.14 environment with PyTorch 2.14, MediaPipe 1.0.1, FastAPI, and Scikit-learn. | `pyproject.toml`, `requirements.txt` |
| **P2-F: Robustness Methodology** | Perturbations on static images lacked methodology documentation | Documented that BDSLW131 comprises still images: spatial noise benchmarks geometric coordinate perturbation, while temporal perturbations evaluate frame sequence consistency. | `scripts/evaluate_robustness.py`, `docs/EXPERIMENT_PROTOCOL.md` |
| **P2-G: Historical Results Flagging** | Pre-fix smoke runs may have used ambiguous manifest paths | Explicitly flagged pre-remediation experimental runs as historical. Full training runs must be executed using canonical split manifests. | `docs/EXPERIMENT_PROTOCOL.md`, `README.md` |

---

## Test Execution Results

```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1
rootdir: F:\SignFlow-BdSL, configfile: pyproject.toml
collected 72 items

tests/test_api.py .................                                      [ 22%]
tests/test_bdslw131_pipeline.py ...........                              [ 37%]
tests/test_config_phase2.py ...                                          [ 41%]
tests/test_continuous_inference.py .....                                 [ 48%]
tests/test_dataset.py .                                                  [ 50%]
tests/test_dataset_phase1.py ..                                          [ 52%]
tests/test_dataset_phase3.py ..                                          [ 55%]
tests/test_eda_and_landmarks_phase2_3.py ..                              [ 58%]
tests/test_evaluation_phase10.py ..                                      [ 61%]
tests/test_features.py ..                                                [ 63%]
tests/test_features_phase8.py ...                                        [ 68%]
tests/test_landmarks.py ..                                               [ 70%]
tests/test_master_regressions.py .....                                   [ 77%]
tests/test_models.py .                                                   [ 79%]
tests/test_normalization_fix.py ...                                      [ 83%]
tests/test_phase4_baseline_models.py ..                                  [ 86%]
tests/test_phase5_transformer_multimodal.py .                            [ 87%]
tests/test_phase6_robustness.py .                                        [ 88%]
tests/test_phase7_realtime.py .                                          [ 90%]
tests/test_robustness_phase13_14.py ....                                 [ 95%]
tests/test_sequence_phase7.py ...                                        [100%]

======================= 72 passed, 1 warning in 10.33s ========================
```

- **Frontend Tests**: 10 passed in `app/frontend` (`vitest run`).
- **Frontend Build**: Built clean production bundle in `app/frontend/dist` with `tsc && vite build`.
- **Linter**: 0 errors with `python -m ruff check src scripts tests`.

---

## Remaining Actions & Next Steps
1. **Full Dataset Landmark Extraction**: When raw dataset images are present at `data/raw/bdslw131/`, execute:
   ```powershell
   python scripts/extract_bdslw131_landmarks.py --data-dir data/raw/bdslw131/ --output-dir data/landmarks/
   ```
2. **Canonical Split Generation**: Run:
   ```powershell
   python scripts/split_dataset.py --manifest data/manifests/bdslw131_dataset_manifest.csv --output-dir data/manifests/
   ```
3. **Production Model Training**:
   ```powershell
   python scripts/train.py --config configs/bdslw131/proposed.yaml --epochs 30
   python scripts/train.py --config configs/bdslw131/baseline.yaml --epochs 20
   ```
4. **Final Model Evaluation**:
   ```powershell
   python scripts/evaluate.py --config configs/bdslw131/proposed.yaml --checkpoint checkpoints/bdslw131/robust_holistic_fusion_best.pt --output results/bdslw131_proposed_evaluation.json
   python scripts/evaluate_robustness.py --checkpoint checkpoints/bdslw131/robust_holistic_fusion_best.pt --output results/bdslw131_robustness_report.json
   ```
