# Project Audit and Reconnaissance Report

**Date**: 2026-10-09  
**Repository**: `RiduanAziz/Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition`  
**Working Workspace**: `F:\Robust-Real-Time-Dynamic-Word-Level-American-Sign-Language-Recognition`  
**Git Branch**: `main` (clean working tree)  

---

## 1. Executive Summary

A comprehensive executable audit was conducted across the codebase, datasets, models, configurations, CI/Docker setups, and tests.
Key findings:
1. **Local Dataset**: 11,980 valid local `.npz` landmark files exist in `data/landmarks/`, corresponding to 11,980 instances from the WLASL dataset. Raw MP4 files are not stored locally, but all 11,980 landmark files are present, representing 2,000 WLASL glosses and 78 signers.
2. **Existing Defects**: Multiple critical bugs, broken imports, and architectural limitations were confirmed across `scripts/`, `src/sign_language/`, and configuration files.
3. **Execution Safety**: All raw dataset artifacts and checkpoints will remain strictly local. No dataset files or private model weights will be committed or tracked.

---

## 2. Detailed Audit of Known Issues

| ID | Issue Description | Confirmed File / Location | Actual Severity | Root Cause & Resolution Plan |
|---|---|---|---|---|
| **ISSUE-01** | `scripts/evaluate_robustness.py` imports `collate_fn` from `training.trainer`, which does not exist. | `scripts/evaluate_robustness.py:13` | 🔴 Blocker | `collate_sequences` was defined in `data.dataloader` but never exported as `collate_fn` in `training.trainer`. Export `collate_fn = collate_sequences` and unify data loading. |
| **ISSUE-02** | Robustness evaluator initializes `LandmarkPipeline` with model input dim (4977) instead of raw landmark dim (1659). | `scripts/evaluate_robustness.py:25` | 🔴 Blocker | `config.model.input_dim` is 4977 (including velocity and acceleration). Raw holistic landmark dimension is 1659. Passing 4977 fails dimension checks. Fix: initialize pipeline with raw feature dimension (1659). |
| **ISSUE-03** | `scripts/evaluate.py` evaluates random tensors against fabricated labels `[0, 1]` rather than loading a checkpoint and test dataset. | `scripts/evaluate.py:18-21` | 🔴 Blocker | Fake smoke evaluation script. Refactor `evaluate.py` to load model weights, checkpoint metadata, test manifest, compute Top-1 accuracy, Macro-F1, confusion matrix, and export full reports. |
| **ISSUE-04** | `RealTimePredictor` falls back to an untrained model when the checkpoint path is missing or invalid. | `src/sign_language/api/inference.py:37` | 🔴 Safety Blocker | `RealTimePredictor` logged a warning and continued with randomly initialized weights. Fix: Raise `FileNotFoundError` or `ValueError` immediately upon checkpoint load failure. |
| **ISSUE-05** | FastAPI creates `RealTimePredictor` inside `/predict` request handler instead of on application startup. | `src/sign_language/api/main.py:30` | 🟠 Performance / Architecture | Model was instantiated on every incoming HTTP request. Fix: Use FastAPI `lifespan` handler to load and cache the predictor once at application startup. |
| **ISSUE-06** | Inconsistent checkpoint paths across scripts (`models/temporal_transformer_trained.pt` vs `models/{model_name}.pt`). | `scripts/realtime.py:15`, `scripts/train.py:115` | 🟡 Usability / Inconsistency | Standardize checkpoint naming: save rich checkpoints to `models/{model_name}_best.pt` with class mapping, config, and preprocessing artifacts. |
| **ISSUE-07** | OpenCV BGR frames passed directly into MediaPipe `mp.Image(image_format=SRGB)` without color conversion. | `src/sign_language/landmarks/extractor.py:92` | 🔴 Accuracy Degrader | OpenCV reads BGR; MediaPipe expects RGB. Landmark accuracy suffers without BGR-to-RGB conversion. Fix: Explicitly convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. |
| **ISSUE-08** | Flattened landmarks normalized as 1D vectors instead of 3D landmark geometry. | `src/sign_language/landmarks/normalization.py:52-55` | 🔴 Algorithmic Flaw | When landmarks are 2D arrays `[T, 1659]`, normalization subtracted mean across all features. Fix: Reshape to `[T, N, 3]` and normalize geometric coordinates relative to anatomical anchor points. |
| **ISSUE-09** | Visibility masks saved in `.npz` but ignored during training and feature computation. | `src/sign_language/landmarks/pipeline.py`, `scripts/train.py` | 🔴 Data Distortion | Missing landmarks (0, 0, 0) were treated as real points. Fix: Pass point and sequence masks to models and pooling layers. |
| **ISSUE-10** | Zero padding before derivative computation creates massive boundary spikes; naive truncation drops frames. | `src/sign_language/landmarks/pipeline.py:63-73` | 🔴 Data Distortion | Padding zeros caused `v = 0 - last_frame`. Fix: Resample uniformly across valid sequence frames before temporal derivative calculation, or mask boundary derivatives. |
| **ISSUE-11** | `LandmarkPipeline.process(frame)` repeats a single frame across the entire sequence. | `src/sign_language/landmarks/pipeline.py:27` | 🔴 Flawed Logic | Frame repetition simulates artificial static gestures instead of dynamic sequences. Deprecate and replace with rolling sequence buffer. |
| **ISSUE-12** | `scripts/extract_landmarks.py` instantiated an extractor per video and defaulted to `cpu_count()`. | `scripts/extract_landmarks.py:31, 51` | 🟠 Performance / Resource | Creates thread contention and high initialization overhead. Fix: Reuse extractor within worker processes, default to conservative worker count (e.g. 2). |
| **ISSUE-13** | WLASL JSON ingestion fell back to `class_id = 0` for all records because instance `class_id` was absent. | `src/sign_language/data/ingestion.py:77, 97` | 🔴 Critical Bug | In WLASL metadata, instances do not have `class_id`; gloss name is at top level. Fallback gave all records class 0. Fix: Map gloss name deterministically to sorted class index. |
| **ISSUE-14** | `configs/experiments/proposed.yaml` declared `model.name: lstm` instead of a distinct proposed model. | `configs/experiments/proposed.yaml:2` | 🔴 Research Integrity | The proposed method was an ordinary LSTM. Fix: Implement and configure `RobustHolisticFusionClassifier` with modality encoders, spatial projection, mask-aware pooling, and fusion. |
| **ISSUE-15** | Model factory has no route to `fusion` or `robust_holistic_fusion`. | `src/sign_language/models/factory.py:16-40` | 🔴 Blocker | Factory only supported `mlp`, `lstm`, `gru`, and `transformer`. Add `robust_holistic_fusion` and `fusion`. |
| **ISSUE-16** | Docker Compose referenced non-existent `./app/frontend` volume. | `docker-compose.yml:18` | 🟡 Broken Service | Directory `./app/frontend` does not exist. Fix: Clean up compose config to focus on backend API service and documentation. |
| **ISSUE-17** | `pyproject.toml` omitted `httpx` from dependencies, causing CI failures on test client collection. | `pyproject.toml:11-33` | 🔴 CI Blocker | Add `httpx` to dependencies and `src` to `tool.pytest.ini_options.pythonpath`. |

---

## 3. Local Dataset & Hardware Assessment

- **Landmark files detected**: 11,980 `.npz` files in `data/landmarks/`.
- **Landmark array shape**: `landmarks`: `(T, 1659)` float32, `mask`: `(T, 1659)` float32.
- **Landmark breakdown**:
  - Left Hand: 21 landmarks × 3 = 63 values
  - Right Hand: 21 landmarks × 3 = 63 values
  - Pose: 33 landmarks × 3 = 99 values
  - Face: 478 landmarks × 3 = 1434 values
  - Total: 553 landmarks × 3 = 1659 values per frame.
- **Derived temporal features**: Position (1659) + Velocity (1659) + Acceleration (1659) = 4977 features per frame.
- **Environment**: Python 3.14.7 virtual environment with PyTorch 2.14.1 (CPU), MediaPipe 1.0.1, OpenCV 5.0.0, NumPy 2.5.3, Scikit-Learn 1.9.1, FastAPI, Pytest.
- **MediaPipe Task Asset**: Present at `models/mediapipe/holistic_landmarker.task` (13.6 MB).
