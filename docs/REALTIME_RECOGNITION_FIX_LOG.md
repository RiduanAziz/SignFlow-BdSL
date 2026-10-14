# Real-Time Recognition Fix Log: SignFlow-BdSL

**Date:** 2026-10-14  
**Project:** SignFlow-BdSL — Robust Bangla Sign Language Word Recognition (BDSLW131)  
**Status:** **All Phases Completed & Verified (100% Pass Rate)**

---

## 1. Tracking Matrix of Findings, Fixes, and Verification

| Phase | Finding / Requirement | Affected Files | Status | Verification Result |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Eliminate sequence-length contradiction (`min_sequence_frames = 1` for static models) | `src/sign_language/api/main.py` | **Fixed** | Verified via `test_session_state_min_frames_for_static_classifier` & `test_single_frame_static_inference_reaches_prediction` |
| **Phase 1** | Remove inappropriate motion gate (`if not is_moving: continue`) for static pose models | `src/sign_language/api/main.py` | **Fixed** | Verified via `test_motion_gate_does_not_block_stationary_hand` |
| **Phase 1** | Session state machine coherence (clean reset, duplicate suppression, buffer alignment) | `src/sign_language/api/main.py` | **Fixed** | Verified via `test_websocket_live_connection_and_handshake` and reset tests |
| **Phase 2** | MediaPipe initialization verification (eager initialization during lifespan, fail-closed) | `src/sign_language/api/main.py`, `src/sign_language/landmarks/extractor.py` | **Fixed** | Verified via `/health` query & unit tests |
| **Phase 2** | Structured extraction error logging & diagnostics (no bare exception swallow) | `src/sign_language/api/main.py` | **Fixed** | Verified via `test_recoverable_bad_frame_does_not_break_websocket` |
| **Phase 2** | Detailed, honest `/health` endpoint distinguishing process, asset, detector, and model | `src/sign_language/api/main.py` | **Fixed** | Verified via `test_api_health` |
| **Phase 3** | Standalone detection diagnostic script (`scripts/diagnose_detection.py`) | `scripts/diagnose_detection.py` | **Implemented** | Verified on synthetic and real test images with `--no-gui` |
| **Phase 3** | Real bounding box computation for left and right hands (no hardcoded `None`) | `src/sign_language/api/main.py`, `src/sign_language/landmarks/schema.py` | **Fixed** | Verified via `test_bounding_box_computation` and WebSocket payloads |
| **Phase 4** | Monotonic `frame_id` and reliable `frame_processed` acknowledgement protocol | `src/sign_language/api/main.py`, `app/frontend/src/` | **Fixed** | Verified in WebSocket frames and frontend tests |
| **Phase 4** | Bounded frame pipeline (1 frame in-flight, stall watchdog, no arbitrary 120ms timeout) | `app/frontend/src/components/CameraPanel.tsx` | **Fixed** | Verified via Vitest suite (12 passed) |
| **Phase 5** | Safe MediaPipe lifecycle and resource management | `src/sign_language/landmarks/extractor.py` | **Fixed** | Verified clean instantiation and release |
| **Phase 6** | Finite value checks, mask propagation, no classification of zero/missing hands | `src/sign_language/api/main.py`, `src/sign_language/api/inference.py` | **Fixed** | Verified via `test_missing_model_reports_unavailable_honestly` |
| **Phase 7** | BDSLW131 Checkpoint and vocabulary verification (131 classes, 1659 dim) | `src/sign_language/api/inference.py` | **Fixed** | Verified via `test_bdsl_checkpoint_contract` |
| **Phase 8** | Word prediction stabilization, debounce, bilingual resolution | `src/sign_language/api/main.py` | **Fixed** | Verified via `test_single_frame_static_inference_reaches_prediction` |
| **Phase 9** | Frontend detection feedback: update `landmarksStatus` from `hand_landmarks` | `app/frontend/src/App.tsx`, `CameraPanel.tsx` | **Fixed** | Verified: L-Hand and R-Hand chips reflect active tracking |
| **Phase 9** | Hand overlay coordinate mapping and safe focus-view canvas drawing | `app/frontend/src/components/CameraPanel.tsx` | **Fixed** | Zero-dimension check guards `ctx.drawImage` against crashes |
| **Phase 10** | Detailed live diagnostics and performance metrics | `src/sign_language/api/main.py`, `scripts/diagnose_detection.py` | **Fixed** | Measured separate extraction and inference latencies |
| **Phase 11** | Comprehensive backend and frontend regression test suites | `tests/test_realtime_repair.py`, `app/frontend/src/App.test.tsx` | **Fixed** | 7/7 backend repair tests, 12/12 frontend vitest tests passed |
| **Phase 12** | Local benchmark execution and verification | Terminal execution | **Completed** | Full suite: 79 passed, 0 failed |
| **Phase 13** | Honest documentation and reproducibility instructions | `README.md`, `docs/PROJECT_STATUS.md` | **Completed** | Academic integrity preserved: static image model |

---

## 2. Details of Key Fixes

### 1. Elimination of Buffer Contradiction (`SessionState`)
* **File:** `src/sign_language/api/main.py`
* **Fix:** In `SessionState.__init__`, set:
  ```python
  if self.seq_len == 1:
      self.min_sequence_frames = 1
  else:
      self.min_sequence_frames = min(
          self.seq_len,
          max(1, min_valid_frames if min_valid_frames is not None else min(10, self.seq_len)),
      )
  ```
* **Impact:** Rolling buffer requires only 1 frame for static BDSLW131 word models, enabling live recognition to trigger without synthetic multi-frame padding.

### 2. Removal of Motion Gating for Static Poses
* **File:** `src/sign_language/api/main.py`
* **Fix:** Removed the `if not is_moving and s.current_candidate is None: continue` gate for static classification models.
* **Impact:** Static held hand gestures are classified immediately without requiring continuous hand jitter or artificial movement.

### 3. Dynamic Hand Bounding Box Calculation
* **Files:** `src/sign_language/landmarks/schema.py`, `src/sign_language/api/main.py`
* **Fix:** Implemented `compute_landmark_bounding_box(coords, mask, padding)` in `schema.py`. Integrated into `main.py` to extract `left_box` and `right_box` from valid landmarks.
* **Impact:** The WebSocket now emits actual normalized bounding boxes (`xmin, ymin, xmax, ymax, width, height`), enabling the frontend Hand Focus / Zoom viewport to track hands dynamically.

### 4. Frame Acknowledgement Protocol & In-Flight Lock
* **Files:** `src/sign_language/api/main.py`, `app/frontend/src/types/index.ts`, `app/frontend/src/components/CameraPanel.tsx`, `app/frontend/src/App.tsx`
* **Fix:** 
  - Monotonic `frame_id` attached to every transmitted frame.
  - Backend responds with matching `frame_id` on `hand_landmarks`, `live_prediction`, and final `frame_processed`.
  - Frontend releases in-flight lock strictly upon receiving `lastProcessedFrameId >= inFlightFrameIdRef.current` with a 2000ms watchdog for stall recovery.
* **Impact:** Prevents request congestion, out-of-order execution, and queue bloat without dropping valid frames prematurely.

### 5. Frontend Detection Status Synchronization
* **Files:** `app/frontend/src/App.tsx`, `app/frontend/src/components/CameraPanel.tsx`
* **Fix:**
  - `App.tsx` updates `landmarksStatus` directly inside `case 'hand_landmarks':`.
  - `CameraPanel.tsx` chips check `Boolean(landmarksStatus?.left_hand || leftLandmarks.some(p => p.valid))`.
* **Impact:** Solves the persistent "No hand detected" UI bug. Tracking chips immediately display `Tracking` when hands enter the camera frame.

### 6. Defensively Guarded Focus Canvas Drawing
* **File:** `app/frontend/src/components/CameraPanel.tsx`
* **Fix:** Added `if (sw < 5 || sh < 5 || sb.width <= 0.01 || sb.height <= 0.01) return;` before `ctx.drawImage` in the Hand Focus close-up effect.
* **Impact:** Eliminates browser `IndexSizeError` and division-by-zero crashes when hands temporarily exit or bounding boxes are degenerate.

### 7. Standalone Diagnostic Utility
* **File:** `scripts/diagnose_detection.py`
* **Capabilities:** Supports image, video, and webcam inputs (`--source 0`). Prints landmark counts, detection flags, extraction latency in milliseconds, FPS, and bounding boxes. Verified with real MediaPipe HolisticLandmarker asset.
