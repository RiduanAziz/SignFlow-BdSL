# Reference Repository Review

This document records the architectural, algorithmic, licensing, and implementation findings from inspecting reference repositories and standards for **Robust Real-Time Dynamic Word-Level American Sign Language Recognition**.

---

## Reference A: Official WLASL
- **Repository URL**: [https://github.com/dxli94/WLASL](https://github.com/dxli94/WLASL)
- **Inspected Files**:
  - `code/TGCN/data_reader.py`: Data ingestion, gloss-to-instance mapping, missing video handling.
  - `code/I3D/train_i3d.py` & `code/I3D/pytorch_i3d.py`: 3D CNN RGB video baseline.
  - `code/TGCN/models.py`: Pose-TGCN (Temporal Graph Convolutional Network on 2D OpenPose keypoints).
  - `data/WLASL_v0.3.json`: Primary dataset metadata (glosses, instances, bounding boxes, URLs, signers, splits).
  - `data/nslt_100.json`, `nslt_300.json`, `nslt_1000.json`, `nslt_2000.json`: Official split indices.
  - `C-UDA-1.0.pdf`: Computational Use of Data Agreement.
- **Research / Task Match**:
  - Direct task match: WLASL is the benchmark dataset used in this thesis.
  - Relevant for dataset schema, gloss index conventions, split evaluation protocol (Top-1, Top-5, Top-10), and handling broken/unavailable YouTube links.
- **Architecture / Engineering Pattern**:
  - WLASL uses OpenPose 2D landmarks (body pose + hands, 27 joints total for Pose-TGCN) or dense RGB video frames for I3D.
  - Evaluates across subsets: WLASL-100, WLASL-300, WLASL-1000, WLASL-2000.
  - Evaluates standard train/val/test splits, though the original splits contain signer overlap across partitions.
- **License & Restrictions**:
  - C-UDA-1.0 (Computational Use of Data Agreement).
  - Raw videos and extracted features must remain local and cannot be redistributed. Third-party checkpoints must not be distributed without authorization.
  - Dataset citation required: Li et al., "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison", WACV 2020.
- **Adopted Approach**:
  - Deterministic gloss-to-ID mapping derived from sorted glosses.
  - Exact sample matching using `video_id`.
  - Machine-readable tracking of missing/unreadable videos (9,103 missing locally out of 21,083 WLASL instances).
  - Explicit recognition metrics (Top-1, Macro-F1, Confusion Matrix).
- **Excluded Approach**:
  - End-to-end RGB I3D baseline: requires multi-GPU clusters and massive video storage unsuitable for real-time edge landmark processing.
  - Reusing original splits as the sole evaluation: original splits permit signer leakage; our primary benchmark mandates a strictly signer-independent split.
- **Validation Method**:
  - Ingestion tests verifying gloss-to-ID mapping and missing video isolation without assigning false class 0.

---

## Reference B: SignBart
- **Repository URL**: [https://github.com/TinhNguyen2312/SignBart](https://github.com/TinhNguyen2312/SignBart)
- **Inspected Files**:
  - `models/sign_bart.py`: Skeleton sequence encoder and Bart-based representation.
  - `datasets/preprocess.py`: Landmark coordinate normalization and component separation.
  - `configs/default.yaml`: Modality configurations and ablation parameters.
- **Research / Task Match**:
  - Highly relevant for isolated sign landmark representation: demonstrates that component-aware coordinate normalization (hands relative to wrist, body relative to shoulders) vastly outperforms unstructured vector flattening.
- **Architecture / Engineering Pattern**:
  - Structured 3D coordinate tensor `[T, J, 3]` with explicit landmark masks.
  - Modality separation: Left Hand (21), Right Hand (21), Pose (upper body joints), Face (lips/contours).
  - Normalization: anatomical centering per component; scale normalization by inter-shoulder distance or hand bounding box diagonal.
- **License & Restrictions**:
  - MIT License.
- **Adopted Approach**:
  - Geometric coordinate normalization operating on 3D coordinates `(N, 3)` rather than flattened feature vectors.
  - Component-specific spatial projection encoders before temporal fusion.
  - Independent modality masking so missing hands or face do not pollute feature means.
- **Excluded Approach**:
  - Full BART language model backbone: excessive compute overhead for isolated word classification; our research investigates real-time spatial-temporal fusion architectures.
- **Validation Method**:
  - Unit tests verifying translation and scale invariance on 3D geometric coordinates with partial modality dropouts.

---

## Reference C: OpenHands
- **Repository URL**: [https://github.com/AI4Bharat/openhands](https://github.com/AI4Bharat/openhands)
- **Inspected Files**:
  - `openhands/models/pose/`: Pose-based recognizers (SL-GCN, ST-GCN, HRNet).
  - `openhands/apis/inference.py`: Pose extraction and inference pipeline.
- **Research / Task Match**:
  - Demonstrates spatial-temporal graph convolutions (ST-GCN / SL-GCN) on 2D/3D pose landmarks for sign language recognition.
- **Architecture / Engineering Pattern**:
  - Graph convolutional network using physical body/hand adjacency matrix.
  - Fixed landmark schemas and PyTorch Lightning-based experiment management.
- **License & Restrictions**:
  - MIT License.
- **Adopted Approach**:
  - Modality graph grouping concepts (pose vs. left hand vs. right hand) and lightweight temporal pooling mechanisms.
- **Excluded Approach**:
  - Wholesale external dependency on OpenHands package: introduces heavyweight conflicting dependency trees and rigid dataset format expectations.
- **Validation Method**:
  - Native PyTorch baseline implementations (MLP, LSTM, GRU, Temporal Transformer) sharing identical tensor schemas and collators.

---

## Reference D: Real-Time Landmark-Based LSTM Implementation
- **Repository URL**: [https://github.com/bNhN-0/asl-recognition-lstm](https://github.com/bNhN-0/asl-recognition-lstm)
- **Inspected Files**:
  - `train.py`: Sequence windowing and LSTM training.
  - `app.py`: OpenCV webcam loop and rolling buffer.
- **Research / Task Match**:
  - Real-time webcam inference loop for word/gesture recognition using rolling temporal windows.
- **Architecture / Engineering Pattern**:
  - Rolling sequence buffer `deque(maxlen=T)` for webcam frames.
  - Prediction smoothing and confidence thresholding.
  - Label dictionary and normalization statistics saved alongside model weights.
- **License & Restrictions**:
  - MIT License.
- **Adopted Approach**:
  - Requirement that training, offline evaluation, and webcam inference share identical versioned preprocessing pipelines, label encodings, and normalization statistics.
  - Real-time rolling buffer with confidence threshold gating.
- **Excluded Approach**:
  - Replicating their specific dataset or presenting their numbers as WLASL baselines (their model is trained on a small private gesture set).
- **Validation Method**:
  - End-to-end test verifying that feeding an identical sequence to offline evaluation and `RealTimePredictor` yields bitwise identical logits.

---

## Reference E: Runtime and Deployment Design (Sign-Language-Translator)
- **Repository URL**: [https://github.com/Abdul-Nafee/Sign-Language-Translator](https://github.com/Abdul-Nafee/Sign-Language-Translator)
- **Inspected Files**:
  - `backend/app.py`: Backend inference server.
  - `frontend/`: UI display, prediction debouncing, and latency measuring.
- **Research / Task Match**:
  - Production-ready separation of frame capture, landmark extraction, and model inference.
- **Architecture / Engineering Pattern**:
  - Decoupled inference service loading model once at startup.
  - Prediction smoothing / debouncing to eliminate frame-to-frame flicker.
- **License & Restrictions**:
  - MIT License.
- **Adopted Approach**:
  - Single predictor initialization on FastAPI startup (`lifespan` handler).
  - Debounce/smoother utility with configurable window.
- **Excluded Approach**:
  - Full sentence generation / text-to-speech (out of scope for dynamic word-level classification).
- **Validation Method**:
  - FastAPI tests validating startup model residency, health status, and batch prediction latency.

---

## Reference F: Official MediaPipe Tasks API
- **Repository / Docs URL**: [https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker/python](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker/python)
- **Inspected Files / APIs**:
  - `mediapipe.tasks.python.vision.HolisticLandmarker`
  - `HolisticLandmarkerOptions`, `RunningMode.VIDEO`, `RunningMode.IMAGE`
  - Model asset: `holistic_landmarker.task` (13.6 MB, located in `models/mediapipe/`)
- **Research / Task Match**:
  - Canonical holistic feature extractor producing 553 landmarks:
    - Left Hand: 21 landmarks
    - Right Hand: 21 landmarks
    - Pose: 33 landmarks
    - Face: 478 landmarks
- **Architecture / Engineering Pattern**:
  - Video processing mode requires monotonically increasing timestamps in milliseconds (`detect_for_video(mp_image, timestamp_ms)`).
  - Frames must be in SRGB format: OpenCV BGR frames require explicit `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.
  - Reusing an initialized landmarker instance per worker thread to avoid heavy re-initialization.
  - Explicit resource release via `.close()`.
- **License & Restrictions**:
  - Apache 2.0 License.
- **Adopted Approach**:
  - Proper color space conversion (BGR -> RGB).
  - Video mode with monotonic timestamping.
  - Worker-level landmarker reuse with graceful resource release.
  - Zero-placeholder for missing landmarks accompanied by an explicit boolean validity mask.
- **Excluded Approach**:
  - Deprecated legacy `mp.solutions.holistic` pipeline.
  - Silently substituting zeros when the model asset is absent.
- **Validation Method**:
  - Unit tests verifying BGR-to-RGB conversion, timestamp monotonicity, and landmarker resource lifecycle.
