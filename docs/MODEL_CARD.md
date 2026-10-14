# Model Card: SignFlow-BdSL Recognition Architectures

## Model Details
- **Model Name**: Robust Holistic Modality Fusion Classifier (`robust_holistic_fusion`) & Baseline Multi-Layer Perceptron (`baseline`)
- **Version**: 1.0.0
- **Primary Task**: Static Word-Level Bangla Sign Language Recognition (Task A) & Real-Time Webcam Debounced Classification (Task C)
- **Target Vocabulary**: 131 distinct BDSLW131 classes (`configs/bdslw131/vocabulary.json`)
- **Framework**: PyTorch 2.14+, NumPy, MediaPipe Holistic Landmarker
- **License**: Research & Academic Use

---

## Intended Use
- **Primary Use Cases**:
  - Offline classification of Bangla sign language words represented by holistic landmarks.
  - Interactive assistive communication via webcam stream using stability debouncing and bilingual Unicode text-to-speech output.
  - Benchmarking model robustness against controlled geometric, sensor, and occlusion perturbations.
- **Out-of-Scope / Prohibited Uses**:
  - Continuous sentence-level sign language translation (BDSLW131 provides isolated static word tokens, not continuous conversational video).
  - High-stakes life-safety or legal translation without human interpreter validation.
  - Production deployment in safety-critical domains.

---

## Architecture Specifications

### 1. Baseline Model (`BaselineClassifier`)
- **Input Dimension**: 1,659 (flattened 553 3D normalized coordinates)
- **Hidden Layers**: Fully-connected dense layers: `[1659 -> 512 -> 256 -> 131]`
- **Regularization**: Layer Normalization, ReLU activation, Dropout (`p=0.3`)
- **Parameter Count**: ~1.01M parameters

### 2. Proposed Model (`RobustHolisticFusion`)
- **Input Modalities**:
  - Left Hand: 21 landmarks x 3 = 63 dims
  - Right Hand: 21 landmarks x 3 = 63 dims
  - Upper Pose: 33 landmarks x 3 = 99 dims
  - Face Mesh: 478 landmarks x 3 = 1,434 dims
  - Total Raw Input Dimension: 1,659 dims
- **Modality Encoders**: Dedicated feedforward projection sub-networks per anatomical group.
- **Missingness Handling**: Modality validity vectors are evaluated; missing modalities (e.g. unobserved hand or occluded face) are masked before cross-modality attention aggregation.
- **Fusion Layer**: Concatenated multi-head attention fusion mechanism with LayerNorm and residual connection.
- **Classification Head**: Linear projection to 131 logits.

---

## Training Data & Procedure
- **Dataset**: BDSLW131 (29,930 images across 131 classes)
- **Splits**: Class-stratified 70% train (20,950 images), 15% validation (4,431 images), 15% test (4,549 images)
- **Loss Function**: Categorical Cross-Entropy (`nn.CrossEntropyLoss`) with optional label smoothing
- **Optimizer**: AdamW (`lr=1e-3`, weight decay `1e-4`)
- **Learning Rate Scheduler**: ReduceLROnPlateau / Cosine Annealing
- **Batch Size**: 64 (configurable in `configs/bdslw131/*.yaml`)
- **Hardware Acceleration**: CPU, CUDA, and Apple MPS supported.

---

## Evaluation Metrics
The models are evaluated against the held-out 4,549-sample test set across:
1. **Top-1 Accuracy**: Proportion of test instances where the highest logit matches ground truth.
2. **Top-5 Accuracy**: Proportion of test instances where ground truth is within the top 5 predictions.
3. **Macro-F1 & Weighted-F1**: Balanced assessment across all 131 classes regardless of sample count variation.
4. **Latency**: End-to-end inference time per observation (milliseconds).

---

## Limitations & Ethical Considerations
1. **Static Modality Boundary**: Because BDSLW131 consists of still photos, the trained weights represent spatial pose and hand configuration. Recognition of words differing *only* by temporal trajectory requires video-level training data.
2. **Signer Independence**: Metadata lacks signer IDs, precluding strict signer-split validation. Results demonstrate class-stratified held-out generalization, not signer-independent generalization.
3. **Lighting & Occlusion Sensitivity**: Performance degrades when hands or torso joints are clipped by the camera frame. The application incorporates a `HandQualityCoach` to guide users on optimal framing.
