# Experiment Protocol: SignFlow-BdSL Benchmarking

This document specifies the rigorous benchmarking protocol, evaluation standards, and data separation rules for comparative experiments in **SignFlow-BdSL**.

---

## 1. Experimental Design & Principles

### A. Non-Contamination Rule
To prevent data leakage:
- Model selection, hyperparameter tuning, and early stopping must be guided strictly by the **Validation Split** (`data/manifests/bdslw131_val_manifest.json`).
- The **Test Split** (`data/manifests/bdslw131_test_manifest.json`) is evaluated only once per final model checkpoint.
- Train, validation, and test splits share **zero samples** (`0.0% overlap`).
- All comparative models (Baseline vs. Proposed) must be evaluated on **identically identical test manifests and random seeds**.

### B. Single Source of Truth for Vocabulary
- Every experiment must reference `configs/bdslw131/vocabulary.json`.
- Checkpoints must store `class_names` and `num_classes` corresponding to the canonical 131-class mapping.
- Experiments must never map old ASL labels or arbitrary indices to BDSLW131 classes.

---

## 2. Benchmark Models

| Model ID | Architecture | Config Path | Checkpoint Path | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `baseline` | Deep MLP (LayerNorm + Dropout) | `configs/bdslw131/baseline.yaml` | `checkpoints/bdslw131/baseline_best.pt` | Reference standard for feature expressiveness |
| `proposed` | Robust Holistic Modality Fusion | `configs/bdslw131/proposed.yaml` | `checkpoints/bdslw131/robust_holistic_fusion_best.pt` | Modality-aware cross-attention fusion with masking |

---

## 3. Metrics Calculation

Metrics are generated programmatically via `scripts/evaluate.py` from stored logits and ground truth labels:

1. **Top-1 Accuracy**:
   $$\text{Acc}_1 = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$

2. **Top-k Accuracy** ($k=5$):
   $$\text{Acc}_k = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \in \text{top}_k(\hat{\mathbf{p}}_i))$$

3. **Macro-Averaged F1-Score**:
   $$\text{Macro-F1} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
   Ensures classes with slightly fewer samples (180) are weighted equally to larger classes (254).

4. **Weighted F1-Score**:
   $$\text{Weighted-F1} = \sum_{c=1}^C \frac{N_c}{N} \cdot \text{F1}_c$$

5. **Inference Latency**:
   - Evaluated over 100 consecutive forward passes on CPU and GPU.
   - Measures mean latency (ms), p95 latency (ms), and throughput (samples/sec).

---

## 4. Robustness Stress-Testing Protocol

Robustness is assessed via `scripts/evaluate_robustness.py` using physical transformations:

1. **Coordinate Jitter (Spatial Noise)**:
   - Gaussian noise: $\epsilon \sim \mathcal{N}(0, \sigma^2)$ where $\sigma = 0.05 \times \text{severity}$.
   - Severities tested: `[0.0, 0.1, 0.2, 0.3, 0.5]`.
   - Zero-severity tolerance: must match clean accuracy within $\pm 0.0001$.

2. **Camera Translation**:
   - Uniform translation applied consistently across anatomical joints: $\mathbf{x}' = \mathbf{x} + \delta$.

3. **Camera Scale / Zoom**:
   - Centered zoom scaling: $\mathbf{x}' = \mathbf{x}_c + s \cdot (\mathbf{x} - \mathbf{x}_c)$.

4. **Landmark Dropout (Occlusion)**:
   - Structured masking of hand or face clusters to simulate partial view obstructions.
   - Evaluated across severities `[0.1, 0.2, 0.3, 0.5]`.

---

## 5. Artifact & Report Generation

Every evaluation run exports traceability files to `results/`:
- `results/bdslw131_<model>_evaluation.json`: Complete metrics dictionary, sample counts, and checkpoint hashes.
- `results/bdslw131_<model>_per_class_metrics.csv`: Per-class precision, recall, and F1.
- `results/bdslw131_<model>_confusion_matrix.csv`: Full $131 \times 131$ confusion matrix.
- `results/bdslw131_robustness_report.json`: Corruption benchmarks across severities.
