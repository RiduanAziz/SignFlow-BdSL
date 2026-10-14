# Thesis Research Experiment Summary

## Training Summary
- Epochs Trained: 10
- Final Train Loss: 1.6061
- Best Validation Macro-F1: 0.2159

## Held-Out Test Evaluation
- Total Test Samples: 20
- Top-1 Accuracy: 0.2000
- Macro-F1: 0.0556
- Weighted-F1: 0.0667

## Robustness Evaluation (Controlled Disturbances)
- Clean Accuracy: 0.2000
| Disturbance Type | Severity | Noisy Accuracy | Accuracy Degradation |
|---|---|---|---|
| spatial_coordinate_jitter | 0.0 | 0.2000 | 0.0000 |
| spatial_coordinate_jitter | 0.1 | 0.2000 | 0.0000 |
| spatial_coordinate_jitter | 0.3 | 0.2000 | 0.0000 |
| spatial_coordinate_jitter | 0.5 | 0.2000 | 0.0000 |
| spatial_landmark_dropout | 0.0 | 0.2000 | 0.0000 |
| spatial_landmark_dropout | 0.1 | 0.2000 | 0.0000 |
| spatial_landmark_dropout | 0.3 | 0.2000 | 0.0000 |
| spatial_landmark_dropout | 0.5 | 0.2000 | 0.0000 |
| spatial_scale | 0.0 | 0.2000 | 0.0000 |
| spatial_scale | 0.1 | 0.2000 | 0.0000 |
| spatial_scale | 0.3 | 0.2000 | 0.0000 |
| spatial_scale | 0.5 | 0.2000 | 0.0000 |
| spatial_translation | 0.0 | 0.2000 | 0.0000 |
| spatial_translation | 0.1 | 0.2000 | 0.0000 |
| spatial_translation | 0.3 | 0.2000 | 0.0000 |
| spatial_translation | 0.5 | 0.2000 | 0.0000 |
| temporal_frame_drop | 0.0 | 0.2000 | 0.0000 |
| temporal_frame_drop | 0.1 | 0.2000 | 0.0000 |
| temporal_frame_drop | 0.3 | 0.2000 | 0.0000 |
| temporal_frame_drop | 0.5 | 0.2000 | 0.0000 |
| temporal_frame_duplicate | 0.0 | 0.2000 | 0.0000 |
| temporal_frame_duplicate | 0.1 | 0.2000 | 0.0000 |
| temporal_frame_duplicate | 0.3 | 0.2000 | 0.0000 |
| temporal_frame_duplicate | 0.5 | 0.2000 | 0.0000 |
| temporal_sequence_truncate | 0.0 | 0.2000 | 0.0000 |
| temporal_sequence_truncate | 0.1 | 0.2000 | 0.0000 |
| temporal_sequence_truncate | 0.3 | 0.2000 | 0.0000 |
| temporal_sequence_truncate | 0.5 | 0.2000 | 0.0000 |