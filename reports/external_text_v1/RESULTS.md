# Frozen external text evaluation

The 243 CC0 transcripts are held outside canonical training, calibration,
threshold selection, and model-family selection. They contain English scammer
speech from public scambaiting videos and are positive-only, so they measure
recall but cannot measure FPR or ROC-AUC.

| Frozen detector | Detected | Missed | Recall |
|---|---:|---:|---:|
| Previous `model_ladder_v1` SGD | 3 | 240 | 1.23% |
| Calibrated mixed-source candidate v2 | 15 | 228 | **6.17%** |

The candidate improves relative recall by 5×, but the absolute result blocks
promotion. The dataset is not copied into the canonical corpus and no threshold
was changed after seeing it.
