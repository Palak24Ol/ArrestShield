# Classical Baseline Results

## Status

The classical detector ladder is trained, but neither SGD nor XGBoost is accepted as a robust detector. The apparently near-perfect ordinary split scores are invalid as a production claim because the model can exploit source and writing-style shortcuts.

## Pre-registered operating rule

- Choose thresholds on validation data only.
- Require false-positive rate (FPR) at or below 5% on hard negatives.
- At that operating point, maximize recall and report stable turns-to-detection.
- Repeat with seeds 17, 42, and 93.
- Evaluate the test split only after family and threshold selection.

## Ordinary split result (supporting only)

At deployment seed 42, the ordinary conversation split produced:

| Family | Validation recall | Validation FPR | Test recall | Test FPR |
|---|---:|---:|---:|---:|
| SGD | 1.0000 | 0.0000 | 0.9812 | 0.0006 |
| XGBoost | 1.0000 | 0.0016 | 0.9969 | 0.0006 |

These values are not used as evidence of real-world robustness.

## Strict unseen-source audit (primary shortcut check)

The representation and classifier were refit after excluding each mixed-label source channel in turn. Results are source-macro mean ± standard deviation across the three seeds:

| Family | Recall | FPR | F1 | Meets FPR ≤ 5%? |
|---|---:|---:|---:|---:|
| SGD | 0.5315 ± 0.1610 | 0.1115 ± 0.0425 | 0.6670 ± 0.1089 | No |
| XGBoost | 0.5293 ± 0.0041 | 0.0702 ± 0.0049 | 0.6377 ± 0.0045 | No |

XGBoost is more stable and has lower FPR than SGD under this audit, but its 7.0% mean FPR still violates the 5% gate. Therefore, neither family is promoted as the final scam detector. XGBoost remains useful as a downstream risk-fusion candidate after leakage-safe component scores are available.

## Interpretation

The current positive set is predominantly synthetic or source-silver and differs from live scammer speech. The audit demonstrates why high random-split accuracy must not be reported as real-world performance. The next detector stage uses a multilingual pretrained encoder and retains the same source-aware evaluation controls. The LLM honeypot is not used in any detector score or label decision.

Machine-readable evidence is stored in `reports/model_ladder_v1/metrics.json` and `reports/model_ladder_v1/loso_metrics.json`.
