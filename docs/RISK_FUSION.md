# XGBoost risk fusion and entity extraction

## Purpose

Risk fusion combines a frozen trained detector score with deterministic operational signals: conversation length, numeric density, privacy-redacted entities, lexical manipulation indicators, and a coarse lexical stage indicator. The fusion learner is XGBoost. It is a second trained ML layer, not an LLM and not the honeypot.

The lexical tactic features exist to make fusion executable before a validated multilingual transformer is promoted. They must not be described as the transformer tactic head or as human-quality psychological labels.

## Leakage control

The XGBoost training rows never receive in-sample predictions from the base detector. Five-fold stratified out-of-fold scoring clones and refits both the TF-IDF representation and base model inside each fold; each training row is transformed and scored by components that did not fit on that row. Validation and test use the frozen base detector. Split membership remains conversation-level and the test split is supporting-only during selection.

## Entity extraction

`src/arrestshield/entities.py` extracts URLs, UPI IDs, emails, Indian phone numbers, context-qualified account/Aadhaar/OTP candidates, monetary amounts, case references, claimed authorities, and payment apps. Extraction is local and deterministic. Sensitive values are redacted by default; explicit trusted callers must opt in to raw values. Numeric strings are not labeled as accounts unless nearby same-clause context supports that interpretation.

## Promotion rule

The fusion threshold is selected on the validation hard-negative view at false-positive rate at most 5%. Promotion additionally requires validation recall not below the frozen base detector and a frozen human-gold test set. Because the human set is not collected, `risk_fusion_v1` must remain `research_only_not_promoted` regardless of apparent corpus metrics.

The mixed-source score is followed by `scripts/evaluate_risk_fusion_loso.py`. For every mixed-label source, this audit excludes that source before refitting each TF-IDF fold, base detector, OOF score, and XGBoost model. A mean FPR below 5% is insufficient: every source/seed run must satisfy the 5% gate.

## Run

```powershell
.venv\Scripts\python.exe scripts\train_risk_fusion.py
```

The local XGBoost artifact is written below `artifacts/models/risk_fusion_v1` and is Git-ignored. Compact configuration, metrics, feature importance, hashes, and limitations are written below `reports/risk_fusion_v1`.
