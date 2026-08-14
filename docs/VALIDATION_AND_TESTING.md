# Validation and testing strategy

## Automated tests

The suite covers canonical data loading, split and provenance rules, threshold selection, stable early detection, source-aware model-ladder utilities, causal prefix labels, tactic masks, head-tail tokenization, frozen human-test gates, ASR validation and metrics, entity redaction, risk feature contracts, inference policy, FastAPI request validation, and exported transformer output formatting.

Tests use small deterministic fixtures and mocked models for logic. Real-artifact smokes separately verify saved joblib reconstruction, local Whisper decoding, API OpenAPI generation, completed classical multi-task reconstruction, and optional exported-transformer reconstruction. The classical trainer also tests bounded transform batches, sparse-label probability alignment, unsupported labels, and artifact path/hash contracts. A passing unit suite is necessary but not evidence of model quality.

## Model evaluation gates

1. Split by conversation group; never split individual turns or windows across partitions.
2. Select thresholds only on the validation hard-negative view at false-positive rate at most 5%.
3. Report recall and stable scammer-turn detection latency at that fixed operating point.
4. Run three seeds and report mean plus standard deviation.
5. Run strict leave-one-mixed-source-out evaluation to expose dataset-origin shortcuts.
6. Treat test metrics as supporting-only after model family/checkpoint selection.
7. Require the independent frozen human-gold set before promotion.

The current calibrated candidate additionally enforces these mechanics:

- Feature families are compared with leave-one-mixed-source-out ROC-AUC inside
  the training partition only; neither validation nor test rows select features.
- Validation rows are divided into disjoint calibration and threshold views
  within each source/label stratum.
- A single threshold must satisfy the 5% FPR limit for every eligible negative
  validation source and every seed.
- External-evaluation records are schema-checked and forbidden from training,
  calibration, threshold selection, or model-family selection.

Reproduce the candidate and frozen external audit with:

```powershell
.venv\Scripts\python.exe scripts\train_mixed_source_candidate.py
.venv\Scripts\python.exe scripts\prepare_external_evaluation.py
.venv\Scripts\python.exe scripts\evaluate_external_text.py
```

## Pre-registered ablations

Ablations use the same conversation splits, seeds, training budget, hard-negative threshold rule, and supporting test policy as the full model. They are diagnostics, not alternative selection metrics. Report validation recall/FPR, macro-F1, stable scammer-turn latency, and mean plus standard deviation; do not select whichever ablation happens to have the best test result.

| Ablation ID | Change from full system | Question answered |
|---|---|---|
| `binary_head_only` | Set scam-type, tactic, and stage loss weights to zero | Does auxiliary supervision help binary detection? |
| `no_tactic_head` | Set tactic loss weight to zero | Does partial tactic supervision help or add label noise? |
| `no_stage_head` | Set stage loss weight to zero | Does causal stage supervision improve early detection? |
| `full_context_only` | Train only 100% conversation windows | Do causal 25%/50% prefix windows improve detection latency? |
| `right_truncation_control` | Replace head-tail context with right truncation at the same 256-token budget | Does preserving late payment evidence matter? |
| `base_score_only_fusion` | XGBoost receives only the out-of-fold base score | Do engineered fusion signals add value beyond calibration? |
| `no_entity_fusion` | Remove all entity-count features | Are extracted operational entities contributing? |
| `no_lexical_fusion` | Remove all deterministic tactic/stage signals | Are lexical rules driving apparent performance? |
| `reference_transcript` | Detect from manual transcript rather than ASR text | How much detection loss is attributable to ASR? |

The transformer ablations are GPU-comparison work after the completed laptop model and human-label improvement. Running nine additional three-seed transformer fits on this CPU laptop would consume disproportionate time and storage without strengthening the missing human-gold evidence. The exact variants remain fixed here before such compute is acquired.

## Required edge cases

- Empty and over-limit conversations are rejected.
- Hindi combining marks survive ASR metric normalization.
- Long conversations preserve both the beginning and ending evidence rather than right-truncating payment stages.
- Prefix windows never inherit future tactic or stage labels.
- Unknown tactic labels are downweighted rather than silently converted into full negatives.
- Normal bank/police/courier language is included as hard-negative content.
- URLs with punctuation, UPI IDs, emails, phone numbers, OTPs, and context-qualified account/Aadhaar numbers are disambiguated and redacted.
- Unsupported, oversized, or over-duration audio is rejected before transcription.
- Unpromoted model outputs never trigger the honeypot.

## Known external blockers

The required 150-conversation independently annotated human-gold set and a representative consented/licensed Hindi/Hinglish audio-validation set are not collected. Their schemas, collection protocols, leakage checks, and freeze tools are implemented, but their absence remains a non-negotiable limitation rather than a reason to fabricate data.
