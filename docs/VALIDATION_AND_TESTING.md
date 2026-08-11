# Validation and testing strategy

## Automated tests

The suite covers canonical data loading, split and provenance rules, threshold selection, stable early detection, source-aware model-ladder utilities, causal prefix labels, tactic masks, head-tail tokenization, frozen human-test gates, ASR validation and metrics, entity redaction, risk feature contracts, inference policy, FastAPI request validation, and exported transformer output formatting.

Tests use small deterministic fixtures and mocked models for logic. Real-artifact smokes separately verify saved joblib reconstruction, local Whisper decoding, API OpenAPI generation, and—after training—exported transformer reconstruction. A passing unit suite is necessary but not evidence of model quality.

## Model evaluation gates

1. Split by conversation group; never split individual turns or windows across partitions.
2. Select thresholds only on the validation hard-negative view at false-positive rate at most 5%.
3. Report recall and stable scammer-turn detection latency at that fixed operating point.
4. Run three seeds and report mean plus standard deviation.
5. Run strict leave-one-mixed-source-out evaluation to expose dataset-origin shortcuts.
6. Treat test metrics as supporting-only after model family/checkpoint selection.
7. Require the independent frozen human-gold set before promotion.

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
