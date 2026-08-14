# Simple transcript-content detector v1

## Purpose

This is the binary model used by `major_project/`: transcript in, `SCAM` or
`NOT_SCAM` out. The LLM is not used during detection.

## Training pool

- 36,626 canonical training conversations from all cleaned relevant sources.
- 161 source-URL-grouped public scam-call transcripts for adaptation training.
- 30 project-authored mixed-label hard examples distinguishing scam demands
  from legitimate safety warnings.
- Word 1-2 gram and character 3-5 gram TF-IDF features.
- SGD logistic classifier across seeds 17, 42, and 93; seed 42 packaged.

## Results

| Evaluation | Result |
|---|---:|
| Fixed professor-demo behavioral suite | **18/20 (90%)** |
| Behavioral scam recall | **10/10 (100%)** |
| Behavioral legitimate specificity | **8/10 (80%)** |
| Public scam-call adaptation holdout, seed 42 | 49/50 (98%) recall |
| Canonical test, seed 42 | 98.79% accuracy |
| Canonical test, seed 42 | 100% recall, 1.26% FPR |

The 90% behavioral result is the clearest demo number. The public adaptation
holdout shares corpus characteristics with training, and the canonical test is
dominated by easier legitimate sources. Neither is presented as real-world
accuracy.

## Observed errors

The behavioral suite preserves two legitimate false alarms: a fraud-awareness
seminar statement and a KYC safety warning. No behavioral scam example was
missed. These errors show that legitimate warnings may repeat the same vocabulary
as scammers.

## Decision boundary

Only the saved trained model and fixed threshold create the binary label.
Whisper only transcribes audio. A positive label makes the separate LLM
honeypot eligible; a negative label blocks it.
