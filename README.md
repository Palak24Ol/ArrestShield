# ArrestShield ML

ArrestShield is a multilingual research pipeline for detecting digital-arrest and related phone scams from English, Hindi, and Hinglish conversations. It includes versioned dataset ingestion, conversation-level splitting, classical and transformer detectors, local Whisper transcription, threat-entity extraction, XGBoost risk fusion, evaluation gates, and a local FastAPI inference service.

The scam decision is made only by trained ML models and frozen policy thresholds. The LLM-powered fake-victim honeypot is a separate downstream system. It is never used to label training data, calculate risk, choose a threshold, or decide whether a conversation is a scam.

## Current status

This repository is an implemented research prototype, not a production detector.

| Component | State | Evidence |
|---|---|---|
| Canonical dataset | Built and validated: 52,206 conversations, 615,084 turns, zero split-group leakage | `docs/datasets/PROCESSED_DATA_VALIDATION.json` |
| SGD and SVD-XGBoost ladder | Trained across seeds 17, 42, and 93 | `reports/model_ladder_v1` |
| Strict source audit | Failed the pre-registered 5% FPR gate: SGD 11.15%, XGBoost 7.02% source-macro FPR | `reports/model_ladder_v1/loso_metrics.json` |
| CPU multi-task detector | SGD binary plus XGBoost scam-type, stage, and 9 supported tactic heads; three seeds trained | `reports/classical_multitask_v1` |
| Optional transformer comparison | Causal-prefix/head-tail trainer and exact resume implemented; full DistilBERT CPU run was stopped as impractical | `docs/MULTITASK_TRANSFORMER.md` |
| XGBoost risk fusion | Trained across three seeds using out-of-fold base scores; strict source audit fails at 27.71% macro FPR | `reports/risk_fusion_v1` |
| Whisper ASR | Local multilingual Whisper-tiny works end to end; Hindi/Hinglish backend selection remains gated | `reports/asr_smoke_v1` |
| Entity extraction | Local deterministic extraction with sensitive-value redaction by default | `src/arrestshield/entities.py` |
| Inference API | Text/audio routes, research status, redaction, and honeypot boundary implemented | `docs/INFERENCE_API.md` |
| Mixed-source detector v1 | Historical source-shortcut correction: held-out Hinglish ROC-AUC 0.550 to 0.756 across three seeds | `reports/mixed_source_v1` |
| Calibrated mixed-source candidate | Character TF-IDF + SGD + Platt; held-out Hinglish ROC-AUC 0.815 ± 0.023, one threshold across seeds | `reports/mixed_source_candidate_v2` |
| Frozen external scam-call check | Recall improves from 3/243 to 15/243, but 93.8% of unseen English scam calls remain undetected | `reports/external_text_v1` |
| Historical unseen-English audit | Old detector: 18.51% Banking77 FPR and 5.06% aggregate FPR over 48,216 conversations | `reports/unseen_english_v1` |
| LLM honeypot | Signed handoff, default-deny gate, live engagement verified; blocked from live mode by policy | `docs/HONEYPOT.md` |
| Human-gold promotion set | Not collected: 0 of 150 required conversations | `data/human_test/COLLECTION_STATUS.json` |
| Audio validation set | Not collected; no backend is promoted from one English smoke clip | `data/audio_validation/COLLECTION_STATUS.json` |

The ordinary mixed-source split produces very high scores, including 98.43% supporting test recall for XGBoost risk fusion at seed 42. Those values are not treated as real-world performance because every current positive label is silver, 71% of positive supervision is synthetic, and source/style shortcuts are measurable. The strict risk-fusion source audit confirms this: 27.71% source-macro FPR, 80.95% worst-source FPR, and only one of four held-out sources below the 5% gate. Promotion is blocked until the strict source gates and independently annotated human-gold gate pass.

## Data truth

The canonical build retains 49,950 negative and 2,256 positive conversations after deduplication. All 2,256 positives are source-silver; no positive conversation is human-gold. The negative pool is dominated by Banking77, DailyDialog, and Schema-Guided Dialogue. This imbalance is documented rather than hidden.

Two consequences of that imbalance are measured rather than assumed. First, because those three corpora contain no scam examples, source identity was close to a perfect label; restricting training to sources holding both labels and selecting character features raises held-out Hinglish ROC-AUC from 0.550 to 0.815 ± 0.023. Second, the old detector learned that financial vocabulary was itself suspicious: on 48,216 unseen English conversations it flagged 18.51% of Banking77. The new threshold is selected on a disjoint validation view with a per-source 5% gate and produces 4.54% Banking77 FPR on its test split, but this is not an unseen-source claim because Banking77 validation rows participate in threshold selection.

The frozen external YouTube source exposes the remaining gap. The old served detector found 3 of 243 scam calls; the calibrated candidate finds 15 of 243. That five-fold relative improvement is still only 6.17% recall, so deployment and live honeypot routing remain blocked.

`indian_cyber_scam_phonecall_hinglish` ships 10,000 rows of which 9,257 are exact duplicates, leaving 743 unique conversations (633 scam, 110 legitimate). The Source counts table in `docs/datasets/CANONICAL_BUILD_REPORT.md` is labelled input-before-deduplication for this reason. Only 21 legitimate Hinglish conversations reach the held-out test view, so no Hinglish false-positive rate in this project is a measurable quantity.

Raw, processed, audio, and model files are intentionally excluded from Git. Dataset identities, pinned revisions, licenses, download URLs, checksums, and blocked sources are recorded in `data/manifests/dataset_registry.json`. Gated, unlicensed, oversized, or language-inappropriate corpora are registered but never silently downloaded.

## Install

Python 3.10 is the supported local runtime.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements\dev.txt
.venv\Scripts\python.exe -m pytest -q
```

FFmpeg/ffprobe must be available on `PATH` for audio transcription.

## Rebuild and train

Run commands from the repository root in this order:

```powershell
.venv\Scripts\python.exe scripts\download_datasets.py --registry data\manifests\dataset_registry.json
.venv\Scripts\python.exe scripts\profile_raw_datasets.py
.venv\Scripts\python.exe scripts\build_canonical_dataset.py
.venv\Scripts\python.exe scripts\validate_processed_data.py
.venv\Scripts\python.exe scripts\build_evaluation_views.py

.venv\Scripts\python.exe scripts\train_model_ladder.py
.venv\Scripts\python.exe scripts\evaluate_leave_one_source_out.py
.venv\Scripts\python.exe scripts\train_risk_fusion.py
.venv\Scripts\python.exe scripts\evaluate_risk_fusion_loso.py
.venv\Scripts\python.exe scripts\train_classical_multitask.py
.venv\Scripts\python.exe scripts\train_mixed_source_candidate.py
.venv\Scripts\python.exe scripts\download_datasets.py --registry data\manifests\dataset_registry.json --dataset youtube_scam_phone_call_transcripts
.venv\Scripts\python.exe scripts\prepare_external_evaluation.py
.venv\Scripts\python.exe scripts\evaluate_external_text.py
```

The laptop deployment uses the completed compact classical multi-task artifact. Optional transformer comparison training remains available through `scripts/train_multitask_transformer.py`; it automatically resumes from a trainable-state-only checkpoint, but a GPU is recommended.

ASR backend comparison requires a compliant `data/audio_validation/manifest.jsonl`:

```powershell
.venv\Scripts\python.exe scripts\evaluate_asr.py
```

The independent human set can be frozen only after consent, redaction, two annotations, adjudication, evidence spans, quotas, and leakage checks pass:

```powershell
.venv\Scripts\python.exe scripts\freeze_human_test_set.py
```

## Local inference API

```powershell
.venv\Scripts\python.exe scripts\run_api.py
```

The service binds to `127.0.0.1:8000`; OpenAPI documentation is at `http://127.0.0.1:8000/docs`. The checked-in policy loads the calibrated mixed-source candidate, marks it `research_only_not_promoted`, disables the failed research-fusion override, and disables honeypot handoff. Sensitive entities are redacted unless a trusted caller explicitly opts in.

The container definition in `deploy/Dockerfile` runs as a non-root user, uses offline model loading, and expects model artifacts to be mounted read-only. It is not intended for direct Internet exposure.

## Repository map

- `configs/`: data schemas, model hyperparameters, evaluation protocols, and deployment policy.
- `data/manifests/`: pinned source registry; large data folders are local and ignored.
- `docs/`: annotation, architecture, dataset, evaluation, API, and collection documentation.
- `reports/`: compact metrics, limitations, run metadata, and artifact hashes.
- `scripts/`: reproducible data, training, evaluation, freeze, and service entrypoints.
- `src/arrestshield/`: reusable data, model, ASR, entity, fusion, inference, and API code.
- `tests/`: deterministic unit/contract tests; model-quality claims require the separate evaluation reports.

## Decision boundary

The runtime order is:

`audio/text -> ASR/formatting -> trained detector -> optional trained XGBoost fusion -> frozen threshold/hysteresis -> policy decision`

Only after an eligible detector and deployment policy approve a handoff may a separate LLM/RAG honeypot engage. See `docs/ARCHITECTURE.md` for the complete training and runtime diagrams.

The requirement-by-requirement evidence and honest completion boundary are recorded in `docs/COMPLETION_AUDIT.md`.
