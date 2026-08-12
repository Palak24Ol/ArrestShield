# ML implementation status and evidence matrix

This matrix distinguishes implemented code, completed local training, verified artifacts, external collection gates, and claims that remain prohibited. A green unit test or high mixed-source score is not substituted for the specific evidence named below.

| Requirement | Implementation | Authoritative evidence | Current claim status |
|---|---|---|---|
| Canonical multilingual data | Download registry, parsers, normalization, exact/near deduplication, grouped splitting, validators | `docs/datasets/CANONICAL_BUILD_REPORT.json`, `docs/datasets/PROCESSED_DATA_VALIDATION.json`, processed file hashes | Engineering dataset complete; not human-gold |
| Binary baseline | Word/character TF-IDF plus class-weighted SGD, validation-only threshold | `reports/baseline_v1`, local artifact hash verification | Trained; research-only |
| XGBoost baseline | TF-IDF, SVD, XGBoost across seeds 17/42/93 | `reports/model_ladder_v1/metrics.json` | Trained; not promoted |
| Unseen-source control | Refits representation/model after excluding each mixed-label source | `reports/model_ladder_v1/loso_metrics.json` | Both classical families fail 5% FPR gate |
| Multi-task transformer | Binary, scam type, tactic, and stage heads; causal prefix labels; head-tail 256-token context; partial tactic supervision; multi-loss training | `src/arrestshield/multitask.py`, `scripts/train_multitask_transformer.py`, `configs/model/multitask_transformer.json` | Implementation and smoke export verified; full CPU run must produce `reports/multitask_transformer_v1` |
| Transformer recovery | Trainable-only atomic step checkpoints with optimizer, scheduler, RNG state, completed seeds, deterministic sortish batches | `tests/test_transformer_resume.py`, local `training_checkpoint.pt` during a run | Implemented; checkpoint deleted only after successful report/export |
| Risk fusion | XGBoost over out-of-fold base scores plus deterministic lexical/entity features | `reports/risk_fusion_v1/metrics.json`, local artifact SHA-256 | Three seeds trained; research-only |
| Strict fusion source audit | Excludes each source before every representation, OOF base-score, and fusion fit | `configs/evaluation/risk_fusion_loso.json`, `scripts/evaluate_risk_fusion_loso.py` | Report required at `reports/risk_fusion_v1/loso_metrics.json` |
| Audio-to-text | Local Whisper-family adapter with validation, WER/CER, downstream detector comparison | `reports/asr_smoke_v1`, `docs/ASR_EVALUATION.md` | English functionality verified; no Hindi/Hinglish backend promotion |
| ASR alternative | AI4Bharat Vistaar IndicWhisper Hindi registered behind compatible local checkpoint | `configs/model/asr_backends.json` | Registered GPU candidate; not installed on CPU laptop |
| Entity extraction | UPI, phone, email, URL, account/Aadhaar/OTP candidates, amounts, cases, authorities, payment apps | `src/arrestshield/entities.py`, `tests/test_entities.py` | Implemented; sensitive values redacted by default |
| Text/audio API | Local FastAPI routes, model/status introspection, upload limits, temporary deletion, redaction | `docs/INFERENCE_API.md`, `tests/test_api.py`, real local API smoke | Implemented; localhost research service |
| Secure packaging | Non-root container, offline model flags, read-only artifact mount guidance | `deploy/Dockerfile`, `deploy/README.md` | Definition implemented; Docker Desktop engine must run to build locally |
| Human-gold test | Consent, two annotations, adjudication, evidence spans, quotas, leak checks, write-once freeze | `configs/data/human_frozen_test_protocol.json`, `scripts/freeze_human_test_set.py` | 0/150 collected; promotion blocked |
| Hindi/Hinglish audio validation | Manifest schema, consent/license, PII and both-class gates, downstream selection rule | `configs/data/audio_validation_record.schema.json`, `scripts/evaluate_asr.py` | 0 records collected; ASR promotion blocked |
| Artifact integrity | Direct data/model/checkpoint hashes and policy-boundary checks | `scripts/verify_local_artifacts.py` | Final verification must pass without `--allow-incomplete-transformer` |
| Ablation protocol | Fixed multi-task, context, fusion, entity, lexical, and ASR variants with unchanged selection gates | `docs/VALIDATION_AND_TESTING.md` | Pre-registered; supporting compute, never a substitute for human-gold evidence |
| LLM boundary | Explicit false flags in artifacts/config/API; honeypot handoff disabled while research-only | `configs/deployment/api.json`, API contract tests, artifact verifier | Enforced: LLM never decides scam status |

## Completion gates for the local ML build

The local implementation is complete only when all of the following are true:

1. The full configured three-seed transformer run writes reconstructable artifacts and compact reports.
2. `MultiTaskPredictor` reconstructs that export and returns all four trained heads on a real text request.
3. The strict XGBoost fusion source audit finishes and its result is documented without promotion-by-average.
4. The full automated suite passes.
5. `scripts/verify_local_artifacts.py` passes with the transformer requirement enabled.
6. The real text and audio API path loads the final local artifacts, redacts sensitive entities, deletes temporary audio, and keeps honeypot handoff blocked.
7. The working tree is clean and `main` is synchronized with GitHub using the user's configured author identity.

The two external human/audio collection sets are not fabricated to satisfy these local gates. Their absence continues to block any production-readiness or real-world accuracy claim after the local implementation is complete.
