# ArrestShield ML completion audit

This audit records the evidence for the completed local research implementation. It does not promote the detector for real-world use. External human-gold and representative Hindi/Hinglish audio collection remain explicit future data gates.

| Requirement | Authoritative evidence | Result |
|---|---|---|
| Versioned dataset ingestion and licensing | `data/manifests/dataset_registry.json`, download/profile scripts and dataset inventory | Implemented; inaccessible or unsuitable sources are registered rather than silently substituted |
| Canonical multilingual conversation data | `docs/datasets/CANONICAL_BUILD_REPORT.json`, `docs/datasets/PROCESSED_DATA_VALIDATION.json` | 52,206 conversations and 615,084 turns validated |
| Leakage-resistant splitting | Processed validation report, split manifest hash, grouped-split tests | Conversation/group leakage is zero; source shortcuts are evaluated separately |
| Binary and XGBoost baselines | `reports/baseline_v1`, `reports/model_ladder_v1` | SGD and SVD-XGBoost trained with seeds 17, 42, and 93 |
| Calibrated mixed-source candidate | `reports/mixed_source_candidate_v2` | Character features selected by source holdout; Platt scaling and one three-seed threshold fitted on disjoint validation views |
| Frozen external audit | `reports/external_text_v1` | 15/243 scam calls detected (6.17% recall); promotion remains blocked |
| Rigorous source evaluation | `reports/model_ladder_v1/loso_metrics.json`, `reports/risk_fusion_v1/loso_metrics.json` | Strict 5% FPR gate fails and promotion remains blocked; high mixed-source scores are not presented as real-world accuracy |
| Multilingual multi-task detection | `reports/classical_multitask_v1`, reconstructable local artifact | Three-seed XGBoost heads trained for scam type, stage, and nine supported tactics; unsupported labels return unavailable |
| Transformer comparison path | `src/arrestshield/multitask.py`, transformer trainer, recovery tests and documentation | Causal labels, head-tail context, masked losses, and exact resume implemented; full CPU fit is not claimed or deployed |
| Context and early detection | Selection protocol, prefix-window tests, risk-fusion metrics | Causal prefixes cannot inherit future labels; stable scammer-turn latency is reported at the fixed operating point |
| Class imbalance controls | Model configs and training modules | Source-balanced sampling, provenance weights, class weights, negative caps, sparse-label gating, and three seeds implemented |
| XGBoost risk fusion | `reports/risk_fusion_v1`, local artifact hash | Trained on out-of-fold base scores plus deterministic lexical/entity features; remains research-only after strict audit failure |
| Whisper audio-to-text | ASR adapter/tests, model manifest hashes, `reports/integration_v1/runtime_smoke.json` | Real local FLAC request transcribed and temporary upload deleted; Hindi/Hinglish backend promotion remains blocked pending data |
| Entity extraction and privacy | Entity module/tests and real text smoke | Operational entities extracted; sensitive values redacted by default |
| Inference API | API contract/tests and runtime smoke | Text/audio/model/health paths work with actual artifacts; configured auxiliary backend is disclosed |
| Artifact integrity | `reports/verification_v1/verification.json` | 34 of 34 checksum, status, and boundary checks passed with no skips |
| Automated testing | Full local pytest run on 2026-08-15 | 98 tests passed |
| Deployment packaging | Localhost policy, non-root container definition, offline model paths | Implemented as a local research service, not an Internet-ready production service |
| LLM/honeypot separation | Config, manifests, verifier, API responses and tests | LLM use for detection is false; auxiliary heads cannot decide `is_scam`; honeypot handoff is disabled |
| Reproducibility and documentation | Configs, scripts, reports, architecture/API/testing docs | Commands, seeds, thresholds, hashes, limitations, folder layout, and decision boundary are documented |

## Honest completion boundary

The local ML engineering build is complete and reconstructable. It is deliberately labeled `research_only_not_promoted`. The current positives are all silver, 71% of positive supervision is synthetic, the external scam-call audit misses 93.8% of examples, the independent 150-conversation human-gold set is at 0/150, and the representative Hindi/Hinglish audio-validation set is uncollected. These facts prevent a production or real-world accuracy claim but do not invalidate completion of the local research pipeline.
