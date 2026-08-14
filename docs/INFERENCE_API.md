# Inference API contract

## Safety status

The API is a local research prototype bound to `127.0.0.1`. The checked-in calibrated character detector is explicitly marked `research_only_not_promoted`. The failed XGBoost fusion override and honeypot handoff are both disabled by policy. No endpoint invokes an LLM, and no LLM output contributes to a scam score.

## Endpoints

### `GET /healthz`

Returns process health, ASR availability, and `llm_used_for_detection: false`.

### `GET /v1/model`

Returns loaded detector families, seeds, thresholds, promotion status, and handoff policy.

### `POST /v1/detect/text`

Request:

```json
{
  "conversation_id": "demo-001",
  "turns": [
    {"speaker_role": "caller", "text": "Main CBI officer bol raha hoon."},
    {"speaker_role": "caller", "text": "Kisi ko mat batana, abhi paise transfer karo."}
  ],
  "include_sensitive_entities": false
}
```

The response includes the selected trained-ML score, threshold, base score, optional XGBoost fusion score, transparent auxiliary signals, privacy-redacted entities, model status, and a honeypot-boundary block. `is_scam` is a research output while `production_eligible` remains false.

The checked-in service configuration loads the completed `classical` multi-task backend. Its XGBoost scam-type, stage, and supported tactic heads appear under `auxiliary_signals`; they do not replace the API's selected scam decision source. The labels `phantom_riches`, `liking`, `pretext_trust`, `reciprocity`, `consistency_commitment`, and `social_proof` have no positive causal training windows, so they are returned as unavailable with null scores instead of fake always-negative predictions. If an auxiliary artifact is absent, the same field truthfully identifies its fallback as `deterministic_lexical_rules`. An explicitly configured `transformer` backend is also supported for a future completed comparison export.

`GET /v1/model` reports `calibration_method`, `feature_variant`,
`multitask_backend`, and `multitask_auxiliary_loaded`, making it possible to
verify which implementation is actually serving a request. The checked-in base
artifact uses character TF-IDF, SGD logistic loss, Platt scaling, seed 42, and
the shared threshold selected across seeds 17, 42, and 93.

### `POST /v1/detect/audio`

Multipart fields are `file`, optional `language_hint` (`en` or `hi`), optional `conversation_id`, and optional `include_sensitive_entities`. The service enforces extension and byte/duration limits, writes the upload only into an isolated temporary directory, transcribes locally, deletes the temporary file, and passes the transcript to the same trained detector. It never accepts a server-side file path from the client.

## Run locally

```powershell
.venv\Scripts\python.exe scripts\run_api.py
```

Interactive OpenAPI documentation is then available at `http://127.0.0.1:8000/docs`. Public binding, authentication, TLS, rate limiting, durable audit logging, and telecom integration are deliberately outside this local prototype and are required before any real deployment.
