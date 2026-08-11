# Inference API contract

## Safety status

The API is a local research prototype bound to `127.0.0.1`. Current detectors are explicitly marked `research_only_not_promoted`; XGBoost fusion is allowed only because the checked-in configuration opts into research mode. Honeypot handoff is disabled. No endpoint invokes an LLM, and no LLM output contributes to a scam score.

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

When a complete exported multi-task artifact is available, the service loads its scam-type, tactic, and stage heads as `auxiliary_signals`. Its feasibility binary output is reported but does not replace the API's selected decision source. Before that artifact exists, the same field truthfully identifies its fallback as `deterministic_lexical_rules` and states that those values are not transformer predictions.

### `POST /v1/detect/audio`

Multipart fields are `file`, optional `language_hint` (`en` or `hi`), optional `conversation_id`, and optional `include_sensitive_entities`. The service enforces extension and byte/duration limits, writes the upload only into an isolated temporary directory, transcribes locally, deletes the temporary file, and passes the transcript to the same trained detector. It never accepts a server-side file path from the client.

## Run locally

```powershell
.venv\Scripts\python.exe scripts\run_api.py
```

Interactive OpenAPI documentation is then available at `http://127.0.0.1:8000/docs`. Public binding, authentication, TLS, rate limiting, durable audit logging, and telecom integration are deliberately outside this local prototype and are required before any real deployment.
