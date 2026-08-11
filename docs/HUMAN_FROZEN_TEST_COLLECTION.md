# Human Frozen Test Set Collection

## Why this is required

The current canonical corpus cannot support a real-world performance claim. All 2,256 positive conversations are source-silver, 1,599 are synthetic, and the 49,950 negatives are dominated by three negative-only sources. A random conversation split therefore rewards source and writing-style recognition. The strict leave-one-source-out audit exposed this failure.

`project_authored_arrestshield_frozen_test_v1` is a separate, independently annotated evaluation artifact. It is never used for training, threshold selection, prompt development, data cleaning decisions, or model-family selection.

## Collection design

Collect at least 150 consented, deidentified conversations. Both scam and non-scam conversations must be collected through every included channel so that collection channel cannot become the label.

Preferred channel for the first version:

- Recruit volunteers and record or transcribe role-played calls.
- Randomly assign each volunteer to both scam and benign scenarios using the same device and collection procedure.
- Ask the caller to improvise rather than read a fixed template.
- Include English, Hindi, and Hinglish in both classes.
- Do not use an LLM to write, paraphrase, translate, label, or adjudicate any frozen-test conversation.

Consented deidentified victim recollections or call transcripts may be included as a separately reported channel. Never collect an active scam call, contact a suspected scammer, or expose a participant to financial or safety risk.

## Required quotas

The executable quotas are in `configs/data/human_frozen_test_protocol.json`:

- at least 150 accepted conversations;
- at least 60 scam and 60 non-scam conversations;
- at least 35 English, 35 Hindi, and 35 Hinglish conversations;
- positive coverage for digital arrest, KYC/bank, courier/customs, OTP/account takeover, and other impersonation scams;
- both labels represented in every collection channel.

These are minimums, not a prevalence estimate. Report metrics by class, language, scam type, and collection channel; do not report accuracy as if the sample represented population prevalence.

## Roles and blinding

Use four logically separate roles where possible:

1. A collector obtains consent and removes PII.
2. Annotator A labels the deidentified transcript independently.
3. Annotator B labels it independently without seeing A's labels.
4. A third person adjudicates every disagreement and records a reason.

The model developer must not reveal model predictions to annotators before the set is frozen. Annotator and adjudicator identifiers may be pseudonyms, but must remain stable so independence can be checked.

## Input files

All inputs use one JSON object per line. Store them under the ignored `data/human_test` directories; they may contain sensitive research data and are not pushed to GitHub.

### Intake

```json
{"conversation_id":"AHFT-0001","collection_channel":"volunteer_roleplay","language_profile":["hinglish"],"turns":[{"speaker_role":"caller","text":"<deidentified text>"},{"speaker_role":"recipient","text":"<deidentified text>"}],"consent_confirmed":true,"pii_redacted":true,"llm_generated":false}
```

### Independent annotation

Provide two rows per conversation in `annotations.jsonl`:

```json
{"conversation_id":"AHFT-0001","annotator_id":"ANN-A","is_scam":1,"scam_type":"digital_arrest","tactics":["authority_impersonation"],"stage":"authority_claim","evidence_spans":[{"tactic":"authority_impersonation","turn_id":0,"text":"<short deidentified evidence>"}],"llm_used":false}
```

### Adjudication

Include a row only when the two independent annotations disagree. The row contains the final labels plus `adjudicator_id`, `reason`, and `llm_used:false`.

## Freeze command

```powershell
.\.venv\Scripts\python.exe scripts\freeze_human_test_set.py `
  --intake data\human_test\raw\intake.jsonl `
  --annotations data\human_test\annotations\annotations.jsonl `
  --adjudications data\human_test\annotations\adjudications.jsonl
```

The command refuses to freeze when any gate fails. It checks consent, PII redaction, human-only creation and annotation, independent annotators, third-party adjudication, evidence spans, quotas, collection-channel balance, duplicate transcripts, and exact ID/transcript leakage against the canonical corpus.

The output is write-once and includes a SHA-256 manifest. Keep the frozen labels inaccessible during model development. Evaluate it only after the family and operating threshold have been fixed on validation data.

## Current status

No human records have been fabricated. `data/human_test/COLLECTION_STATUS.json` remains `not_collected` until the workflow succeeds. Until then, transformer scores are feasibility results and the project must not claim validated real-world performance.
