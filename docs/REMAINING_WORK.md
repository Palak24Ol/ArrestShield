# ArrestShield — What Still Needs Fixing

Prioritised by what a reviewer will actually challenge. P0 items are
documentation honesty problems and cost hours. P1 items make the project
defensible. P2 is future work you can name but need not do.

---

## P0 — Before you submit

The earlier status-report contradictions, stale README milestone, and ambiguous
pre-deduplication source counts have been corrected. The generated status PDF
must be rebuilt whenever `docs/ML_STATUS.md` changes.

The public headline must remain the frozen external result: **6.17% recall
(15/243)**. The 98.74% random-split F1 is retained only as evidence of the
dataset shortcut, never as an accuracy claim.

---

## P1 — Makes the project defensible (days to weeks)

### 5. Collect 40–60 real Hinglish multi-turn conversations
**This is the only item that raises the ceiling.** Everything else is polish.

You have 743 unique Hinglish conversations and 21 held-out negatives. No public
dataset fixes this — confirmed across multiple searches; real digital-arrest
recordings are criminal evidence and are not published.

Realistic sources, no ethics approval needed:
- Hindi/Hinglish scambaiting videos on YouTube — real scammer speech, publicly
  posted. Transcribe with the Whisper you already have wired.
- News segments that play recorded digital-arrest calls.
- Family and neighbours who received such calls, recounting them as a roleplay.

Freeze them as a test set only. Do not train on them. 40 conversations
independently annotated would be the only multi-turn Hinglish data in the project.

### 6. Transcribe any real Hinglish audio
Whisper has only ever been tested on one English clip. Run it on a single real
Hinglish call and report the WER, even informally. "We measured it" beats "we used
Whisper" in a viva, and it is 30 minutes of work.

### 7. Measure the honeypot
Engagement quality is currently unmeasured. Published work reports Information
Disclosure Rate and Human Acceptance Rate. Even a hand-scored 10-conversation
sample gives you a number and a citation.

---

## P2 — Future work (name it, don't do it)

8. Transformer comparison — HingRoBERTa-Mixed, MuRIL, XLM-R on GPU. The code and
    the causal-prefix fixes are in place; only compute is missing.
9. IndicWhisper evaluation against Whisper, judged on downstream detection F1
    rather than WER alone.
10. RAG knowledge layer with retrieval tests and prompt-injection controls.
11. Threat-intelligence extraction and dashboard.
12. ML → honeypot handoff in `live` mode, unreachable until the detector passes
    its promotion gate.

---

## Known negative results (keep these — they are evidence)

| Approach | Effect | Why it failed |
|---|---|---|
| LLM-generated Hinglish augmentation | AUC 0.756 → 0.743 | A third synthetic style adds no information |
| External BFSI Hinglish hard negatives | AUC 0.756 → 0.615 | Single-label source reintroduced the shortcut |
| XGBoost risk fusion | Source-macro FPR 7.0% → 27.7% | Learned the shortcut harder than the base model |

## Improvements completed on 14 August 2026

| Change | Result | Interpretation |
|---|---|---|
| Training-only feature ablation | Character AUC **0.880 ± 0.018**, then untouched test-source AUC **0.815 ± 0.023** | More robust to Romanized Hindi spelling without test-set selection |
| Platt calibration on disjoint validation rows | Internal Brier 0.13138 → **0.00428** | Internal calibration improved; it does not transfer externally |
| One shared threshold across three seeds | Maximum eligible-source validation FPR 4.98% | Previous seed-specific threshold instability removed |
| Frozen CC0 scam-call audit | Recall 1.23% → **6.17%** | Five-fold relative gain, but 93.8% miss rate still blocks promotion |

---

# Resume bullet points

Pick 3–4. Every number below is reproducible from the repository.

## Short version (standard resume)

- Built an end-to-end multilingual scam-call detection system in Python (ASR →
  detector → risk fusion → gated LLM honeypot) with 98 automated tests, artifact
  hash verification, and enforced architectural boundaries.

- Diagnosed a dataset shortcut inflating held-out F1 to a misleading 98.7%;
  corrected the training regime and feature representation to raise
  held-out-source Hinglish ROC-AUC from 0.550 to 0.815.

- Identified 92.6% exact duplication in a 10,000-row public Hinglish corpus,
  reducing it to 743 unique conversations and correcting every downstream metric
  in the project.

- Added calibration/threshold views and a pooled three-seed operating point,
  keeping every eligible validation source at or below 5% FPR while preventing
  seed-specific threshold selection.

- Built a frozen external audit over 243 real scammer-speech transcripts,
  measuring a five-fold recall improvement while correctly blocking deployment
  at an absolute recall of only 6.17%.

- Designed a post-detection LLM honeypot with HMAC-signed handoff, default-deny
  eligibility gating, and synthetic identifiers constructed to fail real
  validation (Verhoeff-invalid Aadhaar, reserved-range phone and UPI).

## Longer version (project section / portfolio)

**ArrestShield — Multilingual Digital-Arrest Scam Detection** · Python,
scikit-learn, XGBoost, PyTorch, Transformers, Whisper, FastAPI

- Engineered a reproducible ML pipeline over 52,206 conversations and 615,084
  turns: licensed-source ingestion, deduplication, canonical schema, and
  conversation-grouped splits with zero cross-split group leakage.

- Pre-registered a model-selection protocol before training — fixed false-positive
  operating point, three-seed reporting, hysteresis-based early-detection latency
  — to prevent post-hoc metric selection, then applied it to reject two of the
  project's own candidate models.

- Ran leave-one-source-out audits that refit representation and classifier per
  fold, revealing that near-perfect random-split scores were an artifact of
  single-label source corpora rather than genuine detection ability.

- Improved held-out-source Hinglish ROC-AUC from 0.550 ± 0.043 to 0.815 ± 0.023
  by restricting training to label-balanced sources and selecting character
  features under a fixed three-seed source-holdout protocol.

- Fixed causal label leakage in multi-task prefix windows, where early
  conversation windows inherited late-stage labels, invalidating early-detection
  measurements.

- Built a Groq-backed fake-victim honeypot enforcing a strict boundary: no LLM
  output contributes any feature, score, threshold, or label to detection —
  enforced by import-level tests and HMAC-signed handoff events.

## What to say if asked "what's your accuracy?"

> On a random split it scores 98.7% F1, but that number is an artifact — most of
> the legitimate conversations come from corpora that contain no scams, so the
> model can identify the source instead of the content. With an entire source
> held out, the candidate's Hinglish ROC-AUC is 0.815, but on a separate set of
> 243 real scam-call transcripts it detects only 6.17%. The bottleneck is matched,
> human-labelled real call data. So the pipeline is research-complete, while
> deployment remains deliberately blocked by the evidence gate.

That answer is worth more than a high number. Most candidates cannot explain why
their metric is wrong.
