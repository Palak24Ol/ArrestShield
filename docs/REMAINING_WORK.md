# ArrestShield — What Still Needs Fixing

Prioritised by what a reviewer will actually challenge. P0 items are
documentation honesty problems and cost hours. P1 items make the project
defensible. P2 is future work you can name but need not do.

---

## P0 — Before you submit (a few hours total)

### 1. Withdraw or regenerate the status PDF
`docs/ArrestShield_ML_Status_and_Testing_Guide.pdf` page 1 says
**"ML COMPONENT — Complete"**. Its own page 4 lists the reasons production is
blocked. This is the single thing in the repository a reviewer could fairly call
dishonest, and it is contradicted by your own evidence.

`docs/ML_STATUS.md` is the corrected replacement. Either regenerate the PDF from
it or delete the PDF and reference the markdown.

### 2. Stop quoting 98.74% F1 anywhere
It appears in `reports/baseline_v1/RESULTS.md` and is the headline in the PDF.
It is a random-split number produced under a corpus shortcut. Replace with:

- Held-out-source Hinglish ROC-AUC **0.756 ± 0.055**
- Unseen English FPR **18.51%** on banking dialogue

Keep the 98.74% in the report as the *before* number in the shortcut narrative —
it is a good story, just not a result.

### 3. Fix the Source counts table
`docs/datasets/CANONICAL_BUILD_REPORT.md` lists source counts that are **input**
counts, before 10,140 exact duplicates were removed. It reads as if the Hinglish
corpus has 10,000 conversations; it has 743. Label the column, or add the
retained column beside it.

### 4. Update README's "Current ML milestone"
It still describes the TF-IDF baseline as the current milestone. The mixed-source
regime supersedes it.

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

### 6. Calibrate the probability
A 3-second sine tone scores 0.9473 against a 0.9479 threshold. The score is not a
probability. Fit Platt scaling or isotonic regression on a validation view that
contains hard negatives. Cheap, and it directly attacks the banking77 failure.

### 7. Make threshold selection reproducible
Seeds 17/42/93 chose thresholds 0.657 / 0.896 / 0.879, producing 5,698 / 813 /
810 false positives — a 7× swing from the seed alone. Select on a pooled
multi-seed score distribution rather than per seed, and report the spread.

### 8. Transcribe any real Hinglish audio
Whisper has only ever been tested on one English clip. Run it on a single real
Hinglish call and report the WER, even informally. "We measured it" beats "we used
Whisper" in a viva, and it is 30 minutes of work.

### 9. Measure the honeypot
Engagement quality is currently unmeasured. Published work reports Information
Disclosure Rate and Human Acceptance Rate. Even a hand-scored 10-conversation
sample gives you a number and a citation.

---

## P2 — Future work (name it, don't do it)

10. Transformer comparison — HingRoBERTa-Mixed, MuRIL, XLM-R on GPU. The code and
    the causal-prefix fixes are in place; only compute is missing.
11. IndicWhisper evaluation against Whisper, judged on downstream detection F1
    rather than WER alone.
12. RAG knowledge layer with retrieval tests and prompt-injection controls.
13. Threat-intelligence extraction and dashboard.
14. ML → honeypot handoff in `live` mode, unreachable until the detector passes
    its promotion gate.

---

## Known negative results (keep these — they are evidence)

| Approach | Effect | Why it failed |
|---|---|---|
| LLM-generated Hinglish augmentation | AUC 0.756 → 0.743 | A third synthetic style adds no information |
| External BFSI Hinglish hard negatives | AUC 0.756 → 0.615 | Single-label source reintroduced the shortcut |
| XGBoost risk fusion | Source-macro FPR 7.0% → 27.7% | Learned the shortcut harder than the base model |

---

# Resume bullet points

Pick 3–4. Every number below is reproducible from the repository.

## Short version (standard resume)

- Built an end-to-end multilingual scam-call detection system in Python (ASR →
  detector → risk fusion → gated LLM honeypot) with 90 automated tests, artifact
  hash verification, and enforced architectural boundaries.

- Diagnosed a dataset shortcut inflating held-out F1 to a misleading 98.7%;
  corrected the training regime to raise genuine unseen-source ROC-AUC from 0.550
  to 0.756 while removing 94% of the training data.

- Identified 92.6% exact duplication in a 10,000-row public Hinglish corpus,
  reducing it to 743 unique conversations and correcting every downstream metric
  in the project.

- Measured false-positive rate across 48,216 unseen conversations, exposing an
  18.5% failure rate on banking dialogue that a 5.1% aggregate had concealed.

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

- Improved unseen-source Hinglish ROC-AUC from 0.550 ± 0.043 to 0.756 ± 0.055 by
  restricting training to label-balanced sources, a ~4σ improvement achieved by
  discarding 33,733 training examples.

- Fixed causal label leakage in multi-task prefix windows, where early
  conversation windows inherited late-stage labels, invalidating early-detection
  measurements.

- Built a Groq-backed fake-victim honeypot enforcing a strict boundary: no LLM
  output contributes any feature, score, threshold, or label to detection —
  enforced by import-level tests and HMAC-signed handoff events.

## What to say if asked "what's your accuracy?"

> On a random split it scores 98.7% F1, but that number is an artifact — most of
> the legitimate conversations come from corpora that contain no scams, so the
> model can identify the source instead of the content. When I hold out an entire
> source, ROC-AUC on real Hinglish is 0.756. The bottleneck is that only 743
> unique Hinglish conversations exist in public data, and 71% of the positives
> are synthetic. So the honest framing is that the pipeline is production-grade
> and the evidence isn't yet — which is why I built the audits that show it.

That answer is worth more than a high number. Most candidates cannot explain why
their metric is wrong.
