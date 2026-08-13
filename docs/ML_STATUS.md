# ArrestShield ML Status

**This file is the single source of truth for project status.**
`docs/ArrestShield_ML_Status_and_Testing_Guide.pdf` is generated from it by
`scripts/build_status_pdf.py`. Edit this markdown and rebuild; never edit the PDF
directly.

The earlier hand-written PDF (12 August 2026) claimed "ML COMPONENT - Complete"
on its front page while its own page 4 listed the reasons production was blocked.
Generating the PDF from this file removes that failure mode by construction.

Last verified: 13 August 2026.

## One-sentence status

The software pipeline is complete and works end to end; the detector is not
accurate enough to deploy, and the reason is the training data, not the model.

## What is complete

| Component | State | Evidence |
|---|---|---|
| Dataset pipeline | Complete | 52,206 conversations, 615,084 turns, zero split-group leakage |
| Baseline + ladder models | Complete | SGD and XGBoost, seeds 17/42/93 |
| Multi-task auxiliary heads | Complete | Scam type, stage, 9 tactics |
| Risk fusion | Complete | XGBoost over out-of-fold base scores |
| ASR | Complete | Local Whisper-tiny; 0.00 WER on one English clip |
| Inference API | Complete | `/healthz`, `/v1/model`, `/v1/detect/text`, `/v1/detect/audio` |
| LLM honeypot | Complete | Signed handoff, default-deny gate, live Groq engagement verified |
| Audio → honeypot chain | Complete | `scripts/run_honeypot.py --audio` |
| Tests | 90 passing | `python -m pytest tests/ -q` |

Verified end to end on 13 August 2026:
`audio → Whisper → transcript → detector → HMAC-signed handoff → eligibility gate
→ live Groq persona reply → non-evidential transcript`.

## What is not complete

| Limitation | Measurement | Consequence |
|---|---|---|
| False positives on financial dialogue | 18.51% FPR on 13,071 unseen banking conversations | Unusable in the deployment domain |
| Threshold instability | Seeds chose 0.657 / 0.896 / 0.879 → 5,698 / 813 / 810 false positives | Operating point is not reproducible |
| Score calibration | 3s sine tone scores 0.9473 vs 0.9479 threshold | Score is a style detector, not a probability |
| Hinglish detection quality | Held-out-source ROC-AUC 0.756 ± 0.055 | Better than 0.550, still not deployable |
| Hinglish evaluation size | 21 held-out negatives | Hinglish FPR is not a measurable quantity |
| Positive label quality | All 2,256 positives source-silver; 71% synthetic | No real-call accuracy claim is possible |
| Hinglish ASR | Never tested | Only an English clip has been transcribed |
| ASR speed | 38s for 3s of audio on CPU | File upload only; not live monitoring |
| Human gold set | 0 of 150 collected | Final promotion gate unavailable |

## Corrected headline numbers

Do not quote the 98.74% F1 from `reports/baseline_v1/`. It is a random-split
number produced under a corpus shortcut and is not a performance claim.

The defensible numbers are:

- **Held-out-source Hinglish ROC-AUC: 0.756 ± 0.055** (3 seeds, threshold-free)
- **Unseen English FPR: 18.51%** on banking dialogue, 5.06% aggregate over 48,216
- **Synthetic English recall: 1.000** — not evidence of anything

## The corpus finding

`indian_cyber_scam_phonecall_hinglish` ships 10,000 rows. **9,257 are exact
duplicates.** 743 unique conversations survive: 633 scam, 110 legitimate.

The Source counts table in `docs/datasets/CANONICAL_BUILD_REPORT.md` lists input
counts, before the 10,140 exact duplicates were removed. Any figure derived from
that table overstates the Hinglish corpus by roughly 13×.

## The shortcut, and the fix that worked

49,950 of 52,206 conversations come from three corpora containing only legitimate
examples. "Does this look like banking77 / DailyDialog / schema-guided?" was
therefore close to a perfect label, and the detector learned it.

Restricting training to sources containing **both** labels removed the shortcut
by construction:

| Held-out source | Full corpus (36,101 train) | Mixed-only (2,368 train) |
|---|---:|---:|
| **indian_cyber_scam_hinglish** | **0.550 ± 0.043** | **0.756 ± 0.055** |
| synthetic_multi_agent | 0.979 ± 0.009 | 1.000 ± 0.000 |
| synthetic_scam_dialogue | 0.971 ± 0.012 | 1.000 ± 0.000 |

Deleting 94% of the training data improved held-out Hinglish AUC by 0.21, roughly
four times the seed noise. Reproduce with
`python scripts/train_mixed_source_detector.py`.

## Approaches tested and rejected

Both were expected to help and did not. They are recorded because a negative
result from a controlled experiment is evidence.

| Approach | Result | Why it was dropped |
|---|---|---|
| LLM-generated Hinglish augmentation (Groq) | AUC 0.756 → 0.743 | No gain at n=8; a third synthetic style does not add information |
| External BFSI Hinglish hard negatives (2,452 rows) | AUC 0.756 → **0.615** | Single-label source reintroduced the shortcut it was meant to remove |

The second is the more instructive: adding legitimate in-domain conversation
*hurt*, because it arrived as a source containing no scam examples. The
requirement is legitimate financial conversation from a source that also contains
scams — which is what collection has to provide.

## Boundary guarantee

No LLM produces a feature, score, threshold, or label anywhere in detection.

- `src/arrestshield/honeypot.py` imports no detector module; a test enforces it.
- The honeypot requires an HMAC-signed detector event and cannot be reached otherwise.
- Honeypot transcripts are stamped `excluded_from_detector_training: true`.
- Whisper is a trained ASR model, not an LLM, and produces words only.
- LLM-generated training rows are `provenance: llm_synthetic`, train-split only,
  weighted 0.5, and structurally barred from validation and test.

## How to verify on this PC

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
.venv\Scripts\python.exe scripts\train_mixed_source_detector.py
.venv\Scripts\python.exe scripts\evaluate_unseen_english.py
.venv\Scripts\python.exe scripts\run_api.py
.venv\Scripts\python.exe scripts\run_honeypot.py --demo --research-mode
```

The honeypot requires `GROQ_API_KEY` in a git-ignored `.env`. Without it every
other command still runs.
