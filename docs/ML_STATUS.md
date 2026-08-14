# ArrestShield ML Status

**This file is the single source of truth for project status.**
`docs/ArrestShield_ML_Status_and_Testing_Guide.pdf` is generated from it by
`scripts/build_status_pdf.py`. Edit this markdown and rebuild; never edit the PDF
directly.

The earlier hand-written PDF (12 August 2026) claimed "ML COMPONENT - Complete"
on its front page while its own page 4 listed the reasons production was blocked.
Generating the PDF from this file removes that failure mode by construction.

Last verified: 15 August 2026.

## One-sentence status

The research pipeline works end to end and now uses a calibrated,
seed-stable character model; external scam-call recall is still only 6.17%, so
the detector is not promoted and cannot automatically trigger the honeypot.

## What is complete

| Component | State | Evidence |
|---|---|---|
| Dataset pipeline | Complete | 52,206 conversations, 615,084 turns, zero split-group leakage |
| Baseline + ladder models | Complete | SGD and XGBoost, seeds 17/42/93 |
| Calibrated mixed-source candidate | Complete | Character TF-IDF + SGD + Platt scaling; one threshold across three seeds |
| Frozen external-text evaluation | Complete | 243 CC0 scam-call transcripts excluded from training and selection |
| Multi-task auxiliary heads | Complete | Scam type, stage, 9 tactics |
| Risk fusion | Complete | XGBoost over out-of-fold base scores |
| ASR | Complete | Local Whisper-tiny; 0.00 WER on one English clip |
| Inference API | Complete | `/healthz`, `/v1/model`, `/v1/detect/text`, `/v1/detect/audio` |
| LLM honeypot | Complete | Signed handoff, default-deny gate, live Groq engagement verified |
| Audio → honeypot chain | Complete | `scripts/run_honeypot.py --audio` |
| Tests | 98 passing | `python -m pytest tests/ -q` |

Verified end to end on 13 August 2026: audio to Whisper to transcript to trained
detector to HMAC-signed handoff to eligibility gate to live Groq persona reply
to a non-evidential transcript.

## What is not complete

| Limitation | Measurement | Consequence |
|---|---|---|
| External scam-call recall | 15/243 detected (6.17%); 228 missed | Decisive promotion blocker |
| External calibration transfer | Brier loss 0.9703 on the external positive-only set | Internally calibrated scores do not transfer to real scammer speech |
| Financial-dialogue false positives | 4.54% on the Banking77 test partition | Passes the configured limit, but Banking77 validation data participated in threshold selection |
| Hinglish detection quality | Held-out-source ROC-AUC 0.815 ± 0.023 | Improved ranking, still not a deployable accuracy claim |
| Hinglish evaluation size | 21 held-out negatives | Hinglish FPR is not a measurable quantity |
| Positive label quality | All 2,256 positives source-silver; 71% synthetic | No real-call accuracy claim is possible |
| Hinglish ASR | Never tested | Only an English clip has been transcribed |
| ASR speed | 38s for 3s of audio on CPU | File upload only; not live monitoring |
| Human gold set | 0 of 150 collected | Final promotion gate unavailable |

## Corrected headline numbers

Do not quote the 98.74% F1 from `reports/baseline_v1/`. It is a random-split
number produced under a corpus shortcut and is not a performance claim.

The defensible candidate numbers are:

- **Held-out-source Hinglish ROC-AUC: 0.815 ± 0.023** (character features,
  three seeds, threshold-free)
- **Frozen external scam-call recall: 6.17%** (15/243; positive-only, so FPR
  cannot be measured)
- **Banking77 test FPR: 4.54%**, with the important caveat that Banking77
  validation rows participated in threshold selection
- **Shared threshold: 0.0349148595** across seeds 17/42/93; every eligible
  negative validation source remains at or below 5% FPR for every seed

## The corpus finding

`indian_cyber_scam_phonecall_hinglish` ships 10,000 rows. Of those, 9,257 are
exact duplicates. Only 743 unique conversations survive: 633 scam and 110
legitimate.

The Source counts table in `docs/datasets/CANONICAL_BUILD_REPORT.md` lists input
counts, before the 10,140 exact duplicates were removed. Any figure derived from
that table overstates the Hinglish corpus by roughly 13×.

## The shortcut, and the fixes that worked

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

The follow-up ablation compared word, character, and combined TF-IDF features
using source holdouts inside the training partition only. Character features
scored `0.880 ± 0.018` on the training-held-out Hinglish source, versus
`0.711 ± 0.013` for word features and `0.818 ± 0.029` for combined features.
After selection, an untouched test-source audit measured `0.815 ± 0.023` for
the selected character family. Platt calibration reduced internal validation
Brier loss from `0.13138` to `0.00428`. Calibration and threshold rows are
disjoint, and a single threshold is used for all three seeds. Reproduce the
complete candidate with `python scripts/train_mixed_source_candidate.py`.

## Frozen external evaluation

The CC0 YouTube scam-call corpus is imported into a separate external manifest.
Its records are marked ineligible for training, calibration, threshold selection,
and model-family selection. Exact transcript duplicates are removed. It contains
243 English positive conversations, so it measures recall only.

| Frozen detector | Detected | Missed | Recall |
|---|---:|---:|---:|
| Previous served SGD | 3 | 240 | 1.23% |
| Calibrated character candidate | 15 | 228 | **6.17%** |

The five-fold relative improvement is useful engineering evidence, but the
absolute miss rate is 93.8%. The API therefore reports
`research_only_not_promoted`, risk fusion is disabled by policy, and honeypot
handoff remains disabled.

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
.venv\Scripts\python.exe scripts\train_mixed_source_candidate.py
.venv\Scripts\python.exe scripts\evaluate_external_text.py
.venv\Scripts\python.exe scripts\run_api.py
.venv\Scripts\python.exe scripts\run_honeypot.py --demo --research-mode
```

The honeypot requires `GROQ_API_KEY` in a git-ignored `.env`. Without it every
other command still runs.
