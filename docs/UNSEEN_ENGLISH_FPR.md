# False-Positive Rate on Unseen English Conversation

Reproduce with `python scripts/evaluate_unseen_english.py`.
Machine-readable output: `reports/unseen_english_v1/metrics.json`.

## Why this measurement exists

The Hinglish held-out view contains **21 negative conversations**. One
conversation flipping moves its false-positive rate by 4.8 percentage points, so
no FPR claim from that view is supportable.

Under the mixed-source training regime, `banking77`, `daily_dialog` and
`schema_guided_dialogue` are excluded from training by construction, because each
contains only one label. That makes all 48,216 of their conversations genuinely
unseen negatives — a sample large enough for an interval narrow enough to mean
something.

## Result

Detector: TF-IDF word/char + class-weighted SGD, trained on mixed-label sources
only (2,893 conversations). Threshold chosen on mixed-source validation at the
pre-registered 5% FPR ceiling. Seeds 17, 42, 93.

| Unseen English source | Flagged as scam | n | FPR | 95% CI |
|---|---:|---:|---:|---|
| **banking77** | **2,420** | 13,071 | **18.51%** | [17.86%, 19.19%] |
| daily_dialog | 19 | 12,320 | 0.15% | [0.10%, 0.24%] |
| schema_guided_dialogue | 1 | 22,825 | 0.00% | [0.00%, 0.02%] |
| **Aggregate** | 2,440 | 48,216 | 5.06% | [4.87%, 5.26%] |

Recall on synthetic English scam conversations: **1.000 ± 0.000**.

## Interpretation

**The aggregate is misleading and must not be quoted alone.** 5.06% looks like
the model just clears its own 5% gate. It does not. The figure is an average over
two corpora the detector finds trivially easy (35,145 conversations, 20 total
false positives) and one it fails on.

`banking77` is short customer-service queries about cards, transfers, balances
and account access — the vocabulary of the deployment domain. On that domain the
detector flags **roughly one in five legitimate conversations as a scam**. In
deployment that is an unusable rate: a user would see false alarms on ordinary
banking calls often enough to ignore the system entirely.

Because `banking77` is single-turn queries rather than calls, this figure is not
a direct estimate of deployed FPR. It is a lower bound on the problem: the
detector has learned that financial vocabulary is itself suspicious, which is the
predictable consequence of a corpus whose legitimate examples are mostly general
chit-chat and task dialogue.

Perfect recall on synthetic English is not evidence of detection ability. Those
positives come from two corpora written by the same generator, and the model has
seen that generator's style during training.

## Threshold instability

The three seeds selected materially different operating points:

| Seed | Threshold | Total flagged |
|---|---:|---:|
| 17 | 0.6567 | 5,698 |
| 42 | 0.8960 | 813 |
| 93 | 0.8792 | 810 |

A **7.0× swing in false positives** from the random seed alone. The threshold is
selected on a validation view whose negatives are easy, so the score distribution
near the operating point is sparse and the choice is unstable. Any single-seed
operating point reported from this pipeline is not reproducible, and the
three-seed mean conceals rather than resolves the problem.

## Corroborating observation

A 3-second 440 Hz sine tone, transcribed by Whisper to `.`, receives a fusion
score of **0.9473** against a threshold of 0.9479 — 0.0006 below firing. Content-
free audio sits at the decision boundary. The score behaves as a style detector,
not as a calibrated probability, which is the same defect the `banking77` result
measures at scale.

## What would fix it

1. Legitimate in-domain conversation in the training pool. `colloquial_hinglish_bfsi`
   (2,452 legitimate Hinglish BFSI conversations) is fetched and wired but was not
   retained, because adding it as a single-label source reduced held-out Hinglish
   AUC from 0.756 to 0.615 — it reintroduced the shortcut it was meant to remove.
   The requirement is legitimate financial conversation from a source that *also*
   contains scam examples.
2. Probability calibration (Platt or isotonic) fitted on a validation view that
   contains hard negatives, so the score means something at the boundary.
3. A threshold-selection rule robust to sparse score regions — for example
   selecting on a pooled multi-seed score distribution rather than per seed.
