# Mixed-source detector results

## Summary

Training on mixed-label sources only raises held-out Hinglish ROC-AUC from
`0.550` to `0.756` while deleting 94% of the training corpus. Two further
interventions were tested and both made it worse. The corpus, not the model, is
the binding constraint.

All numbers are three seeds (17, 42, 93). The primary metric is held-out-source
ROC-AUC: it is threshold-free, so it reports the representation rather than
threshold transfer.

## The regime change

Sources holding only one label let the detector use source identity as a proxy
for the label. Three corpora — `banking77`, `daily_dialog`,
`schema_guided_dialogue` — contribute 48,216 conversations and are all
legitimate. Restricting training to sources containing both labels removes that
shortcut by construction.

| Held-out source | Full corpus | Mixed-label sources only |
| --- | ---: | ---: |
| `indian_cyber_scam_phonecall_hinglish` | 0.550 ± 0.043 | **0.756 ± 0.055** |
| `synthetic_multi_agent_scam_conversation` | 0.979 ± 0.009 | 1.000 ± 0.000 |
| `synthetic_scam_dialogue` | 0.971 ± 0.012 | 1.000 ± 0.000 |
| `indian_multilingual_scam_messages` | 0.933 ± 0.054 | 0.956 ± 0.031 |

Training rows fell from 36,101 to 2,368 for the Hinglish fold. The gain is
roughly four times the seed-to-seed spread.

## Two interventions that failed

Both are recorded because a negative result on a plausible idea is evidence, and
because the leave-one-source-out audit is what caught them.

**LLM augmentation, 8 conversations.** Held-out Hinglish AUC `0.756 -> 0.743`.
The batch is far too small to conclude anything; it is reported only to show the
direction was not positive at that scale.

**External Hinglish BFSI hard negatives, 2,452 conversations.** Held-out
Hinglish AUC `0.756 -> 0.615`, and `indian_multilingual` `0.956 -> 0.622`. The
corpus is legitimate Hinglish banking, KYC, loan, and insurance dialogue — the
exact vocabulary the detector over-flags — and it still hurt. It is a
single-label source, so it reintroduced the shortcut the regime change removed,
and its uniform 2-turn length is a structural signature absent everywhere else.
It is not used.

## The measurement problem

The corpus contains no multi-turn Hinglish data.

| Source | Rows | Median turns | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| `indian_cyber_scam_phonecall_hinglish` scam | 633 | 1.0 | 1 | 1 |
| `indian_cyber_scam_phonecall_hinglish` legitimate | 110 | 1.0 | 1 | 1 |
| `indian_multilingual_scam_messages` | 49 | 1.0 | 1 | 1 |
| `synthetic_scam_dialogue` | 1,598 | 11–17 | 6 | 28 |
| `synthetic_multi_agent_scam_conversation` | 1,600 | 10–13 | 3 | 20 |
| `daily_dialog` | 12,320 | 7.0 | 2 | 35 |
| `schema_guided_dialogue` | 22,825 | 20.0 | 4 | 58 |

Consequences:

- Every Indian conversation is a single utterance. The Hinglish source is 743
  unique single-line records, not phone calls: 10,000 input rows collapse to 743
  after 92.6% exact-duplicate removal.
- Reported early detection on Hinglish (`median_turn 1.0, minimum 1, maximum 1`)
  is an artifact of having one turn, not evidence of fast detection. Stable
  turns-to-detection is exercised only by synthetic English.
- Turn count is close to a source identifier, so structure alone separates much
  of the corpus without reading any text.
- Held-out Hinglish has 21 legitimate conversations. A single flipped prediction
  moves FPR by 4.8 points, so no Hinglish false-positive-rate claim is supported
  at any operating point. Wilson intervals are reported and are correspondingly
  wide.

## What the detector currently does

ArrestShield classifies single Hinglish utterances as scam or legitimate, at
held-out-source ROC-AUC 0.756. Multi-turn Hinglish call detection is untrained
and unmeasured. The synthetic English folds reach 1.000 and should be read as
evidence that synthetic English is easy, not that the system is accurate.

Promotion remains blocked. The independently annotated frozen human-gold set is
uncollected, and it is now the only route to multi-turn Hinglish evaluation data
as well as the only route to a defensible FPR.

## Reproducing

```bash
python scripts/train_mixed_source_detector.py
```

Roughly four minutes on CPU, no GPU required. Machine-readable output is
`metrics.json`; `config.json` is the exact accepted configuration. No LLM
produced a feature, score, threshold, or label.
