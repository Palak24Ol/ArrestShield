# Canonical Dataset Build Report

- Input conversations: 62,346
- Retained after exact deduplication/validation: 52,206
- Retained turns: 615,084
- Exact duplicates excluded: 10,140
- Leakage-safe groups: 51,503
- Multi-member similarity groups: 109
- Largest similarity group: 161
- Group leakage across splits: 0

## Split counts

- train: 36,626
- validation: 7,819
- test: 7,761

## Source counts

Input counts are **before** exact deduplication. Quote the retained column: it is
what training and evaluation actually see. The two differ by 13x for the Hinglish
source, so any figure taken from the input column overstates that corpus badly.

| Source | Input | Exact duplicates | **Retained** |
|---|---:|---:|---:|
| schema_guided_dialogue | 22,825 | 0 | **22,825** |
| daily_dialog | 13,118 | 798 | **12,320** |
| banking77 | 13,083 | 12 | **13,071** |
| synthetic_multi_agent_scam_conversation | 1,600 | 0 | **1,600** |
| synthetic_scam_dialogue | 1,600 | 2 | **1,598** |
| indian_cyber_scam_phonecall_hinglish | 10,000 | 9,257 | **743** |
| indian_multilingual_scam_messages | 120 | 71 | **49** |
| **Total** | **62,346** | **10,140** | **52,206** |

`indian_cyber_scam_phonecall_hinglish` ships 10,000 rows of which 92.6% are exact
duplicates, leaving 743 unique conversations: 633 scam and 110 legitimate. This is
the binding constraint on Hinglish detection quality and on Hinglish evaluation —
the held-out test view contains 21 legitimate conversations, which is too few to
estimate a false-positive rate. See `docs/ML_STATUS.md`.

## Guardrails

- Exact duplicates are excluded from conversations.jsonl and turns.jsonl but retained with flags in conversations.all.jsonl.
- High-similarity synthetic conversations remain included but are clustered into one split to prevent paraphrase leakage.
- Positive turn-level tactics/stages from public data are weak_rule labels, never gold.
- Original public train/test splits are retained only as provenance and ignored for ArrestShield splitting.
- HINMIX remains a separate language-robustness source and is not mislabeled as scam/non-scam.
