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

- banking77: 13,083
- daily_dialog: 13,118
- indian_cyber_scam_phonecall_hinglish: 10,000
- indian_multilingual_scam_messages: 120
- schema_guided_dialogue: 22,825
- synthetic_multi_agent_scam_conversation: 1,600
- synthetic_scam_dialogue: 1,600

## Guardrails

- Exact duplicates are excluded from conversations.jsonl and turns.jsonl but retained with flags in conversations.all.jsonl.
- High-similarity synthetic conversations remain included but are clustered into one split to prevent paraphrase leakage.
- Positive turn-level tactics/stages from public data are weak_rule labels, never gold.
- Original public train/test splits are retained only as provenance and ignored for ArrestShield splitting.
- HINMIX remains a separate language-robustness source and is not mislabeled as scam/non-scam.
