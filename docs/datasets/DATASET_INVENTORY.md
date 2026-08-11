# ArrestShield Dataset Inventory

## What is downloaded automatically

The approved seed bundle contains:

1. Indian Hinglish scam-call text (core domain seed).
2. Indian multilingual scam/legitimate messages (small auxiliary set).
3. Apache-licensed synthetic multi-turn scam and non-scam conversations.
4. BANKING77 legitimate banking queries as hard negatives.
5. DailyDialog legitimate general multi-turn dialogue as hard negatives.
6. Google's Schema-Guided Dialogue corpus for legitimate banking, payment,
   travel, and support conversations.
7. Small gold HINMIX romanized/noisy Hinglish sets for language robustness.

Every source is pinned to an immutable revision. Source payloads are stored
unchanged below `data/raw/<dataset_id>/`; a download receipt records the URL,
SHA-256, byte count, timestamp, license, and revision.

## What is intentionally not downloaded

- **INDICA telecom fraud calls:** highly relevant but approximately 378.6 GB,
  with no license stated by the dataset API/card at the inspected revision.
- **Sonexis Hinglish speech:** gated; the user must accept its conditions on
  Hugging Face. It is non-commercial and not scam-labelled.
- **VISHGUARD:** promising 3,000-call synthetic vishing corpus, but the data
  record/download terms must be verified separately from the article license.
- **Full HINMIX training corpus:** approximately 2.45 GB and not scam-labelled;
  only the small gold/robustness splits are required at this stage.

## Critical gap

No public bundle covers the full ArrestShield label space with trustworthy
Indian, English/Hindi/Hinglish, turn-level annotations for scam presence, scam
type, manipulation tactics, and scam stage. We therefore still must create the
project-authored ArrestShield corpus. Public and synthetic datasets are seeds,
augmentation, hard negatives, and robustness data—not a substitute for the
primary annotated corpus or frozen real-world evaluation set.

## Rules before training

- Parse sources into the canonical conversation/turn schema without editing raw
  files.
- Assign `source_type`, `source_dataset`, `source_revision`, `license`, and
  `scenario_group_id` to every record.
- Deduplicate exact and near-duplicate conversations across all synthetic sets.
- Split by conversation/scenario/template, never by isolated turn.
- Never use public repository split labels automatically; build ArrestShield
  train/validation/test groups after deduplication.
- Never put synthetic-only data in the headline test result.
- Preserve attribution and ShareAlike/non-commercial boundaries in any derived
  release.
