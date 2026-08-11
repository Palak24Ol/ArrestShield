# ArrestShield ML

This repository contains the data and implementation pipeline for the trained
multilingual ArrestShield scam detector. The LLM-powered honeypot is a separate
post-detection service and is not allowed to decide whether a call is a scam.

## Data layout

- `data/raw/`: immutable source files, grouped by dataset and pinned revision.
- `data/external/`: gated or manually supplied data; never committed.
- `data/interim/`: parsed records before unified annotation.
- `data/processed/`: normalized conversation/turn records.
- `data/splits/`: conversation-group train/validation/test manifests.
- `data/manifests/`: provenance, checksums, licenses, and download receipts.
- `docs/datasets/`: dataset inventory, citations, limitations, and decisions.

Run `python scripts/download_datasets.py --registry data/manifests/dataset_registry.json`
to fetch the approved seed data. Large, gated, or license-unverified datasets
are recorded in the registry but are not downloaded automatically.

## Current ML milestone

The first executable model is a multilingual word/character TF-IDF baseline
with a class-balanced linear classifier. It uses the fixed conversation-group
train/validation/test manifest, chooses its operating threshold on validation
only, and measures early detection on incremental conversation prefixes.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements\dev.txt
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv\Scripts\python.exe scripts\train_baseline.py
```

See `docs/models/BASELINE_TRAINING.md` for the protocol, artifact layout, and
limitations. Fine-grained scam type, manipulation tactic, and scam stage heads
will be trained only after enough human-reviewed labels are available.

The pre-registered comparison rule is documented in
`docs/MODEL_SELECTION_PROTOCOL.md`: maximize recall at a hard-negative false-
positive rate no greater than 5%, report three-seed mean and standard deviation,
and use stable scammer turns-to-detection as the first tiebreaker. The hybrid
PsyScam/digital-arrest label definitions are in `docs/ANNOTATION_CODEBOOK.md`.

## Important boundary

The detector will be trained and evaluated from versioned ML datasets. An LLM
may be invoked only after a calibrated risk-policy activation event, for the
adaptive fake-victim honeypot conversation.
