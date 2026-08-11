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

## Important boundary

The detector will be trained and evaluated from versioned ML datasets. An LLM
may be invoked only after a calibrated risk-policy activation event, for the
adaptive fake-victim honeypot conversation.
