# Binary Baseline Training

This milestone implements the first executable ArrestShield detector. It is a
binary scam/non-scam baseline based on multilingual word and character TF-IDF
features with a class-balanced linear classifier.

## Safety boundary

The baseline is a trained machine-learning detector. It never calls an LLM.
Only a downstream policy service may start the separately implemented
LLM-powered fake-victim honeypot after the detector crosses an approved risk
threshold. Honeypot responses must never feed back into the ground-truth scam
decision.

## Reproducible protocol

1. Use the immutable canonical `conversations.jsonl` file.
2. Join each record to the precomputed conversation-group split manifest.
3. Train only on the `train` split.
4. Select one F2-oriented threshold only on `validation`.
5. Report final metrics once on the untouched `test` split.
6. Report source, language, and positive scam-type slices.
7. Measure early detection by scoring incrementally longer turn prefixes.

The model artifact is deliberately excluded from Git because it is generated
and can be large. Its metadata, configuration, SHA-256 hash, and evaluation
report are saved under `reports/baseline_v1/` and may be versioned.

## Commands

Create the environment and install pinned dependencies:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements\dev.txt
```

Run tests and the full training protocol:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.venv\Scripts\python.exe scripts\train_baseline.py
```

Run one local prediction after training:

```powershell
.venv\Scripts\python.exe scripts\predict_baseline.py "Main CBI se bol raha hoon, kisi ko mat batana."
```

The seed corpus contains substantial synthetic and silver-labeled scam data.
Consequently this result is an engineering baseline, not a production claim.
Gold, human-reviewed Indian call annotations are required before deployment or
training the tactic, stage, and fine-grained scam-type heads.
