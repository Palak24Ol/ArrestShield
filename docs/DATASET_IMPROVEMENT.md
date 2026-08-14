# Dataset improvement implementation

## Implemented sources

`youtube_scam_phone_call_transcripts` is approved only for frozen external
evaluation. Its 243 CC0 English transcripts are downloaded as a small archive,
deduplicated by normalized transcript hash, and converted into an ignored JSONL
manifest with `training_eligible: false` and
`threshold_selection_eligible: false`.

`infobay_hindi_call_center_sample` is registered as manual opt-in because the
archive is about 100 MB and supplies raw Hindi call-centre audio without trusted
reference transcripts. Download it only when someone is ready to redact and
annotate a small validation sample:

```powershell
.venv\Scripts\python.exe scripts\download_datasets.py `
  --registry data\manifests\dataset_registry.json `
  --dataset infobay_hindi_call_center_sample `
  --include-optional
```

The Hybrid Voice Phishing core, INDICA, PsyScam D2, the placeholder-heavy
400/400 corpus, the synthetic call-determination corpus, and the mislabeled
Fraud Call India source are registered with blocked or rejected states. The
downloader refuses them even when selected.

## Reproduce the external evaluation

```powershell
.venv\Scripts\python.exe scripts\download_datasets.py `
  --registry data\manifests\dataset_registry.json `
  --dataset youtube_scam_phone_call_transcripts
.venv\Scripts\python.exe scripts\prepare_external_evaluation.py
.venv\Scripts\python.exe scripts\evaluate_external_text.py
```

The evaluator loads a frozen artifact and threshold. It never fits, calibrates,
or selects anything from the external records.

## Reproduce the improved candidate

```powershell
.venv\Scripts\python.exe scripts\train_mixed_source_candidate.py
```

The trainer performs three leakage-controlled stages:

1. Compare word, character, and combined TF-IDF using
   leave-one-mixed-source-out ROC-AUC inside the training partition over seeds
   17, 42, and 93, then audit only the selected family on untouched test rows.
2. Split validation into disjoint calibration and threshold views, compare no
   calibration, Platt, and isotonic calibration, and select one threshold that
   satisfies the per-source FPR gate for every seed.
3. Export only the deployment-seed research bundle, including its calibrator,
   feature family, shared threshold, provenance, and explicit
   `llm_used_for_detection: false` flag.

The API loads this candidate but keeps research fusion and honeypot handoff off.
Promotion still requires an independently annotated human-gold call set.
