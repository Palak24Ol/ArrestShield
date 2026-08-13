# ArrestShield Binary Baseline Results

> **Superseded. Do not quote the F1 below as performance.**
>
> It is a random-split result produced under a corpus shortcut: 49,950 of 52,206
> conversations came from three sources containing only legitimate examples, so
> source identity was close to a perfect label. Holding out an entire source
> drops Hinglish ROC-AUC to 0.550; the corrected mixed-source regime reaches
> 0.756.
>
> Current numbers: `docs/ML_STATUS.md`. Retained for the record as the "before"
> figure in the shortcut analysis.

## Reproducible engineering run (research only)

- Code revision: `d7fe85917a467743451e706e51e247c26a68c2ec`
- Model: `arrestshield-tfidf-sgd-v1`
- Artifact SHA-256: `ec441b9ea3eab2a2cd23f917c40b97f9c6d06ed03b2daf49710302e905438d76`
- Artifact size: 3,452,642 bytes
- Operating threshold: `0.535`
- Train / validation / test conversations: 36,626 / 7,819 / 7,761

The threshold was selected on validation data only. The reported test split has
no conversation/similarity-group overlap with training or validation, but it is
not an independently sourced human-gold test and must not be treated as a
production benchmark.

## Test results

| Metric | Result |
|---|---:|
| Precision | 99.37% |
| Recall | 98.12% |
| F1 | 98.74% |
| F2 | 98.37% |
| False-positive rate | 0.027% |
| True negatives | 7,440 |
| False positives | 2 |
| False negatives | 6 |
| True positives | 313 |

Among 319 positive test conversations, the latched cumulative detector fired
on 98.43% by the first quarter of the conversation and on 100% by the halfway
point. The median first activation occurred after 13.33% of available turns.
The exact-prefix score can later fall below the threshold; the cumulative
metric therefore represents the intended operational latch behavior.

## Interpretation and limitations

These results validate the engineering pipeline, not production readiness.
The seed corpus contains substantial synthetic and silver-labeled scam data.
Several large dialogue sources contain only non-scam examples, which means the
model may learn dataset-origin or writing-style shortcuts. The test set also
contains only 115 Hinglish conversations and eight examples from the Indian
multilingual message source; several scam subtypes have fewer than ten positive
test examples.

Before real deployment, ArrestShield needs a human-reviewed gold evaluation
set of Indian scam and legitimate call transcripts collected independently of
the training sources. Audio transcription errors, regional code-switching,
unseen scam scripts, adversarial wording, and real background noise are not yet
represented by this evaluation.

This model alone decides the binary scam score. It does not call an LLM. The
separate LLM-powered honeypot may be activated only after an approved trained-ML
risk decision and cannot provide the scam label.

## Files

- `metrics.json`: complete overall, source, language, subtype, and early-detection metrics.
- `run_metadata.json`: environment, artifact checksum, threshold, and data counts.
- `config.json`: the exact accepted training configuration.
- `artifacts/models/baseline_v1/model.joblib`: local generated model; intentionally excluded from Git.
