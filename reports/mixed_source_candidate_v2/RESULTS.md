# Calibrated mixed-source candidate v2

## Outcome

The candidate selects character TF-IDF, an SGD logistic classifier, and Platt
calibration. One threshold (`0.0349148595`) is shared by seeds 17, 42, and 93.
The artifact remains `research_only_not_promoted` and cannot enable the
honeypot.

## Feature ablation

Feature families were selected using leave-one-mixed-source-out ROC-AUC inside
the training partition. No external, validation, or test record selected the
feature family.

| Features | Training-held-out Hinglish ROC-AUC | Training source-macro ROC-AUC |
|---|---:|---:|
| Word | 0.711 ± 0.013 | 0.877 ± 0.003 |
| Character | **0.880 ± 0.018** | **0.920 ± 0.008** |
| Word + character | 0.818 ± 0.029 | 0.920 ± 0.007 |

Character features are substantially more robust to Romanized Hindi spelling
variation than word n-grams in the current corpus. After character features were
selected, the untouched test-source audit measured Hinglish ROC-AUC
`0.815 ± 0.023` and source-macro ROC-AUC `0.937 ± 0.006`. Those test results did
not participate in feature selection.

## Calibration and threshold

Validation rows are deterministically divided into disjoint calibration and
threshold views within source/label strata. Platt scaling and the uncalibrated
scores have equal threshold-view recall, but Platt reduces mean Brier loss from
`0.13138` to `0.00428`. Isotonic calibration has lower recall.

The shared threshold is the lowest value that keeps every negative source with
at least 50 validation negatives at or below 5% FPR for every seed. Validation
recall is `0.9762` on average, with a minimum seed recall of `0.9714`.

On the deployment-seed test split, Banking77 FPR is `4.54%` and overall recall
is `98.75%`. These are supporting split results, not real-world accuracy: the
threshold has seen Banking77 validation data and the positives remain silver or
synthetic.

## External result

At the frozen threshold, the candidate detects 15 of 243 external English
scam-call transcripts (`6.17%` recall). The previous served artifact detected 3
of 243 (`1.23%`). This is a five-fold relative gain but an unacceptable absolute
result. External score Brier loss is `0.9703`, demonstrating that validation
calibration does not transfer to this source.

## Promotion decision

**Blocked.** The candidate is a better research default because it improves
held-out Hinglish ranking, seed-threshold stability, Banking77 test FPR, and
external scam-call recall. It is not safe for automatic caller engagement
because it still misses 93.8% of the only real-scammer-speech external source.

No LLM produced a feature, score, label, calibration target, or threshold.
