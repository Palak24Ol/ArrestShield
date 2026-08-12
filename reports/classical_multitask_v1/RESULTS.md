# Classical multilingual multi-task results

## Outcome

The completed CPU-feasible multi-task run reuses the training-split TF-IDF vocabulary and 128-dimensional SVD representation from the model ladder. It trains XGBoost auxiliary heads for scam type, current stage, and every manipulation tactic with at least 20 positive causal training windows. The binary score remains the selected class-weighted SGD detector; the API scam decision remains the optional trained XGBoost risk-fusion score. Auxiliary heads never replace that decision.

Training used 5,803 source-balanced conversations expanded into 11,279 causal prefix windows, plus 7,819 validation and 7,761 supporting test conversations. Seeds 17, 42, and 93 were trained. Thresholds and deployment seed 42 were fixed using validation only; test data was not used for selection.

| Validation metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Scam-type macro-F1 across all nine manifest classes | 0.8531 | 0.0642 |
| Stage macro-F1 across all seven manifest classes | 0.8391 | 0.0144 |
| Mean F1 across nine supported tactic heads | 0.7065 | 0.0064 |

For deployment seed 42, supporting-test macro-F1 was 0.8026 for scam type and 0.8335 for stage. Mean supporting-test F1 across supported tactics was 0.6971. Stronger tactic heads included authority impersonation (0.9398 test F1), financial demand (0.8462), credential request (0.9000), fear/intimidation (0.8000), and surveillance/control (0.7200). Secrecy instruction (0.3415) and isolation instruction (0.4906) were weak and must not be presented as reliable psychological analysis.

## Unsupported labels

The current corpus contains zero positive causal training windows for `phantom_riches`, `liking`, `pretext_trust`, `reciprocity`, `consistency_commitment`, and `social_proof`. The artifact does not train fake always-negative heads for these labels. Inference returns `available: false`, a null score, and `reason: no_positive_training_supervision`.

## Artifact

The seed-42 auxiliary head bundle is 2,013,527 bytes (1.92 MiB) with SHA-256 `e99795422b9ff4a3a353c77cd3cfeda121b2c458d28ecdd0555aecc19ff366f2`. It references, rather than duplicates, the train-only TF-IDF/SVD representation and selected binary detector. The manifest SHA-256 is `17ab3c7d031a817d4f052a69e5b1ad3498f489301754e25f562a5a381fae4984`.

## Interpretation

These are silver/synthetic corpus results, not a production or real-world accuracy claim. Every positive label is silver, 71% of positive supervision is synthetic, several rare classes have single-digit validation support, and the strict unseen-source binary and risk-fusion audits fail the 5% FPR gate. The independent 150-conversation human-gold set remains uncollected. The checked-in policy therefore remains `research_only_not_promoted`, and honeypot handoff remains disabled. No LLM supplied labels, features, thresholds, predictions, or decisions.
