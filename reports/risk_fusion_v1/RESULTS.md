# XGBoost risk-fusion results

## Outcome

The full `risk_fusion_v1` run trained three XGBoost seeds over 30 fixed features. Base-detector scores for fusion training were generated through five-fold out-of-fold refits of both TF-IDF and the classifier; no fusion training row received an in-sample base score.

At the deployment seed 42 validation threshold of `0.9478815794`, the hard-negative validation view had 100% recall and 0.0535% FPR. The supporting test hard-negative view had 98.43% recall, 0.0578% FPR, and macro-F1 0.9944. Stable detection covered 318 of 319 positive test conversations with median one scammer turn. Across three seeds, validation recall was `1.0000 ± 0.0000`, validation FPR was `0.00107 ± 0.00093`, and supporting test recall was `0.98851 ± 0.00724`.

The saved artifact is `research_only_not_promoted`. Its SHA-256 is `d99e91170f20da7e89ae85e8f0fb35afd9965d4dde9fdc7e8f6e1ed0bb0e6461` and its base detector hash is `345d9ea58be75be489fe6b695d33c8e73734a0c7f180e22ef2d41e6d4b259796`.

## Interpretation

These near-perfect mixed-source results are not a production claim. The current positive pool is entirely silver, 71% of positive supervision is synthetic, and source/style shortcuts are already known to inflate ordinary splits. The frozen base detector itself scored perfectly on this validation view, so risk fusion did not demonstrate meaningful generalization improvement there. The strict leave-one-mixed-source-out audit and the future frozen human-gold set remain the decisive evidence.

## Strict leave-one-source-out audit

The strict audit refit the TF-IDF representation, base classifier, out-of-fold base scores, threshold-selection view, and XGBoost fusion model after excluding each mixed-label source. It used seeds 17, 42, and 93 and required every source/seed run—not merely the average—to remain at or below 5% FPR.

| Held-out source | Mean recall | Mean FPR | Mean macro-F1 | Gate |
| --- | ---: | ---: | ---: | --- |
| `indian_cyber_scam_phonecall_hinglish` | 0.9651 | 0.8095 | 0.5894 | Fail |
| `indian_multilingual_scam_messages` | 0.0000 | 0.0000 | 0.3846 | Fail: zero recall; only 8 test rows |
| `synthetic_multi_agent_scam_conversation` | 0.9216 | 0.2604 | 0.8311 | Fail |
| `synthetic_scam_dialogue` | 0.8411 | 0.0385 | 0.9013 | Pass |

Across sources, recall was `0.68196 ± 0.00978`, FPR was `0.27710 ± 0.00000`, macro-F1 was `0.67659 ± 0.00498`, and the worst source FPR was `0.80952`. The strict gate failed. These results show that risk fusion did not solve the corpus source/style shortcut problem and must remain a research artifact. The full machine-readable result is `loso_metrics.json`.

Promotion failed because the independently annotated frozen human-gold set is not collected. The deployment policy therefore blocks honeypot handoff even when this model crosses its threshold. No LLM produced a feature, label, threshold, or decision.
