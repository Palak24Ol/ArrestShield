# Multilingual multi-task transformer

## Role and status

The transformer is a trained ML detector with four heads:

1. binary scam probability;
2. single-label scam type;
3. multi-label manipulation/operational tactics;
4. single-label current scam stage.

The current local backbone is `distilbert-base-multilingual-cased`. It is a CPU feasibility harness because the preferred IndicBERT checkpoint is gated/unavailable and the registered HingRoBERTa-Mixed, MuRIL, and XLM-R comparison requires GPU compute. A DistilBERT result must not be generalized into a claim that the registered candidates are better or worse.

No LLM supplies an input feature, target, prediction, threshold, checkpoint decision, or final scam label.

## Causal examples

Each conversation is formatted with explicit speaker roles. Positive training conversations generate 25%, 50%, and 100% prefixes; negatives generate 50% and 100% prefixes. Every prefix recomputes tactic and stage targets only from turns visible in that prefix. A prefix never inherits a payment-stage or tactic label from a future turn.

Unknown tactic labels are not treated as confirmed absence. In a tactic-annotated positive conversation, observed labels have target 1.0 and unobserved labels are target 0.0 with the fixed 0.25 supervision weight. Unannotated positive conversations remain fully masked for tactics/stages. Confirmed non-scam examples supply ordinary all-zero tactic supervision and `none_unknown` stage supervision.

Source-balanced sampling keeps every eligible positive, caps the negative-to-positive ratio at 2.5, and first requests negatives from mixed-label sources before distributing remaining capacity across negative sources. Silver examples carry the declared provenance weight.

## Context encoding

The maximum length is 256 tokens. Long conversations use deterministic head-tail selection with a boundary separator, preserving both the opening authority/pretext and the latest payment/credential evidence. Right truncation is forbidden for the primary run because it systematically removes late scam stages.

Encoded examples are stored once. Training uses deterministic sortish batching: examples are globally shuffled, locally length-sorted inside random mega-batches, split into batches, then batch order is shuffled. The collator removes columns that are padding for every row. This reduces padding computation while preserving every example, the 256-token ceiling, randomization, and resumability.

## Model and losses

The embeddings and first five DistilBERT blocks are frozen. The final transformer block and all four classification heads are trainable. Loss weights are:

| Task | Loss | Weight |
|---|---|---:|
| Binary | weighted binary cross-entropy | 1.0 |
| Scam type | weighted cross-entropy | 0.4 |
| Tactics | masked weighted binary cross-entropy | 0.3 |
| Stage | masked weighted cross-entropy | 0.2 |

Training uses AdamW, separate backbone/head learning rates, gradient accumulation, clipping, linear warm-up/decay, three epochs maximum, validation-only checkpoint selection, seeds 17/42/93, and early stopping. Exact values live in `configs/model/multitask_transformer.json`.

## Selection and evaluation

For every seed/epoch, the threshold is selected only on the validation primary view: all positives plus hard negatives. It must satisfy FPR at most 5%, then maximize recall. The checkpoint key is recall, macro-F1, then the lower eligible threshold. Test evaluation occurs only after selection and includes binary metrics, auxiliary-task metrics, and stable scammer-turn latency with hysteresis.

Tactic metrics cover observed positives versus confirmed non-scam negatives and are explicitly not presented as fully supervised tactic classification. The current silver tactic/stage targets cannot support a production psychological-analysis claim.

## Resumable CPU training

The trainer writes `artifacts/models/multitask_transformer_v1/training_checkpoint.pt` atomically every ten optimizer updates and at epoch/seed boundaries. It contains:

- only trainable model tensors (not the pinned frozen backbone);
- optimizer and scheduler state;
- exact Python, NumPy, PyTorch CPU, and CUDA RNG state when applicable;
- active seed, epoch, next batch, loss, patience, and best checkpoint;
- completed seed reports and deployment-export state;
- a signature over config, backbone, dataset counts, seeds, epochs, length, batch, accumulation, and batching policy.

On restart the signature must match. Mid-epoch data order and dropout resume from the recorded point. `--no-resume` deletes the local checkpoint only when an intentional fresh run is requested. The checkpoint is deleted automatically after final export and reporting succeed.

## Export contract

The deployment seed exports:

- `backbone/`: selected fine-tuned backbone weights and config;
- `tokenizer/`: tokenizer assets;
- `multitask_heads.pt`: all four head state dictionaries;
- `manifest.json`: labels, thresholds, encoding, loss, provenance policy, seed, selection role, and `llm_used_for_detection: false`.

The trainer reloads this export before final validation/test scoring. `MultiTaskPredictor` independently reconstructs the same export for the API. The final artifact verifier checks manifest and head hashes and the LLM boundary.

## Run

```powershell
.venv\Scripts\python.exe scripts\train_multitask_transformer.py
```

Generated weights and recovery checkpoints are local and Git-ignored. Only compact configuration, metrics, run metadata, artifact hashes, and candid limitations are committed.
