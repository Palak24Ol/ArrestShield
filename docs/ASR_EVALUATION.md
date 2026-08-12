# ASR integration and evaluation

## Boundary

The ASR component converts audio to text. It does not decide whether a call is a scam. The frozen trained ML detector consumes the transcript and produces the scam score. No LLM participates in transcription, threshold selection, or scam detection.

## Registered backends

- `openai/whisper-tiny` is the installed CPU-feasibility and fallback candidate. It is multilingual and Apache-2.0 licensed.
- AI4Bharat Vistaar IndicWhisper Hindi is registered as a GPU candidate. The official checkpoint is a Whisper-medium derivative and is intentionally not installed on this CPU-only laptop. Supplying the compatible local checkpoint enables the same adapter.

Registration is not evidence that one backend is better. Backend promotion requires the selection-only audio validation corpus.

The adapter uses Whisper's native long-form segmentation rather than Transformers 4.48.3's external chunk iterator. This preserves the attention mask required when the padding and end-of-sequence token are identical. Task and optional language are supplied per request, and redundant checkpoint `forced_decoder_ids` are cleared to avoid conflicting decoder prompts.

## Audio validation corpus

Each JSONL record must conform to `configs/data/audio_validation_record.schema.json`. Audio must be consented or appropriately licensed, manually transcribed, PII-redacted, and assigned to `audio_validation`. The manifest must contain both scam and non-scam calls and must represent English, Hindi, and Hinglish plus more than one source group. Project-relative paths are enforced to prevent accidental arbitrary file access.

Audio validation data is never the final frozen human test set. It exists only to choose the ASR backend. `data/audio_validation/COLLECTION_STATUS.json` truthfully records that this corpus is not yet collected; no synthetic transcript is presented as human evidence.

## Selection rule

The rule is fixed before data collection:

1. Keep the detector family, detector threshold, and all detector weights frozen.
2. Reject any ASR backend whose downstream hard-negative false-positive rate is above 5%.
3. Among eligible backends, maximize downstream scam recall.
4. Break ties by downstream macro-F1, lower median runtime, then lower WER.

WER and CER are reported, but WER-only selection is forbidden. This captures the possibility that different romanization or script choices have similar WER but materially different detector behavior.

## Run

Place audio beneath `data/audio_validation/raw`, create `data/audio_validation/manifest.jsonl`, then run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_asr.py
```

The evaluator writes per-backend downstream metrics and redacted transcripts under `reports/asr_validation_v1`. It records the frozen detector hash and explicitly records both LLM flags as false.
