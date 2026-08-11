# Whisper ASR smoke result

The local `openai/whisper-tiny` multilingual checkpoint successfully transcribed OpenAI Whisper's 11-second public JFK test clip on CPU. The normalized transcript had 0.00 word error rate and took 26.53 seconds.

This is a functionality smoke test only. It does not validate Hindi/Hinglish, telephone audio, scam calls, hard negatives, downstream detection, or backend superiority. Whisper-tiny is therefore not promoted as the chosen production backend. Promotion remains blocked until the consented/licensed audio-validation manifest contains both labels and the required language/source coverage, and the backend passes the frozen detector's downstream 5% false-positive-rate gate.

The checkpoint itself is local and Git-ignored. `MODEL_MANIFEST.json` records its exact file hashes. Neither ASR nor scam detection invokes an LLM.
