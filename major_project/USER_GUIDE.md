# ArrestShield Major Project - User Guide

## 1. What this project does

ArrestShield accepts a scammer transcript directly or transcribes a saved call
recording with Whisper. A trained binary text model predicts `SCAM` or
`NOT_SCAM`. When and only when the result is `SCAM`, the program can start an
LLM-powered fake-victim honeypot that keeps the suspected scammer talking.

The LLM does not detect scams. This is the most important architecture rule.

## 2. Requirements

- Windows 10/11.
- Python 3.10.
- The existing `C:\ArrestShield\.venv` environment.
- FFmpeg on `PATH` only for audio input.
- A Groq API key only for live honeypot replies.

The trained detector is already included at
`major_project\models\selected_detector.joblib`; you do not need to train before
the demo.

## 3. Open the project

Open PowerShell and run:

```powershell
cd C:\ArrestShield\major_project
```

## 4. Fastest professor demo

### Scam example

```powershell
..\.venv\Scripts\python.exe app.py --demo scam
```

Expected result: `SCAM`, followed by `Honeypot: ELIGIBLE`.

### Legitimate example

```powershell
..\.venv\Scripts\python.exe app.py --demo legit
```

Expected result: `NOT_SCAM`, followed by `Honeypot: BLOCKED`.

You can also double-click `run_scam_demo.bat` and `run_legit_demo.bat`.

## 5. Test your own transcript

```powershell
..\.venv\Scripts\python.exe app.py --text "Main police se bol raha hoon, kisi ko mat batana aur safe account me paise bhejo"
```

The output shows:

- final binary prediction;
- scam score;
- fixed threshold;
- influential word patterns;
- confirmation that the LLM did not decide the label;
- whether the honeypot is eligible or blocked.

For JSON output, add `--json`.

## 6. Test a saved call recording

Supported input types are WAV, FLAC, MP3, M4A, and OGG.

```powershell
..\.venv\Scripts\python.exe app.py --audio C:\path\to\call.wav --language hi
```

Use `--language en` for English or omit the language option to let Whisper infer
it. The first ASR load can take time on a CPU laptop. Whisper only creates the
transcript; the binary model still makes the scam decision.

## 7. Enable the live LLM honeypot

1. Copy `.env.example` to `.env`.
2. Put your Groq key after `GROQ_API_KEY=`.
3. Run a scam input with `--engage`:

```powershell
..\.venv\Scripts\python.exe app.py --demo scam --engage
```

The fake victim responds, and you can type further scammer lines. Type `quit`
to stop. A `NOT_SCAM` result cannot start this interaction.

Never commit `.env`; it contains a secret key.

## 8. Run the prediction test

```powershell
..\.venv\Scripts\python.exe evaluate.py
```

Expected packaged result: **18/20 = 90%**. All ten scam examples should be
detected. The detailed result is written to
`reports\behavioral_evaluation.json` and deliberately preserves the two failed
legitimate cases.

Run the full repository tests from `C:\ArrestShield`:

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
.venv\Scripts\python.exe -m pytest major_project\tests\ -q
```

## 9. Retrain the simple detector

The normal demo does not require this. To rebuild from the cleaned local data:

```powershell
cd C:\ArrestShield
.venv\Scripts\python.exe scripts\train_simple_content_detector.py
Copy-Item artifacts\models\simple_content_detector_v1\selected_detector.joblib major_project\models\selected_detector.joblib -Force
cd major_project
..\.venv\Scripts\python.exe evaluate.py
```

Do not edit the behavioral test after checking its result. Doing so would turn
the test into training data and make the displayed accuracy misleading.

## 10. What to tell the professor

Use this short explanation:

> Whisper transcribes the call, then a trained word-and-character TF-IDF
> logistic classifier independently predicts scam or not scam. It scores 18/20
> on the fixed English/Hinglish suite and detects all ten scams. Only a scam
> prediction activates the LLM honeypot; the LLM never decides the label.

Do not claim 99% real-world accuracy. Higher internal dataset scores are useful
engineering checks but are easier than real unseen calls.

## 11. Troubleshooting

- Missing model: confirm `models\selected_detector.joblib` exists.
- Missing Groq key: detection still works; a key is required only for `--engage`.
- FFmpeg/decoder error: install FFmpeg and reopen PowerShell.
- Audio is slow: Whisper-tiny runs locally on CPU and may take longer than the
  recording duration.
- False alarm on a warning message: this is a known limitation because warnings
  repeat scam vocabulary such as OTP, arrest, and fraud.

## 12. Safety and scope

This is a local academic prototype. Use only recordings you are authorised to
process. Do not connect it to real telecom infrastructure without consent,
security review, rate limits, authentication, and legal approval. The honeypot
uses only fake identifiers and must never transfer money or expose personal data.
