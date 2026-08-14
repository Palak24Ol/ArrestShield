# ArrestShield Major Project

This folder is the clean professor-demo version of ArrestShield. It has one
simple job: transcribe a call, classify the transcript as `SCAM` or `NOT_SCAM`,
and start an LLM fake-victim honeypot only after the trained model predicts
`SCAM`.

## Simple flow

```text
Call audio or transcript
          |
          v
Whisper speech-to-text (audio only)
          |
          v
Binary ML content detector
     |              |
 NOT_SCAM          SCAM
     |              |
 Block        LLM honeypot eligible
```

The detector is word/character TF-IDF plus a linear SGD logistic classifier.
It reads the words and short text patterns used in the transcript. XGBoost,
multi-task labels, and the LLM are not part of the binary decision.

## Tested result

- Fixed English/Hinglish behavioral test: **18/20 = 90%**.
- Scam detection in that test: **10/10**.
- Legitimate rejection in that test: **8/10**.
- Canonical deployment-seed test accuracy is higher, but it is not presented as
  real-world accuracy because its dataset distribution is easier.

The two behavioral errors are preserved in
`reports/behavioral_evaluation.json`; the test set was not edited after seeing
the result.

## Quick demo

From `C:\ArrestShield\major_project`:

```powershell
..\.venv\Scripts\python.exe app.py --demo scam
..\.venv\Scripts\python.exe app.py --demo legit
..\.venv\Scripts\python.exe evaluate.py
```

Double-click `run_scam_demo.bat` or `run_legit_demo.bat` for the easiest demo.
For complete setup, audio, honeypot, testing, and professor-presentation steps,
read `USER_GUIDE.md`.

## Folder map

- `app.py` - one command for transcript/audio detection and honeypot routing.
- `detector.py` - trained binary ML inference only.
- `audio.py` - Whisper transcription only.
- `honeypot.py` - post-detection LLM fake victim.
- `evaluate.py` - fixed behavioral prediction test.
- `models/selected_detector.joblib` - trained 3 MB model.
- `data/` - transparent hard examples and untouched behavioral cases.
- `reports/` - evaluation output.
- `USER_GUIDE.md` - full non-confusing usage guide.

## Boundary guarantee

`SCAM`/`NOT_SCAM` comes only from the saved trained model and its fixed
threshold. Whisper only produces text. The LLM is called only after `SCAM` and
cannot change the label.
