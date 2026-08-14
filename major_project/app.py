"""Run the simple ArrestShield flow: transcript -> ML decision -> LLM honeypot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from detector import SimpleScamDetector
from honeypot import interactive_honeypot


PROJECT_DIR = Path(__file__).resolve().parent

SCAM_DEMO = (
    "Main CBI cyber crime division se bol raha hoon. Aapke Aadhaar par illegal "
    "parcel ka case hai. Kisi ko mat batana aur verification account me paisa bhejo."
)
LEGIT_DEMO = "Your courier will arrive tomorrow. No payment or OTP is required."


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Scammer transcript to classify")
    source.add_argument("--audio", type=Path, help="Call recording to transcribe and classify")
    source.add_argument("--demo", choices=("scam", "legit"), help="Run a built-in demo")
    parser.add_argument("--language", choices=("en", "hi"), help="Optional Whisper language hint")
    parser.add_argument("--engage", action="store_true", help="Start the LLM honeypot after SCAM")
    parser.add_argument("--json", action="store_true", help="Print machine-readable result")
    args = parser.parse_args()

    config = json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))
    load_env(PROJECT_DIR / ".env")
    if args.audio:
        from audio import transcribe_audio

        asr = transcribe_audio(
            args.audio,
            str((PROJECT_DIR / config["whisper_local_path"]).resolve()),
            args.language,
        )
        transcript = asr["text"]
        print(f"Transcript: {transcript}")
    elif args.demo:
        transcript = SCAM_DEMO if args.demo == "scam" else LEGIT_DEMO
    else:
        transcript = args.text

    detector = SimpleScamDetector(PROJECT_DIR / config["model_path"])
    result = detector.predict(transcript)
    if args.json:
        print(json.dumps({"transcript": transcript, **result.as_dict()}, indent=2))
    else:
        print(f"\nPrediction : {result.label}")
        print(f"Scam score : {result.scam_score:.4f}")
        print(f"Threshold  : {result.threshold:.4f}")
        if result.matched_patterns:
            print(f"Patterns   : {', '.join(result.matched_patterns)}")
        print("LLM decided the label: NO")

    if not result.is_scam:
        print("Honeypot   : BLOCKED (model predicted NOT_SCAM)")
        return 0
    print("Honeypot   : ELIGIBLE (model predicted SCAM)")
    if args.engage:
        interactive_honeypot(transcript, config["groq_model"])
    else:
        print("Use --engage to start the LLM fake-victim conversation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
