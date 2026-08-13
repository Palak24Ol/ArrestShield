"""Run the post-detection honeypot against a conversation.

The detector runs first and independently. Only its signed decision can open a
honeypot session, and the honeypot's replies never travel back into detection.

    python scripts/run_honeypot.py --demo --research-mode
    python scripts/run_honeypot.py --transcript call.json --research-mode
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.honeypot import (  # noqa: E402
    HandoffRejected,
    HoneypotError,
    build_handoff_event,
    start_session,
)

DEMO_TURNS = [
    {"speaker_role": "caller", "text": "Madam namaste, main CBI cyber crime division Delhi se Inspector Verma bol raha hoon."},
    {"speaker_role": "caller", "text": "Aapke Aadhaar card par ek parcel mila hai jisme drugs aur fake passport hai."},
    {"speaker_role": "victim", "text": "Kya? Mujhe kuch samajh nahi aa raha, main to kuch nahi bheji."},
    {"speaker_role": "caller", "text": "Madam ye digital arrest hai. Aap phone disconnect nahi karengi aur kisi ko batayengi nahi."},
    {"speaker_role": "caller", "text": "Verification ke liye aapko apne account ka paisa RBI ke safe account me transfer karna hoga."},
]


def load_environment_file(path: Path) -> None:
    """Load KEY=VALUE lines from a git-ignored .env without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/deployment/honeypot.json")
    parser.add_argument("--api-config", type=Path, default=PROJECT_ROOT / "configs/deployment/api.json")
    parser.add_argument("--transcript", type=Path, help="JSON file: [{speaker_role, text}, ...]")
    parser.add_argument("--audio", type=Path, help="Audio file (wav/flac/mp3/m4a/ogg) to transcribe first")
    parser.add_argument("--language-hint", type=str, default=None, help="ASR language hint, e.g. hi or en")
    parser.add_argument("--demo", action="store_true", help="Use the built-in digital-arrest sample")
    parser.add_argument("--research-mode", action="store_true", help="Operator override: engage despite an unpromoted detector")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--save", type=Path, help="Write the transcript JSON here")
    parser.add_argument("--dry-run", action="store_true", help="Show the gate decision and stop before any LLM call")
    return parser.parse_args()


def transcribe_audio(audio_path: Path, api_config_path: Path, language_hint: str | None):
    """Run local ASR. Whisper is a trained model, not an LLM: it produces the
    words, and the trained detector alone decides whether they are a scam."""
    from arrestshield.api import load_service_components

    _, transcriber, _ = load_service_components(api_config_path)
    if transcriber is None:
        raise SystemExit("ASR is disabled in configs/deployment/api.json")
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    print(f"== 0. Local ASR ({transcriber.backend_name}) ==")
    print(f"   file: {audio_path.name}")
    result = transcriber.transcribe(audio_path, language_hint=language_hint)
    print(f"   runtime: {result.runtime_seconds:.1f}s for {result.audio.duration_seconds:.1f}s of audio")
    print(f"   transcript: {result.text[:300]}")
    print(f"   llm_used: {result.to_dict().get('llm_used')}\n")
    if not result.text.strip():
        raise SystemExit("ASR produced an empty transcript; nothing to detect.")
    return [{"speaker_role": "caller", "text": result.text}]


def run_detector(turns, api_config_path: Path):
    """Import and run the detector here, at the edge, never inside the honeypot."""
    from arrestshield.inference import DetectorEngine, InferencePolicy

    api_config = json.loads(api_config_path.read_text(encoding="utf-8"))
    models = api_config["models"]
    policy_values = api_config["policy"]
    base_path = PROJECT_ROOT / models["base_detector_path"]
    if not base_path.exists():
        raise SystemExit(
            f"Base detector not found at {base_path}.\n"
            "Train it first:  python scripts/train_model_ladder.py"
        )
    fusion_path = PROJECT_ROOT / models["risk_fusion_path"]
    engine = DetectorEngine.from_paths(
        base_path,
        InferencePolicy(
            detector_status=str(policy_values["detector_status"]),
            allow_research_fusion=bool(policy_values["allow_research_fusion"]),
            enable_honeypot_handoff=bool(policy_values["enable_honeypot_handoff"]),
            maximum_turns=int(policy_values["maximum_turns"]),
            maximum_characters=int(policy_values["maximum_characters"]),
        ),
        fusion_path if fusion_path.exists() else None,
    )
    return engine.detect(turns)


def main() -> int:
    args = parse_args()
    load_environment_file(PROJECT_ROOT / ".env")

    if args.audio:
        turns = transcribe_audio(args.audio, args.api_config, args.language_hint)
    elif args.demo:
        turns = DEMO_TURNS
    elif args.transcript:
        turns = json.loads(args.transcript.read_text(encoding="utf-8"))
    else:
        raise SystemExit("Provide --audio, --demo, or --transcript")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.research_mode:
        config["policy"]["research_mode"] = True
    if args.max_turns:
        config["policy"]["maximum_turns"] = args.max_turns

    print("== 1. Trained detector (no LLM) ==")
    detection = run_detector(turns, args.api_config)
    print(f"   is_scam            : {detection['is_scam']}")
    print(f"   scam_score         : {detection['scam_score']:.4f} (threshold {detection['threshold']:.4f})")
    print(f"   decision_source    : {detection['decision_source']}")
    print(f"   production_eligible: {detection['production_eligible']}")
    print(f"   llm_used_for_detection: {detection['llm_used_for_detection']}")

    secret = os.environ.get("ARRESTSHIELD_HANDOFF_SECRET", "").strip()
    if not secret:
        secret = secrets.token_hex(32)
        print("\n   [no ARRESTSHIELD_HANDOFF_SECRET set; using an ephemeral per-run secret]")

    print("\n== 2. Signed handoff ==")
    event = build_handoff_event(detection, secret)
    print(f"   event_id : {event['event_id']}")
    print(f"   signature: {event['signature'][:32]}... (hmac-sha256)")

    print("\n== 3. Honeypot eligibility gate ==")
    try:
        session = start_session(event, config, secret)
    except HandoffRejected as error:
        print(f"   BLOCKED: {error}")
        print("\n   The honeypot refused to engage. This is the default and correct")
        print("   behaviour while the detector is research-only. Pass --research-mode")
        print("   to override deliberately for a demo.")
        return 0
    except HoneypotError as error:
        print(f"   ERROR: {error}")
        return 1

    print(f"   ALLOWED in mode: {session.mode}")
    print(f"   persona        : {session.identity.display_name}")
    print(f"   synthetic ids  : phone {session.identity.phone} | upi {session.identity.upi_like}")
    print(f"   turn budget    : {session.policy.maximum_turns}")
    if args.dry_run:
        print("\n   --dry-run set; stopping before any LLM call.")
        return 0

    print("\n== 4. Engagement (type 'quit' to stop) ==")
    caller_lines = [turn["text"] for turn in turns if turn.get("speaker_role") != "victim"]
    queued = list(caller_lines[-1:])

    while True:
        if queued:
            caller_text = queued.pop(0)
            print(f"\n   caller  > {caller_text}")
        else:
            try:
                caller_text = input("\n   caller  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if caller_text.lower() in {"quit", "exit"}:
                break
            if not caller_text:
                continue
        try:
            result = session.respond(caller_text)
        except HoneypotError as error:
            print(f"   ERROR: {error}")
            break
        if not result["engaged"]:
            print(f"   [session ended: {result['stop_reason']}]")
            break
        print(f"   persona > {result['reply']}")
        if result.get("redactions"):
            print(f"   [redacted: {', '.join(result['redactions'])}]")

    transcript = session.transcript()
    transcript["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    print("\n== 5. Transcript ==")
    print(f"   turns_used                    : {transcript['turns_used']}")
    print(f"   excluded_from_detector_training: {transcript['excluded_from_detector_training']}")
    print(f"   llm_used_for_detection        : {transcript['llm_used_for_detection']}")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   saved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
