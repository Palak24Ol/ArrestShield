"""Transcribe the frozen audio-validation manifest and score downstream detection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import sha256_file, write_json  # noqa: E402
from arrestshield.asr import WhisperASR  # noqa: E402
from arrestshield.asr_evaluation import (  # noqa: E402
    evaluate_backend_outputs,
    load_audio_validation_manifest,
    select_asr_backend,
)


LOGGER = logging.getLogger("arrestshield.evaluate_asr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/model/asr_backends.json"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/audio_validation/manifest.jsonl",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=PROJECT_ROOT / "artifacts/models/model_ladder_v1/selected_detector.joblib",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports/asr_validation_v1"
    )
    parser.add_argument("--backend", action="append", default=[])
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = read_json(args.config)
    records = load_audio_validation_manifest(args.manifest, PROJECT_ROOT)
    detector_bundle = joblib.load(args.detector)
    requested = set(args.backend)
    backend_names = [
        name
        for name, backend_config in config["backends"].items()
        if backend_config.get("enabled", False) and (not requested or name in requested)
    ]
    unknown = requested - set(config["backends"])
    if unknown:
        raise ValueError(f"Unknown ASR backend(s): {sorted(unknown)}")
    if not backend_names:
        raise ValueError("No enabled ASR backend was selected")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    transcript_rows: list[dict[str, Any]] = []
    for backend_name in backend_names:
        LOGGER.info("Evaluating %s on %d audio records", backend_name, len(records))
        transcriber = WhisperASR(
            backend_name,
            config["backends"][backend_name],
            config["audio"],
            PROJECT_ROOT,
        )
        hypotheses: list[str] = []
        runtimes: list[float] = []
        for record in records:
            language_hint = record.language if record.language in {"en", "hi"} else None
            result = transcriber.transcribe(record.audio_path, language_hint=language_hint)
            hypotheses.append(result.text)
            runtimes.append(result.runtime_seconds)
            transcript_rows.append(
                {
                    "record_id": record.record_id,
                    "backend": backend_name,
                    "text": result.text,
                    "runtime_seconds": result.runtime_seconds,
                    "llm_used": False,
                }
            )
        metrics[backend_name] = evaluate_backend_outputs(
            records, hypotheses, runtimes, detector_bundle
        )

    selection = select_asr_backend(
        metrics, maximum_fpr=float(config["primary_selection"]["maximum"])
    )
    report = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": config["selection_split"],
        "test_used_for_backend_selection": False,
        "detector_sha256": sha256_file(args.detector),
        "detector_family": detector_bundle["model_family"],
        "detector_seed": detector_bundle["seed"],
        "detector_threshold": detector_bundle["threshold"],
        "llm_used_for_transcription": False,
        "llm_used_for_detection": False,
        "selection": selection,
        "backends": metrics,
    }
    write_json(args.report_dir / "metrics.json", report)
    with (args.report_dir / "transcripts.jsonl").open("w", encoding="utf-8") as handle:
        for row in transcript_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    LOGGER.info("ASR selection: %s", selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
