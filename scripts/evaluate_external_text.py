"""Evaluate a frozen detector on an external manifest without tuning it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, write_json  # noqa: E402
from arrestshield.external_evaluation import (  # noqa: E402
    evaluate_external_text,
    load_external_text_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/interim/external_evaluation/youtube_scam_calls.jsonl",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=PROJECT_ROOT / "artifacts/models/mixed_source_candidate_v2/selected_detector.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/external_text_v1/metrics.json",
    )
    args = parser.parse_args()
    records = load_external_text_manifest(args.manifest)
    bundle = joblib.load(args.detector)
    report = {
        "schema_version": "1.0.0",
        "run_id": "arrestshield-external-text-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "manifest": str(args.manifest),
        "detector": str(args.detector),
        **evaluate_external_text(records, bundle),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
