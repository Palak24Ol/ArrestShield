"""Run the trained deterministic detector on one conversation text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.baseline import scam_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="One or more turns of conversation text")
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "artifacts/models/baseline_v1/model.joblib"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "artifacts/models/baseline_v1/metadata.json",
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    threshold = float(metadata["threshold"]["threshold"])
    score = float(scam_scores(model, [args.text])[0])
    result = {
        "schema_version": "1.0.0",
        "model_id": metadata["model_id"],
        "scam_score": score,
        "threshold": threshold,
        "is_scam": score >= threshold,
        "decision_source": "trained_ml_detector",
        "llm_used_for_detection": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
