"""Verify local data/model hashes and the detector/honeypot policy boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import write_json  # noqa: E402
from arrestshield.verification import verify_local_artifacts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete-transformer", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/verification_v1/verification.json",
    )
    args = parser.parse_args()
    report = verify_local_artifacts(
        PROJECT_ROOT,
        require_transformer=not args.allow_incomplete_transformer,
    )
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, report)
    print(json.dumps(report["counts"], indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
