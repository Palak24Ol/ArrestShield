"""Run the local ArrestShield ML inference API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn  # noqa: E402

from arrestshield.api import create_default_app  # noqa: E402


def main() -> int:
    config = json.loads(
        (PROJECT_ROOT / "configs/deployment/api.json").read_text(encoding="utf-8")
    )
    service = config["service"]
    uvicorn.run(
        create_default_app(),
        host=os.environ.get("ARRESTSHIELD_HOST", str(service["host"])),
        port=int(os.environ.get("ARRESTSHIELD_PORT", service["port"])),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
