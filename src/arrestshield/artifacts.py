"""Write reproducible model artifacts and machine-readable reports."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import joblib
import sklearn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_training_run(
    model: Any,
    artifact_dir: Path,
    report_dir: Path,
    config: Mapping[str, Any],
    data_summary: Mapping[str, Any],
    threshold: Mapping[str, Any],
    metrics: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "schema_version": "1.0.0",
        "model_id": config["model_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256_file(model_path),
        "model_bytes": model_path.stat().st_size,
        "git_revision": git_revision(project_root),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "decision_boundary": (
            "This trained detector decides scam risk. No LLM is called by this model "
            "or used to choose the scam label."
        ),
        "threshold": dict(threshold),
        "data_summary": dict(data_summary),
    }
    write_json(artifact_dir / "metadata.json", metadata)
    write_json(artifact_dir / "config.json", dict(config))
    write_json(report_dir / "metrics.json", dict(metrics))
    write_json(report_dir / "run_metadata.json", metadata)
    write_json(report_dir / "config.json", dict(config))
    return metadata
