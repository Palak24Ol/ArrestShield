"""Frozen external-text evaluation that cannot tune or train the detector."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .asr_evaluation import frozen_detector_scores
from .evaluation import binary_metrics


@dataclass(frozen=True)
class ExternalTextRecord:
    record_id: str
    conversation_id: str
    text: str
    label: int
    language: str
    source_group: str
    rights_basis: str
    pii_redacted: bool
    source_url: str = ""


def load_external_text_manifest(path: Path) -> list[ExternalTextRecord]:
    records: list[ExternalTextRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("split") != "external_evaluation":
                raise ValueError(f"Line {line_number}: split must be external_evaluation")
            record_id = str(payload.get("record_id") or "").strip()
            text = str(payload.get("text") or "").strip()
            if not record_id or not text:
                raise ValueError(f"Line {line_number}: record_id and text are required")
            if record_id in seen:
                raise ValueError(f"Line {line_number}: duplicate record_id {record_id}")
            seen.add(record_id)
            label = payload.get("label")
            if label not in (0, 1):
                raise ValueError(f"Line {line_number}: label must be 0 or 1")
            if payload.get("pii_redacted") is not True:
                raise ValueError(f"Line {line_number}: pii_redacted must be true")
            fields = {
                name: str(payload.get(name) or "").strip()
                for name in ("conversation_id", "language", "source_group", "rights_basis")
            }
            if not all(fields.values()):
                raise ValueError(f"Line {line_number}: provenance fields must be non-empty")
            records.append(
                ExternalTextRecord(
                    record_id=record_id,
                    conversation_id=fields["conversation_id"],
                    text=text,
                    label=int(label),
                    language=fields["language"],
                    source_group=fields["source_group"],
                    rights_basis=fields["rights_basis"],
                    pii_redacted=True,
                    source_url=str(payload.get("source_url") or "").strip(),
                )
            )
    if not records:
        raise ValueError("External evaluation manifest is empty")
    return records


def evaluate_external_text(
    records: Sequence[ExternalTextRecord], detector_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    scores = frozen_detector_scores(detector_bundle, [record.text for record in records])
    labels = [record.label for record in records]
    threshold = float(detector_bundle["threshold"])
    by_source: dict[str, Any] = {}
    for source in sorted({record.source_group for record in records}):
        indices = [index for index, record in enumerate(records) if record.source_group == source]
        by_source[source] = binary_metrics(
            [labels[index] for index in indices], scores[indices], threshold
        )
    return {
        "protocol": "external_evaluation_only_no_training_or_threshold_selection",
        "examples": len(records),
        "languages": sorted({record.language for record in records}),
        "source_groups": sorted({record.source_group for record in records}),
        "metrics": binary_metrics(labels, scores, threshold),
        "by_source": by_source,
        "score_distribution": {
            "minimum": float(np.min(scores)),
            "p25": float(np.quantile(scores, 0.25)),
            "median": float(np.median(scores)),
            "p75": float(np.quantile(scores, 0.75)),
            "maximum": float(np.max(scores)),
        },
        "threshold_was_frozen": True,
        "llm_used_for_detection": False,
    }
