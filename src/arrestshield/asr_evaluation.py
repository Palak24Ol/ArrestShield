"""Leakage-safe evaluation of ASR backends through the frozen ML detector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np

from .asr import character_error_rate, word_error_rate
from .evaluation import binary_metrics


@dataclass(frozen=True)
class AudioValidationRecord:
    record_id: str
    conversation_id: str
    audio_path: Path
    reference_text: str
    label: int
    language: str
    source_group: str
    rights_basis: str
    pii_redacted: bool


def _require_text(payload: Mapping[str, Any], field: str, line_number: int) -> str:
    value = str(payload.get(field) or "").strip()
    if not value:
        raise ValueError(f"Line {line_number}: {field} must be non-empty")
    return value


def load_audio_validation_manifest(
    manifest_path: Path,
    project_root: Path,
) -> list[AudioValidationRecord]:
    """Load a selection-only audio manifest and reject unsafe or invalid rows."""
    project_root = project_root.resolve()
    seen_ids: set[str] = set()
    records: list[AudioValidationRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if payload.get("split") != "audio_validation":
                raise ValueError(f"Line {line_number}: split must be audio_validation")
            record_id = _require_text(payload, "record_id", line_number)
            if record_id in seen_ids:
                raise ValueError(f"Line {line_number}: duplicate record_id {record_id}")
            seen_ids.add(record_id)
            label = payload.get("label")
            if label not in (0, 1):
                raise ValueError(f"Line {line_number}: label must be 0 or 1")
            relative_audio = Path(_require_text(payload, "audio_path", line_number))
            if relative_audio.is_absolute():
                raise ValueError(f"Line {line_number}: audio_path must be project-relative")
            audio_path = (project_root / relative_audio).resolve()
            try:
                audio_path.relative_to(project_root)
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number}: audio_path escapes the project root"
                ) from error
            if payload.get("pii_redacted") is not True:
                raise ValueError(f"Line {line_number}: pii_redacted must be true")
            records.append(
                AudioValidationRecord(
                    record_id=record_id,
                    conversation_id=_require_text(payload, "conversation_id", line_number),
                    audio_path=audio_path,
                    reference_text=_require_text(payload, "reference_text", line_number),
                    label=int(label),
                    language=_require_text(payload, "language", line_number),
                    source_group=_require_text(payload, "source_group", line_number),
                    rights_basis=_require_text(payload, "rights_basis", line_number),
                    pii_redacted=True,
                )
            )
    if not records:
        raise ValueError("Audio validation manifest is empty")
    if {record.label for record in records} != {0, 1}:
        raise ValueError("Audio validation manifest must contain both labels")
    return records


def frozen_detector_scores(bundle: Mapping[str, Any], texts: Sequence[str]) -> np.ndarray:
    """Score text with the saved deterministic detector bundle (never an LLM)."""
    if bundle.get("llm_used_for_detection") is not False:
        raise ValueError("Detector bundle does not explicitly prohibit LLM detection")
    matrix = bundle["feature_union"].transform(list(texts))
    if bundle.get("svd") is not None:
        matrix = bundle["svd"].transform(matrix).astype(np.float32)
    model = bundle["model"]
    probabilities = model.predict_proba(matrix)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Detector model has no positive class 1")
    scores = np.asarray(probabilities[:, classes.index(1)], dtype=np.float64)
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        scores = np.asarray(calibrator.predict(scores), dtype=np.float64)
    return scores


def evaluate_backend_outputs(
    records: Sequence[AudioValidationRecord],
    hypotheses: Sequence[str],
    runtimes: Sequence[float],
    detector_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    if not (len(records) == len(hypotheses) == len(runtimes)):
        raise ValueError("records, hypotheses and runtimes must have equal length")
    labels = [record.label for record in records]
    references = [record.reference_text for record in records]
    reference_scores = frozen_detector_scores(detector_bundle, references)
    hypothesis_scores = frozen_detector_scores(detector_bundle, hypotheses)
    threshold = float(detector_bundle["threshold"])
    wers = [word_error_rate(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    cers = [character_error_rate(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    score_deltas = hypothesis_scores - reference_scores
    reference_decisions = reference_scores >= threshold
    hypothesis_decisions = hypothesis_scores >= threshold
    return {
        "examples": len(records),
        "languages": sorted({record.language for record in records}),
        "source_groups": sorted({record.source_group for record in records}),
        "word_error_rate": float(mean(wers)),
        "character_error_rate": float(mean(cers)),
        "median_runtime_seconds": float(median(float(value) for value in runtimes)),
        "total_runtime_seconds": float(sum(float(value) for value in runtimes)),
        "reference_transcript_detection": binary_metrics(labels, reference_scores, threshold),
        "asr_transcript_detection": binary_metrics(labels, hypothesis_scores, threshold),
        "mean_absolute_score_delta_from_reference": float(np.abs(score_deltas).mean()),
        "mean_signed_score_delta_from_reference": float(score_deltas.mean()),
        "decision_flip_rate_from_reference": float(
            np.mean(reference_decisions != hypothesis_decisions)
        ),
        "per_record": [
            {
                **asdict(record),
                "audio_path": str(record.audio_path),
                "hypothesis": hypothesis,
                "wer": float(wer),
                "cer": float(cer),
                "runtime_seconds": float(runtime),
                "reference_score": float(reference_score),
                "asr_score": float(hypothesis_score),
                "decision_changed": bool(reference_decision != hypothesis_decision),
            }
            for record, hypothesis, wer, cer, runtime, reference_score, hypothesis_score,
            reference_decision, hypothesis_decision in zip(
                records,
                hypotheses,
                wers,
                cers,
                runtimes,
                reference_scores,
                hypothesis_scores,
                reference_decisions,
                hypothesis_decisions,
            )
        ],
    }


def select_asr_backend(
    backend_metrics: Mapping[str, Mapping[str, Any]],
    maximum_fpr: float,
) -> dict[str, Any]:
    """Apply the pre-declared constrained ordering to validation metrics only."""
    eligible = [
        (name, metrics)
        for name, metrics in backend_metrics.items()
        if float(metrics["asr_transcript_detection"]["false_positive_rate"])
        <= maximum_fpr + 1e-12
    ]
    if not eligible:
        return {
            "selected_backend": None,
            "gate_passed": False,
            "reason": f"No backend met downstream false-positive rate <= {maximum_fpr}",
        }
    name, metrics = max(
        eligible,
        key=lambda item: (
            float(item[1]["asr_transcript_detection"]["recall"]),
            float(item[1]["asr_transcript_detection"]["macro_f1"]),
            -float(item[1]["median_runtime_seconds"]),
            -float(item[1]["word_error_rate"]),
            item[0],
        ),
    )
    return {
        "selected_backend": name,
        "gate_passed": True,
        "downstream_false_positive_rate": float(
            metrics["asr_transcript_detection"]["false_positive_rate"]
        ),
        "downstream_recall": float(metrics["asr_transcript_detection"]["recall"]),
        "selection_split": "audio_validation_only",
    }
