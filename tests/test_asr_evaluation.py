import json
from pathlib import Path

import pytest

from arrestshield.asr_evaluation import (
    load_audio_validation_manifest,
    select_asr_backend,
)


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def valid_row(record_id: str, label: int) -> dict:
    return {
        "record_id": record_id,
        "conversation_id": f"conversation-{record_id}",
        "audio_path": f"data/audio_validation/raw/{record_id}.wav",
        "reference_text": "reference transcript",
        "label": label,
        "language": "hinglish",
        "source_group": "consented_roleplay",
        "rights_basis": "written participant consent",
        "pii_redacted": True,
        "split": "audio_validation",
    }


def test_manifest_requires_both_classes_and_stays_under_project_root(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [valid_row("positive", 1), valid_row("negative", 0)])
    records = load_audio_validation_manifest(manifest, tmp_path)
    assert [record.label for record in records] == [1, 0]

    escaping = valid_row("escape", 1)
    escaping["audio_path"] = "../outside.wav"
    write_manifest(manifest, [escaping, valid_row("negative", 0)])
    with pytest.raises(ValueError, match="escapes the project root"):
        load_audio_validation_manifest(manifest, tmp_path)


def test_manifest_rejects_non_redacted_and_single_class(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    row = valid_row("unsafe", 1)
    row["pii_redacted"] = False
    write_manifest(manifest, [row, valid_row("negative", 0)])
    with pytest.raises(ValueError, match="pii_redacted"):
        load_audio_validation_manifest(manifest, tmp_path)

    write_manifest(manifest, [valid_row("only", 1)])
    with pytest.raises(ValueError, match="both labels"):
        load_audio_validation_manifest(manifest, tmp_path)


def metrics(fpr: float, recall: float, macro_f1: float, runtime: float, wer: float) -> dict:
    return {
        "asr_transcript_detection": {
            "false_positive_rate": fpr,
            "recall": recall,
            "macro_f1": macro_f1,
        },
        "median_runtime_seconds": runtime,
        "word_error_rate": wer,
    }


def test_backend_selection_applies_fpr_gate_before_recall() -> None:
    selection = select_asr_backend(
        {
            "high_recall_but_unsafe": metrics(0.06, 1.0, 0.95, 1.0, 0.1),
            "eligible": metrics(0.05, 0.8, 0.75, 2.0, 0.2),
            "eligible_lower_recall": metrics(0.01, 0.7, 0.90, 0.5, 0.1),
        },
        maximum_fpr=0.05,
    )
    assert selection["selected_backend"] == "eligible"
    assert selection["gate_passed"] is True

    failed = select_asr_backend(
        {"unsafe": metrics(0.051, 1.0, 1.0, 1.0, 0.0)}, maximum_fpr=0.05
    )
    assert failed["selected_backend"] is None
    assert failed["gate_passed"] is False
