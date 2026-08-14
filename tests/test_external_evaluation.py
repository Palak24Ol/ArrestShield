from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arrestshield.external_evaluation import (
    evaluate_external_text,
    load_external_text_manifest,
)


class FakeRepresentation:
    def transform(self, texts):
        return np.asarray([[0.9 if "urgent" in text else 0.1] for text in texts])


class FakeModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix):
        values = np.asarray(matrix)[:, 0]
        return np.column_stack([1.0 - values, values])


def row(record_id: str, label: int, text: str) -> dict:
    return {
        "record_id": record_id,
        "conversation_id": record_id,
        "text": text,
        "label": label,
        "language": "english",
        "source_group": "external_calls",
        "rights_basis": "CC0",
        "pii_redacted": True,
        "split": "external_evaluation",
    }


def test_external_manifest_is_evaluation_only_and_supports_single_class(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row("one", 1, "urgent transfer")) + "\n", encoding="utf-8")
    records = load_external_text_manifest(manifest)
    report = evaluate_external_text(
        records,
        {
            "feature_union": FakeRepresentation(),
            "svd": None,
            "model": FakeModel(),
            "threshold": 0.5,
            "llm_used_for_detection": False,
        },
    )
    assert report["metrics"]["recall"] == 1.0
    assert report["metrics"]["roc_auc"] is None
    assert report["threshold_was_frozen"] is True


def test_external_manifest_rejects_training_split_and_unredacted_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    bad = row("bad", 0, "normal")
    bad["split"] = "train"
    manifest.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="external_evaluation"):
        load_external_text_manifest(manifest)

    bad = row("bad", 0, "normal")
    bad["pii_redacted"] = False
    manifest.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pii_redacted"):
        load_external_text_manifest(manifest)
