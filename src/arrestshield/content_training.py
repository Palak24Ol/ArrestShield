"""Data helpers for the simple transcript-content scam detector."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import ConversationExample
from .external_evaluation import ExternalTextRecord


def partition_external_calls(
    records: Sequence[ExternalTextRecord],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[ExternalTextRecord]]:
    """Assign whole source URLs to deterministic adaptation splits."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")

    partitions: dict[str, list[ExternalTextRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    group_splits: dict[str, str] = {}
    for record in records:
        group = record.source_url or record.conversation_id
        digest = hashlib.sha256(f"arrestshield-content-v1|{group}".encode()).digest()
        rank = int.from_bytes(digest, "big") / float(1 << (8 * len(digest)))
        if rank < train_fraction:
            split = "train"
        elif rank < train_fraction + validation_fraction:
            split = "validation"
        else:
            split = "test"
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise AssertionError("Source group crossed adaptation splits")
        partitions[split].append(record)
    if any(not rows for rows in partitions.values()):
        raise ValueError("External-call partition produced an empty split")
    return partitions


def external_as_examples(
    records: Sequence[ExternalTextRecord], split: str
) -> list[ConversationExample]:
    return [
        ConversationExample(
            conversation_id=f"content-adaptation::{record.conversation_id}",
            text=record.text,
            label=record.label,
            scam_type="external_scam_call",
            split=split,
            source="youtube_scam_call_content_adaptation",
            languages=(record.language,),
            provenance="source_silver",
            turn_texts=(record.text,),
            sample_weight=1.0,
        )
        for record in records
    ]


def source_label_balanced_weights(
    examples: Sequence[ConversationExample],
) -> np.ndarray:
    """Equalize source/label cells, then equalize the two binary classes."""
    if not examples:
        raise ValueError("At least one example is required")
    labels = {example.label for example in examples}
    if labels != {0, 1}:
        raise ValueError("Both labels are required for balanced training weights")
    counts = Counter((example.source, example.label) for example in examples)
    weights = np.asarray(
        [example.sample_weight / counts[(example.source, example.label)] for example in examples],
        dtype=np.float64,
    )
    for label in (0, 1):
        indices = np.asarray([example.label == label for example in examples])
        weights[indices] *= 0.5 / float(np.sum(weights[indices]))
    weights *= len(weights) / float(np.sum(weights))
    return weights


def split_counts(
    partitions: Mapping[str, Sequence[ExternalTextRecord]],
) -> dict[str, int]:
    return {split: len(records) for split, records in partitions.items()}


def load_project_content_examples(path: Path) -> dict[str, list[ConversationExample]]:
    """Load the small, mixed-label hard-example set used by the simple model."""
    by_split: dict[str, list[ConversationExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("id") or "").strip()
            text = " ".join(str(row.get("text") or "").split())
            split = str(row.get("split") or "").strip()
            label = row.get("label")
            if not record_id or record_id in seen:
                raise ValueError(f"Line {line_number}: id must be non-empty and unique")
            if not text or label not in (0, 1) or split not in by_split:
                raise ValueError(f"Line {line_number}: invalid text, label, or split")
            seen.add(record_id)
            by_split[split].append(
                ConversationExample(
                    conversation_id=f"project-content::{record_id}",
                    text=f"[ROLE=caller] {text}",
                    label=int(label),
                    scam_type="project_authored_scam" if label else "non_scam",
                    split=split,
                    source="project_authored_content_examples",
                    languages=(str(row.get("language") or "unknown"),),
                    provenance="project_authored_silver",
                    turn_texts=(f"[ROLE=caller] {text}",),
                    sample_weight=float(row.get("weight", 1.0)),
                )
            )
    if not by_split["train"] or not by_split["validation"]:
        raise ValueError("Project examples require train and validation rows")
    return by_split
