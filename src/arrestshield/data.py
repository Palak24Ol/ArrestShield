"""Load canonical ArrestShield conversations without leaking split membership."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class ConversationExample:
    conversation_id: str
    text: str
    label: int
    scam_type: str
    split: str
    source: str
    languages: tuple[str, ...]
    provenance: str
    turn_texts: tuple[str, ...]
    sample_weight: float = 1.0


@dataclass(frozen=True)
class LoadSummary:
    retained: int
    excluded: Mapping[str, int]
    split_counts: Mapping[str, int]
    label_counts: Mapping[str, int]


def format_turn(speaker_role: str, text: str) -> str:
    """Add an explicit speaker marker while preserving multilingual text."""
    role = (speaker_role or "unknown").strip().lower().replace(" ", "_")
    clean_text = " ".join(str(text).split())
    return f"[ROLE={role}] {clean_text}"


def format_conversation(turns: Sequence[Mapping[str, object]]) -> tuple[str, tuple[str, ...]]:
    formatted = tuple(
        format_turn(
            str(turn.get("speaker_role", "unknown")),
            str(turn.get("normalized_text") or turn.get("raw_text") or ""),
        )
        for turn in turns
        if str(turn.get("normalized_text") or turn.get("raw_text") or "").strip()
    )
    return "\n".join(formatted), formatted


def load_split_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("conversation_to_split")
    if not isinstance(mapping, dict):
        raise ValueError(f"Missing conversation_to_split mapping in {path}")
    allowed = {"train", "validation", "test"}
    invalid = sorted({str(value) for value in mapping.values()} - allowed)
    if invalid:
        raise ValueError(f"Invalid split names in {path}: {invalid}")
    return {str(key): str(value) for key, value in mapping.items()}


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield record


def _hash_rank(conversation_id: str) -> str:
    return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()


def limit_by_stable_hash(
    examples: Sequence[ConversationExample], max_per_split: int | None
) -> list[ConversationExample]:
    if not max_per_split or max_per_split <= 0:
        return list(examples)
    retained: list[ConversationExample] = []
    for split in ("train", "validation", "test"):
        candidates = [example for example in examples if example.split == split]
        candidates.sort(key=lambda example: _hash_rank(example.conversation_id))
        retained.extend(candidates[:max_per_split])
    return retained


def load_examples(
    conversations_path: Path,
    split_manifest_path: Path,
    allowed_provenance: Iterable[str],
    provenance_weights: Mapping[str, float] | None = None,
    max_per_split: int | None = None,
) -> tuple[dict[str, list[ConversationExample]], LoadSummary]:
    """Load eligible labeled records and join splits by conversation ID.

    Split assignment comes only from the precomputed, group-safe manifest. The
    loader never creates a random row-level split.
    """
    split_map = load_split_map(split_manifest_path)
    allowed = set(allowed_provenance)
    weights = dict(provenance_weights or {})
    excluded: Counter[str] = Counter()
    examples: list[ConversationExample] = []

    for record in iter_jsonl(conversations_path):
        conversation_id = str(record.get("conversation_id", ""))
        split = split_map.get(conversation_id)
        quality = record.get("quality") or {}
        label_data = record.get("conversation_label") or {}
        provenance = str(label_data.get("provenance", "unlabeled"))
        label = label_data.get("is_scam")

        if split is None:
            excluded["missing_split"] += 1
            continue
        if not bool(quality.get("training_eligible", False)):
            excluded["not_training_eligible"] += 1
            continue
        if provenance not in allowed:
            excluded["disallowed_provenance"] += 1
            continue
        if label not in (0, 1):
            excluded["missing_binary_label"] += 1
            continue

        text, turn_texts = format_conversation(record.get("turns") or [])
        if not text:
            excluded["empty_text"] += 1
            continue

        source_data = record.get("source") or {}
        languages = tuple(sorted({str(item) for item in record.get("language_profile") or ["unknown"]}))
        examples.append(
            ConversationExample(
                conversation_id=conversation_id,
                text=text,
                label=int(label),
                scam_type=str(label_data.get("scam_type") or "unknown"),
                split=split,
                source=str(source_data.get("dataset_id") or "unknown"),
                languages=languages,
                provenance=provenance,
                turn_texts=turn_texts,
                sample_weight=float(weights.get(provenance, 1.0)),
            )
        )

    examples = limit_by_stable_hash(examples, max_per_split=max_per_split)
    by_split = {
        split: [example for example in examples if example.split == split]
        for split in ("train", "validation", "test")
    }
    if not by_split["train"] or not by_split["validation"] or not by_split["test"]:
        raise ValueError("Train, validation, and test must all contain eligible examples")
    train_labels = {example.label for example in by_split["train"]}
    if train_labels != {0, 1}:
        raise ValueError(f"Training data must contain both binary classes; found {train_labels}")

    summary = LoadSummary(
        retained=len(examples),
        excluded=dict(sorted(excluded.items())),
        split_counts={key: len(value) for key, value in by_split.items()},
        label_counts={
            f"{split}:{label}": sum(example.label == label for example in values)
            for split, values in by_split.items()
            for label in (0, 1)
        },
    )
    return by_split, summary


def texts(examples: Sequence[ConversationExample]) -> list[str]:
    return [example.text for example in examples]


def labels(examples: Sequence[ConversationExample]) -> list[int]:
    return [example.label for example in examples]


def sample_weights(examples: Sequence[ConversationExample]) -> list[float]:
    return [example.sample_weight for example in examples]
