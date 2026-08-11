"""Validation and freezing gates for the project-authored human test set."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


class FreezeValidationError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("Human frozen-test validation failed:\n- " + "\n- ".join(self.errors))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise FreezeValidationError([f"{path}:{line_number} is invalid JSON"]) from error
            if not isinstance(row, dict):
                raise FreezeValidationError([f"{path}:{line_number} must be an object"])
            rows.append(row)
    return rows


def normalized_transcript(record: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for turn in record.get("turns") or []:
        role = re.sub(r"\s+", "_", str(turn.get("speaker_role") or "unknown").strip().lower())
        body = unicodedata.normalize("NFKC", str(turn.get("text") or ""))
        body = " ".join(body.split()).casefold()
        if body:
            lines.append(f"{role}: {body}")
    return "\n".join(lines)


def transcript_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(normalized_transcript(record).encode("utf-8")).hexdigest()


def annotation_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    is_scam = int(row.get("is_scam", -1))
    scam_type = str(row.get("scam_type") or ("non_scam" if is_scam == 0 else "unknown"))
    tactics = tuple(sorted({str(value) for value in row.get("tactics") or []}))
    stage = str(row.get("stage") or "none_unknown")
    return is_scam, scam_type, tactics, stage


def _validate_intake(record: Mapping[str, Any], protocol: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    identifier = str(record.get("conversation_id") or "<missing-id>")
    collection = protocol["collection"]
    turns = record.get("turns") or []
    if not str(record.get("conversation_id") or "").strip():
        errors.append("intake record is missing conversation_id")
    if record.get("collection_channel") not in collection["allowed_channels"]:
        errors.append(f"{identifier}: disallowed collection_channel")
    if collection["require_consent"] and record.get("consent_confirmed") is not True:
        errors.append(f"{identifier}: consent is not confirmed")
    if collection["require_pii_redaction"] and record.get("pii_redacted") is not True:
        errors.append(f"{identifier}: PII redaction is not confirmed")
    if not collection["llm_generated_dialogue_allowed"] and record.get("llm_generated") is not False:
        errors.append(f"{identifier}: llm_generated must be explicitly false")
    if len(turns) < int(collection["minimum_turns"]):
        errors.append(f"{identifier}: fewer than {collection['minimum_turns']} turns")
    if not normalized_transcript(record):
        errors.append(f"{identifier}: empty normalized transcript")
    languages = record.get("language_profile") or []
    if not languages:
        errors.append(f"{identifier}: language_profile is empty")
    return errors


def _validate_annotation(row: Mapping[str, Any], tactic_names: set[str]) -> list[str]:
    errors: list[str] = []
    identifier = str(row.get("conversation_id") or "<missing-id>")
    if not str(row.get("annotator_id") or "").strip():
        errors.append(f"{identifier}: annotation missing annotator_id")
    if row.get("llm_used") is not False:
        errors.append(f"{identifier}: annotation must state llm_used=false")
    if row.get("is_scam") not in (0, 1):
        errors.append(f"{identifier}: is_scam must be 0 or 1")
    tactics = {str(value) for value in row.get("tactics") or []}
    unknown = sorted(tactics - tactic_names)
    if unknown:
        errors.append(f"{identifier}: unknown tactic labels {unknown}")
    evidence_tactics = {
        str(value.get("tactic"))
        for value in row.get("evidence_spans") or []
        if isinstance(value, dict)
    }
    missing_evidence = sorted(tactics - evidence_tactics)
    if missing_evidence:
        errors.append(f"{identifier}: tactics missing evidence spans {missing_evidence}")
    return errors


def _resolve_label(
    conversation_id: str,
    rows: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Any] | None,
    minimum_annotators: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    annotators = [str(row.get("annotator_id") or "") for row in rows]
    if len(set(annotators)) < minimum_annotators:
        errors.append(
            f"{conversation_id}: needs {minimum_annotators} independent annotators"
        )
        return None, errors
    if len(annotators) != len(set(annotators)):
        errors.append(f"{conversation_id}: duplicate annotation by the same annotator")
    signatures = {annotation_signature(row) for row in rows}
    if len(signatures) == 1:
        signature = next(iter(signatures))
        adjudicated = False
    else:
        if adjudication is None:
            errors.append(f"{conversation_id}: annotation disagreement lacks adjudication")
            return None, errors
        adjudicator = str(adjudication.get("adjudicator_id") or "")
        if not adjudicator or adjudicator in set(annotators):
            errors.append(f"{conversation_id}: adjudicator must be an independent third person")
        if adjudication.get("llm_used") is not False:
            errors.append(f"{conversation_id}: adjudication must state llm_used=false")
        if not str(adjudication.get("reason") or "").strip():
            errors.append(f"{conversation_id}: adjudication reason is required")
        signature = annotation_signature(adjudication)
        adjudicated = True
    if signature[0] not in (0, 1):
        errors.append(f"{conversation_id}: resolved label is invalid")
        return None, errors
    return {
        "is_scam": signature[0],
        "scam_type": signature[1],
        "tactics": list(signature[2]),
        "stage": signature[3],
        "provenance": "human_gold",
        "annotator_count": len(set(annotators)),
        "adjudicated": adjudicated,
    }, errors


def build_frozen_records(
    intake_records: Sequence[Mapping[str, Any]],
    annotation_rows: Sequence[Mapping[str, Any]],
    adjudication_rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    canonical_conversation_ids: Iterable[str] = (),
    canonical_transcript_hashes: Iterable[str] = (),
    tactic_names: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every gate and return deterministic, deidentified frozen rows."""
    errors: list[str] = []
    ids_seen: set[str] = set()
    hashes_seen: set[str] = set()
    canonical_ids = set(canonical_conversation_ids)
    canonical_hashes = set(canonical_transcript_hashes)
    tactic_set = set(tactic_names)
    annotations_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    adjudications_by_id: dict[str, Mapping[str, Any]] = {}

    for row in annotation_rows:
        errors.extend(_validate_annotation(row, tactic_set))
        annotations_by_id[str(row.get("conversation_id") or "")].append(row)
    for row in adjudication_rows:
        identifier = str(row.get("conversation_id") or "")
        adjudication_as_annotation = dict(row)
        adjudication_as_annotation["annotator_id"] = row.get("adjudicator_id")
        errors.extend(_validate_annotation(adjudication_as_annotation, tactic_set))
        if identifier in adjudications_by_id:
            errors.append(f"{identifier}: multiple adjudication rows")
        adjudications_by_id[identifier] = row

    frozen: list[dict[str, Any]] = []
    minimum_annotators = int(protocol["annotation"]["minimum_independent_annotators"])
    for record in sorted(intake_records, key=lambda row: str(row.get("conversation_id") or "")):
        errors.extend(_validate_intake(record, protocol))
        identifier = str(record.get("conversation_id") or "")
        digest = transcript_sha256(record)
        if identifier in ids_seen:
            errors.append(f"{identifier}: duplicate intake conversation_id")
        ids_seen.add(identifier)
        if digest in hashes_seen:
            errors.append(f"{identifier}: duplicate normalized transcript in frozen candidate")
        hashes_seen.add(digest)
        if identifier in canonical_ids:
            errors.append(f"{identifier}: conversation_id leaks from canonical corpus")
        if digest in canonical_hashes:
            errors.append(f"{identifier}: transcript duplicates canonical corpus")
        label, label_errors = _resolve_label(
            identifier,
            annotations_by_id.get(identifier, []),
            adjudications_by_id.get(identifier),
            minimum_annotators,
        )
        errors.extend(label_errors)
        if label is None:
            continue
        frozen.append(
            {
                "schema_version": "1.0.0",
                "conversation_id": identifier,
                "collection_channel": record.get("collection_channel"),
                "language_profile": sorted({str(value) for value in record.get("language_profile") or []}),
                "turns": [
                    {
                        "turn_id": index,
                        "speaker_role": str(turn.get("speaker_role") or "unknown"),
                        "text": " ".join(str(turn.get("text") or "").split()),
                    }
                    for index, turn in enumerate(record.get("turns") or [])
                    if str(turn.get("text") or "").strip()
                ],
                "label": label,
                "normalized_transcript_sha256": digest,
                "test_only": True,
                "llm_used_for_label": False,
            }
        )

    expected_intake_ids = {str(row.get("conversation_id") or "") for row in intake_records}
    orphan_annotations = sorted(set(annotations_by_id) - expected_intake_ids)
    orphan_adjudications = sorted(set(adjudications_by_id) - expected_intake_ids)
    if orphan_annotations:
        errors.append(f"annotations without intake records: {orphan_annotations}")
    if orphan_adjudications:
        errors.append(f"adjudications without intake records: {orphan_adjudications}")

    class_counts = Counter("scam" if row["label"]["is_scam"] else "non_scam" for row in frozen)
    language_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    channel_labels: dict[str, set[int]] = defaultdict(set)
    for row in frozen:
        language_counts.update(row["language_profile"])
        channel_labels[str(row["collection_channel"])].add(int(row["label"]["is_scam"]))
        if row["label"]["is_scam"]:
            type_counts[str(row["label"]["scam_type"])] += 1

    if len(frozen) < int(protocol["minimum_conversations"]):
        errors.append(
            f"only {len(frozen)} accepted conversations; minimum is {protocol['minimum_conversations']}"
        )
    for name, minimum in protocol["minimum_class_counts"].items():
        if class_counts[name] < int(minimum):
            errors.append(f"class {name} has {class_counts[name]}; minimum is {minimum}")
    for name, minimum in protocol["minimum_language_counts"].items():
        if language_counts[name] < int(minimum):
            errors.append(f"language {name} has {language_counts[name]}; minimum is {minimum}")
    for name, minimum in protocol["minimum_positive_scam_type_counts"].items():
        if type_counts[name] < int(minimum):
            errors.append(f"scam type {name} has {type_counts[name]}; minimum is {minimum}")
    if protocol["collection"]["require_both_classes_per_channel"]:
        for channel, labels in sorted(channel_labels.items()):
            if labels != {0, 1}:
                errors.append(f"collection channel {channel} does not contain both classes")

    if errors:
        raise FreezeValidationError(sorted(set(errors)))

    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": protocol["dataset_id"],
        "status": "frozen",
        "conversations": len(frozen),
        "class_counts": dict(sorted(class_counts.items())),
        "language_counts": dict(sorted(language_counts.items())),
        "positive_scam_type_counts": dict(sorted(type_counts.items())),
        "collection_channels": sorted(channel_labels),
        "label_provenance": "human_gold",
        "training_use_allowed": False,
        "threshold_tuning_use_allowed": False,
        "llm_used_for_labels": False,
    }
    return frozen, manifest
