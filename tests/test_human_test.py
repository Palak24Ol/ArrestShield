import pytest

from arrestshield.human_test import FreezeValidationError, build_frozen_records


def protocol():
    return {
        "dataset_id": "test-frozen",
        "minimum_conversations": 2,
        "minimum_class_counts": {"scam": 1, "non_scam": 1},
        "minimum_language_counts": {"hinglish": 2},
        "minimum_positive_scam_type_counts": {"digital_arrest": 1},
        "annotation": {"minimum_independent_annotators": 2},
        "collection": {
            "allowed_channels": ["volunteer_roleplay"],
            "require_both_classes_per_channel": True,
            "require_consent": True,
            "require_pii_redaction": True,
            "llm_generated_dialogue_allowed": False,
            "minimum_turns": 2,
        },
    }


def intake(identifier, first, second):
    return {
        "conversation_id": identifier,
        "collection_channel": "volunteer_roleplay",
        "language_profile": ["hinglish"],
        "turns": [
            {"speaker_role": "caller", "text": first},
            {"speaker_role": "recipient", "text": second},
        ],
        "consent_confirmed": True,
        "pii_redacted": True,
        "llm_generated": False,
    }


def annotation(identifier, annotator, is_scam):
    scam = bool(is_scam)
    return {
        "conversation_id": identifier,
        "annotator_id": annotator,
        "is_scam": int(scam),
        "scam_type": "digital_arrest" if scam else "non_scam",
        "tactics": ["authority_impersonation"] if scam else [],
        "stage": "authority_claim" if scam else "none_unknown",
        "evidence_spans": (
            [
                {
                    "tactic": "authority_impersonation",
                    "turn_id": 0,
                    "text": "CBI officer",
                }
            ]
            if scam
            else []
        ),
        "llm_used": False,
    }


def valid_inputs():
    intake_rows = [
        intake("AHFT-0001", "I am a CBI officer", "Why are you calling?"),
        intake("AHFT-0002", "Your delivery is outside", "Thank you"),
    ]
    annotations = [
        annotation("AHFT-0001", "A", 1),
        annotation("AHFT-0001", "B", 1),
        annotation("AHFT-0002", "A", 0),
        annotation("AHFT-0002", "B", 0),
    ]
    return intake_rows, annotations


def test_valid_human_set_freezes_as_test_only_gold():
    intake_rows, annotations = valid_inputs()
    rows, manifest = build_frozen_records(
        intake_rows,
        annotations,
        [],
        protocol(),
        tactic_names={"authority_impersonation"},
    )
    assert len(rows) == 2
    assert manifest["label_provenance"] == "human_gold"
    assert manifest["training_use_allowed"] is False
    assert all(row["test_only"] for row in rows)
    assert all(row["llm_used_for_label"] is False for row in rows)


def test_freeze_rejects_canonical_leakage():
    intake_rows, annotations = valid_inputs()
    with pytest.raises(FreezeValidationError, match="leaks from canonical corpus"):
        build_frozen_records(
            intake_rows,
            annotations,
            [],
            protocol(),
            canonical_conversation_ids={"AHFT-0001"},
            tactic_names={"authority_impersonation"},
        )


def test_freeze_rejects_missing_independent_annotator():
    intake_rows, annotations = valid_inputs()
    annotations = [row for row in annotations if not (row["conversation_id"] == "AHFT-0001" and row["annotator_id"] == "B")]
    with pytest.raises(FreezeValidationError, match="needs 2 independent annotators"):
        build_frozen_records(
            intake_rows,
            annotations,
            [],
            protocol(),
            tactic_names={"authority_impersonation"},
        )


def test_freeze_rejects_unadjudicated_disagreement():
    intake_rows, annotations = valid_inputs()
    conflicting = annotation("AHFT-0001", "B", 0)
    annotations = [
        conflicting if row["conversation_id"] == "AHFT-0001" and row["annotator_id"] == "B" else row
        for row in annotations
    ]
    with pytest.raises(FreezeValidationError, match="disagreement lacks adjudication"):
        build_frozen_records(
            intake_rows,
            annotations,
            [],
            protocol(),
            tactic_names={"authority_impersonation"},
        )


def test_freeze_rejects_channel_label_shortcut():
    intake_rows, annotations = valid_inputs()
    intake_rows[1]["collection_channel"] = "consented_deidentified_recollection"
    changed = protocol()
    changed["collection"]["allowed_channels"].append("consented_deidentified_recollection")
    with pytest.raises(FreezeValidationError, match="does not contain both classes"):
        build_frozen_records(
            intake_rows,
            annotations,
            [],
            changed,
            tactic_names={"authority_impersonation"},
        )
