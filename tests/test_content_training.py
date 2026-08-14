from __future__ import annotations

from arrestshield.content_training import (
    external_as_examples,
    load_project_content_examples,
    partition_external_calls,
    source_label_balanced_weights,
)
from arrestshield.data import ConversationExample
from arrestshield.external_evaluation import ExternalTextRecord


def external(index: int, source_url: str) -> ExternalTextRecord:
    return ExternalTextRecord(
        record_id=f"record-{index}",
        conversation_id=f"conversation-{index}",
        text=f"[ROLE=caller] urgent transfer {index}",
        label=1,
        language="english",
        source_group="calls",
        rights_basis="CC0",
        pii_redacted=True,
        source_url=source_url,
    )


def test_external_partition_keeps_source_urls_together() -> None:
    records = [external(index, f"url-{index // 2}") for index in range(100)]
    first = partition_external_calls(records)
    second = partition_external_calls(list(reversed(records)))
    first_assignment = {
        record.source_url: split for split, rows in first.items() for record in rows
    }
    second_assignment = {
        record.source_url: split for split, rows in second.items() for record in rows
    }
    assert first_assignment == second_assignment
    assert set(first_assignment.values()) == {"train", "validation", "test"}


def test_external_conversion_and_source_label_weights() -> None:
    positive = external_as_examples([external(1, "url-one")], "train")[0]
    negative = ConversationExample(
        conversation_id="negative",
        text="normal support call",
        label=0,
        scam_type="non_scam",
        split="train",
        source="legitimate_calls",
        languages=("english",),
        provenance="source_gold",
        turn_texts=("normal support call",),
    )
    weights = source_label_balanced_weights([positive, negative])
    assert positive.split == "train"
    assert positive.label == 1
    assert weights.tolist() == [1.0, 1.0]


def test_project_examples_are_valid_mixed_label_data(tmp_path) -> None:
    path = tmp_path / "examples.jsonl"
    path.write_text(
        '{"id":"one","label":1,"split":"train","language":"english","text":"send money now"}\n'
        '{"id":"two","label":0,"split":"train","language":"english","text":"normal appointment"}\n'
        '{"id":"three","label":1,"split":"validation","language":"hinglish","text":"otp batao"}\n'
        '{"id":"four","label":0,"split":"validation","language":"hinglish","text":"otp mat batao"}\n',
        encoding="utf-8",
    )
    examples = load_project_content_examples(path)
    assert {row.label for row in examples["train"]} == {0, 1}
    assert {row.label for row in examples["validation"]} == {0, 1}
