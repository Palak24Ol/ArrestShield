from dataclasses import replace

from arrestshield.multitask import (
    MultiTaskExample,
    aggregate_turn_labels,
    create_context_windows,
    extract_turn_records,
    head_tail_token_ids,
    prefix_example,
    source_balanced_training_sample,
)


TACTICS = [
    "authority_impersonation",
    "fear_intimidation",
    "urgency_scarcity",
    "secrecy_instruction",
    "credential_request",
]
STAGES = [
    "none_unknown",
    "contact",
    "authority_claim",
    "accusation",
    "threat",
    "isolation_control",
    "payment_extraction",
]


def make_example(identifier: str, label: int, source: str) -> MultiTaskExample:
    return MultiTaskExample(
        conversation_id=identifier,
        text="first\nsecond\nthird\nfourth",
        turn_texts=("first", "second", "third", "fourth"),
        split="train",
        source=source,
        languages=("hinglish",),
        provenance="source_silver",
        sample_weight=0.75,
        binary_label=label,
        scam_type_label=1 if label else 0,
        tactic_labels=(0.0,) * len(TACTICS),
        tactic_mask=(0.0,) * len(TACTICS) if label else (1.0,) * len(TACTICS),
        stage_label=0,
        stage_mask=0.0 if label else 1.0,
        turn_tactics=((0.0,) * len(TACTICS),) * 4,
        turn_stages=("none_unknown",) * 4,
    )


def test_positive_unknown_tactics_are_downweighted_negatives():
    turns = [
        {
            "normalized_text": "police se baat kar rahe hain",
            "labels": {
                "tactics": {"fear_threat": 1, "urgency": 0},
                "stage": "threat",
            },
        }
    ]
    labels, mask, stage, stage_mask = aggregate_turn_labels(
        turns, 1, TACTICS, STAGES, unknown_tactic_negative_weight=0.25
    )
    assert labels[TACTICS.index("fear_intimidation")] == 1.0
    assert mask[TACTICS.index("fear_intimidation")] == 1.0
    assert labels[TACTICS.index("urgency_scarcity")] == 0.0
    assert mask[TACTICS.index("urgency_scarcity")] == 0.25
    assert stage == STAGES.index("threat")
    assert stage_mask == 1.0


def test_unannotated_positive_stays_fully_masked():
    turns = [{"normalized_text": "hello", "labels": {}}]
    _, mask, _, stage_mask = aggregate_turn_labels(
        turns, 1, TACTICS, STAGES, unknown_tactic_negative_weight=0.25
    )
    assert mask == (0.0,) * len(TACTICS)
    assert stage_mask == 0.0


def test_non_scam_supplies_all_zero_auxiliary_supervision():
    labels, mask, stage, stage_mask = aggregate_turn_labels([], 0, TACTICS, STAGES)
    assert labels == (0.0,) * len(TACTICS)
    assert mask == (1.0,) * len(TACTICS)
    assert stage == STAGES.index("none_unknown")
    assert stage_mask == 1.0


def test_source_balancing_keeps_all_positives_and_caps_negatives():
    positives = [make_example(f"p{i}", 1, "mixed") for i in range(4)]
    negatives = [make_example(f"m{i}", 0, "mixed") for i in range(10)]
    negatives += [make_example(f"n{i}", 0, "negative_only") for i in range(20)]
    selected = source_balanced_training_sample(
        positives + negatives,
        maximum_negative_to_positive_ratio=2.0,
        mixed_source_negative_ratio=1.0,
    )
    assert sum(item.binary_label == 1 for item in selected) == 4
    assert sum(item.binary_label == 0 for item in selected) == 8
    assert {item.conversation_id for item in positives}.issubset(
        {item.conversation_id for item in selected}
    )


def test_context_windows_are_prefixes_and_deduplicated():
    positive = make_example("positive", 1, "mixed")
    one_turn = replace(
        positive,
        conversation_id="one",
        text="only",
        turn_texts=("only",),
        turn_tactics=positive.turn_tactics[:1],
        turn_stages=positive.turn_stages[:1],
    )
    windows = create_context_windows(
        [positive, one_turn],
        positive_fractions=[0.25, 0.5, 1.0],
        negative_fractions=[1.0],
        maximum_windows=3,
        stage_names=STAGES,
        unknown_tactic_negative_weight=0.25,
    )
    assert [len(item.turn_texts) for item in windows] == [1, 2, 4, 1]
    assert windows[1].text == "first\nsecond"


def test_prefix_windows_do_not_inherit_future_labels():
    turns = [
        {"normalized_text": "namaste sir", "labels": {"stage": "contact"}},
        {
            "normalized_text": "CBI se bol raha hoon",
            "labels": {
                "tactics": {"authority_impersonation": 1},
                "stage": "authority_claim",
            },
        },
        {
            "normalized_text": "abhi paisa transfer karo",
            "labels": {
                "tactics": {"credential_otp_request": 1},
                "stage": "payment_extraction",
            },
        },
    ]
    texts, tactics, stages = extract_turn_records(turns, TACTICS)
    example = replace(
        make_example("causal", 1, "mixed"),
        text="\n".join(texts),
        turn_texts=texts,
        turn_tactics=tactics,
        turn_stages=stages,
        tactics_annotated=True,
        stages_annotated=True,
    )
    early = prefix_example(example, 1, STAGES, 0.25)
    assert early.stage_label == STAGES.index("contact")
    assert early.tactic_labels[TACTICS.index("credential_request")] == 0.0
    middle = prefix_example(example, 2, STAGES, 0.25)
    assert middle.stage_label == STAGES.index("authority_claim")
    assert middle.tactic_labels[TACTICS.index("authority_impersonation")] == 1.0
    assert middle.tactic_labels[TACTICS.index("credential_request")] == 0.0
    full = prefix_example(example, 3, STAGES, 0.25)
    assert full.stage_label == STAGES.index("payment_extraction")
    assert full.tactic_labels[TACTICS.index("credential_request")] == 1.0


def test_head_tail_encoding_preserves_early_and_latest_tokens():
    class FakeTokenizer:
        sep_token_id = 99
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False, **kwargs):
            return {"input_ids": [int(value) for value in text.split()]}

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def build_inputs_with_special_tokens(self, ids):
            return [101] + list(ids) + [102]

    encoded, attention = head_tail_token_ids(
        FakeTokenizer(), " ".join(str(value) for value in range(20)), 12, 0.5
    )
    assert encoded == [101, 0, 1, 2, 3, 99, 15, 16, 17, 18, 19, 102]
    assert attention == [1] * 12
