"""Build source-aware evaluation views from the canonical corpus and split manifest."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERSATIONS = PROJECT_ROOT / "data/processed/conversations.jsonl"
SPLITS = PROJECT_ROOT / "data/splits/split_manifest.json"
OUTPUT = PROJECT_ROOT / "data/splits/evaluation_views.json"

RISK_OVERLAP = re.compile(
    r"\b(bank|account|police|cbi|rbi|aadhaar|aadhar|courier|parcel|otp|pin|payment|"
    r"transfer|money|fraud|card|kyc|loan|refund|security|verification|case|arrest|"
    r"खाता|बैंक|पुलिस|पैसे|भुगतान|ओटीपी|आधार)\b",
    re.IGNORECASE,
)


def main() -> int:
    split_payload = json.loads(SPLITS.read_text(encoding="utf-8"))
    split_map = split_payload["conversation_to_split"]
    group_map = split_payload["conversation_to_group"]
    records: list[dict[str, object]] = []
    source_labels: dict[str, Counter[int]] = defaultdict(Counter)

    with CONVERSATIONS.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            conversation_id = record["conversation_id"]
            label = record["conversation_label"]["is_scam"]
            if label not in (0, 1) or not record["quality"]["training_eligible"]:
                continue
            source = record["source"]["dataset_id"]
            text = " ".join(turn["normalized_text"] for turn in record["turns"])
            source_labels[source][label] += 1
            records.append(
                {
                    "conversation_id": conversation_id,
                    "group_id": group_map[conversation_id],
                    "split": split_map[conversation_id],
                    "source_channel": source,
                    "label": label,
                    "hard_negative": bool(label == 0 and RISK_OVERLAP.search(text)),
                }
            )

    mixed_sources = sorted(
        source for source, counts in source_labels.items() if counts[0] and counts[1]
    )
    views: dict[str, object] = {
        "schema_version": "1.0.0",
        "definition": (
            "source-channel evaluation overlays on the immutable conversation-group split; "
            "no row-level resplitting"
        ),
        "mixed_sources": mixed_sources,
        "source_label_counts": {
            source: {str(label): count for label, count in sorted(counts.items())}
            for source, counts in sorted(source_labels.items())
        },
        "hard_negative_ids": {
            split: [
                item["conversation_id"]
                for item in records
                if item["split"] == split and item["hard_negative"]
            ]
            for split in ("train", "validation", "test")
        },
        "mixed_source_ids": {
            split: [
                item["conversation_id"]
                for item in records
                if item["split"] == split and item["source_channel"] in mixed_sources
            ]
            for split in ("train", "validation", "test")
        },
        "leave_one_mixed_source_out": {},
    }
    for held_out_source in mixed_sources:
        views["leave_one_mixed_source_out"][held_out_source] = {
            "train_ids": [
                item["conversation_id"]
                for item in records
                if item["split"] == "train" and item["source_channel"] != held_out_source
            ],
            "validation_ids": [
                item["conversation_id"]
                for item in records
                if item["split"] == "validation" and item["source_channel"] != held_out_source
            ],
            "test_ids": [
                item["conversation_id"]
                for item in records
                if item["split"] == "test" and item["source_channel"] == held_out_source
            ],
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(views, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Mixed sources: {mixed_sources}")
    print(
        "Hard negatives:",
        {split: len(ids) for split, ids in views["hard_negative_ids"].items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
