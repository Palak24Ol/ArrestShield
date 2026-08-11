from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.data import format_conversation, load_examples  # noqa: E402
from arrestshield.evaluation import choose_threshold, early_detection_metrics  # noqa: E402


class TrainingFoundationTests(unittest.TestCase):
    def test_format_conversation_preserves_role_and_unicode(self) -> None:
        text, turns = format_conversation(
            [
                {"speaker_role": "Caller", "normalized_text": "  Main CBI se bol raha hoon. "},
                {"speaker_role": "victim", "raw_text": "क्या हुआ?"},
            ]
        )
        self.assertEqual(turns[0], "[ROLE=caller] Main CBI se bol raha hoon.")
        self.assertIn("[ROLE=victim] क्या हुआ?", text)

    def test_threshold_selection_honors_precision_constraint(self) -> None:
        result = choose_threshold(
            labels=[0, 0, 1, 1],
            scores=[0.1, 0.8, 0.7, 0.9],
            beta=2.0,
            min_precision=0.6,
            grid_size=99,
        )
        self.assertTrue(result["min_precision_satisfied"])
        self.assertGreaterEqual(result["validation_precision"], 0.6)

    def test_loader_uses_manifest_split_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversations = root / "conversations.jsonl"
            manifest = root / "splits.json"
            records = []
            for index, (split, label) in enumerate(
                [("train", 0), ("train", 1), ("validation", 0), ("test", 1)]
            ):
                records.append(
                    {
                        "conversation_id": f"conversation-{index}",
                        "conversation_label": {
                            "is_scam": label,
                            "scam_type": "other_scam" if label else "non_scam",
                            "provenance": "source_silver",
                        },
                        "quality": {"training_eligible": True},
                        "source": {"dataset_id": "fixture"},
                        "language_profile": ["hinglish"],
                        "turns": [
                            {
                                "speaker_role": "caller",
                                "normalized_text": f"fixture {index}",
                            }
                        ],
                    }
                )
            conversations.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            manifest.write_text(
                json.dumps(
                    {
                        "conversation_to_split": {
                            f"conversation-{index}": split
                            for index, (split, _) in enumerate(
                                [("train", 0), ("train", 1), ("validation", 0), ("test", 1)]
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            by_split, summary = load_examples(
                conversations,
                manifest,
                allowed_provenance=["source_silver"],
                provenance_weights={"source_silver": 0.75},
            )
            self.assertEqual([example.label for example in by_split["train"]], [0, 1])
            self.assertEqual(summary.split_counts["validation"], 1)
            self.assertEqual(by_split["test"][0].sample_weight, 0.75)

    def test_early_detection_uses_prefixes(self) -> None:
        from arrestshield.data import ConversationExample

        example = ConversationExample(
            conversation_id="positive",
            text="safe\nscam",
            label=1,
            scam_type="digital_arrest",
            split="test",
            source="fixture",
            languages=("hinglish",),
            provenance="source_silver",
            turn_texts=("safe", "scam"),
        )

        def score(values: list[str]) -> np.ndarray:
            return np.asarray([0.9 if "scam" in value else 0.1 for value in values])

        result = early_detection_metrics([example], score, threshold=0.5, fractions=[0.5, 1.0])
        self.assertEqual(result["detection_rate_by_available_conversation"]["at_50_percent"], 0.0)
        self.assertEqual(result["detection_rate_by_available_conversation"]["at_100_percent"], 1.0)
        self.assertEqual(result["median_fraction_to_first_detection"], 1.0)


if __name__ == "__main__":
    unittest.main()
