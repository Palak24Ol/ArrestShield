from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.data import ConversationExample  # noqa: E402
from arrestshield.ladder import (  # noqa: E402
    build_feature_union,
    build_prefix_batch,
    choose_family,
    stable_latency_from_flat_scores,
)


class LadderTests(unittest.TestCase):
    def test_feature_union_supports_predeclared_ablation_groups(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "configs/model/model_ladder.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [name for name, _ in build_feature_union(config, ["word"]).transformer_list],
            ["word"],
        )
        self.assertEqual(
            [name for name, _ in build_feature_union(config, ["char"]).transformer_list],
            ["char"],
        )
        with self.assertRaisesRegex(ValueError, "enabled_groups"):
            build_feature_union(config, ["invalid"])

    def test_choose_family_uses_latency_inside_variance_band(self) -> None:
        aggregates = {
            "sgd": {
                "recall_mean": 0.80,
                "recall_std": 0.02,
                "median_stable_turn_mean": 3.0,
                "macro_f1_mean": 0.82,
                "artifact_bytes": 10,
            },
            "xgboost": {
                "recall_mean": 0.81,
                "recall_std": 0.02,
                "median_stable_turn_mean": 2.0,
                "macro_f1_mean": 0.80,
                "artifact_bytes": 20,
            },
        }
        self.assertEqual(choose_family(aggregates), "xgboost")

    def test_prefix_latency_counts_scammer_turns(self) -> None:
        example = ConversationExample(
            conversation_id="fixture",
            text="",
            label=1,
            scam_type="digital_arrest",
            split="test",
            source="fixture",
            languages=("hinglish",),
            provenance="source_silver",
            turn_texts=(
                "[ROLE=caller] hello",
                "[ROLE=victim] yes",
                "[ROLE=caller] transfer now",
            ),
        )
        positives, _, owners = build_prefix_batch([example])
        result = stable_latency_from_flat_scores(
            positives,
            owners,
            [0.1, 0.2, 0.9],
            entry_threshold=0.8,
            exit_threshold=0.6,
        )
        self.assertEqual(result["stable_scammer_turns"]["median_turn"], 2.0)


if __name__ == "__main__":
    unittest.main()
