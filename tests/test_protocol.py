from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.protocol import (  # noqa: E402
    select_threshold_at_fpr,
    stable_detection_turn,
    summarize_seed_values,
    summarize_stable_detection,
)


class ProtocolTests(unittest.TestCase):
    def test_ablation_protocol_keeps_selection_invariants(self) -> None:
        protocol = json.loads(
            (PROJECT_ROOT / "configs/evaluation/ablation_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(protocol["invariants"]["seeds"], [17, 42, 93])
        self.assertEqual(protocol["invariants"]["maximum_hard_negative_fpr"], 0.05)
        self.assertFalse(protocol["invariants"]["test_used_for_selection"])
        self.assertFalse(protocol["llm_used_for_any_ablation"])
        self.assertEqual(
            set(protocol["transformer_variants"]),
            {
                "binary_head_only",
                "no_tactic_head",
                "no_stage_head",
                "full_context_only",
                "right_truncation_control",
            },
        )

    def test_operating_point_respects_fpr_constraint(self) -> None:
        labels = [0, 0, 0, 0, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
        point = select_threshold_at_fpr(labels, scores, maximum_fpr=0.0)
        self.assertEqual(point.false_positive_rate, 0.0)
        self.assertEqual(point.recall, 1.0)
        self.assertGreater(point.threshold, 0.3)

    def test_stable_detection_requires_remaining_scores_above_exit(self) -> None:
        scores = [0.2, 0.8, 0.3, 0.75, 0.7]
        self.assertEqual(stable_detection_turn(scores, 0.7, 0.5), 4)
        self.assertIsNone(stable_detection_turn([0.8, 0.2], 0.7, 0.5))

    def test_stable_detection_summary_includes_undetected(self) -> None:
        summary = summarize_stable_detection([1, 2, None, 3])
        self.assertEqual(summary["median_turn"], 2.0)
        self.assertEqual(summary["undetected_rate"], 0.25)

    def test_seed_summary_reports_sample_deviation(self) -> None:
        summary = summarize_seed_values([0.7, 0.8, 0.9])
        self.assertAlmostEqual(summary["mean"], 0.8)
        self.assertAlmostEqual(summary["standard_deviation"], 0.1)


if __name__ == "__main__":
    unittest.main()
