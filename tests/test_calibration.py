from __future__ import annotations

import numpy as np
import pytest

from arrestshield.calibration import (
    fit_score_calibrator,
    select_shared_threshold_at_group_fpr,
    split_calibration_and_threshold,
)
from arrestshield.data import ConversationExample


def example(index: int, label: int, source: str) -> ConversationExample:
    return ConversationExample(
        conversation_id=f"conversation-{index}",
        text=f"text {index}",
        label=label,
        scam_type="other_scam" if label else "non_scam",
        split="validation",
        source=source,
        languages=("english",),
        provenance="source_silver",
        turn_texts=(f"[ROLE=caller] text {index}",),
    )


@pytest.mark.parametrize("method", ["none", "platt", "isotonic"])
def test_calibrators_are_bounded_and_preserve_shape(method: str) -> None:
    scores = [0.05, 0.2, 0.8, 0.95]
    targets = [0, 0, 1, 1]
    calibrated = fit_score_calibrator(method, scores, targets).predict(scores)
    assert calibrated.shape == (4,)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_calibration_and_threshold_views_are_disjoint_and_stratified() -> None:
    examples = [
        example(index, label, source)
        for index, (source, label) in enumerate(
            [("a", 0), ("a", 0), ("a", 1), ("a", 1), ("b", 0), ("b", 0), ("b", 1), ("b", 1)]
        )
    ]
    calibration, threshold = split_calibration_and_threshold(examples)
    assert set(calibration).isdisjoint(threshold)
    assert set(calibration) | set(threshold) == set(range(len(examples)))
    assert {examples[index].label for index in calibration} == {0, 1}
    assert {examples[index].label for index in threshold} == {0, 1}


def test_shared_threshold_satisfies_every_seed_and_negative_source() -> None:
    result = select_shared_threshold_at_group_fpr(
        labels=[0, 0, 0, 0, 1, 1],
        score_rows=[
            [0.9, 0.2, 0.1, 0.05, 0.8, 0.7],
            [0.8, 0.3, 0.2, 0.1, 0.9, 0.4],
        ],
        groups=["bank", "bank", "bank", "bank", "positive", "positive"],
        maximum_fpr=0.25,
        minimum_group_negatives=4,
    )
    assert result["all_seed_source_gates_passed"] is True
    assert result["threshold"] > 0.3
    assert all(row["maximum_source_fpr"] <= 0.25 for row in result["per_seed"])
