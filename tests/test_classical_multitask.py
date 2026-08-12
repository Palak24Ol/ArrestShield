import numpy as np
import pytest

from arrestshield.classical_multitask import (
    aligned_probabilities,
    balanced_sample_weights,
    format_classical_multitask_outputs,
    select_f1_threshold,
)


class FakeMulticlassModel:
    classes_ = np.asarray([0, 2])

    def predict_proba(self, matrix):
        return np.asarray([[0.25, 0.75] for _ in range(len(matrix))])


MANIFEST = {
    "binary_threshold": 0.6,
    "binary_family": "sgd",
    "selection_role": "research_only_auxiliary",
    "labels": {
        "scam_types": ["non_scam", "digital_arrest"],
        "tactics": ["authority_impersonation", "phantom_riches"],
        "stages": ["none_unknown", "payment_extraction"],
    },
    "supported_tactics": ["authority_impersonation"],
    "tactic_thresholds": {"authority_impersonation": 0.4},
}


def test_balanced_weights_are_finite_capped_and_aligned() -> None:
    weights = balanced_sample_weights([0, 0, 0, 1], [1.0, 0.75, 1.0, 0.75], 3.0)
    assert weights.shape == (4,)
    assert np.isfinite(weights).all()
    assert weights[3] > weights[0]
    assert weights[3] <= 0.75 * 3.0


def test_probability_alignment_preserves_missing_manifest_column() -> None:
    output = aligned_probabilities(FakeMulticlassModel(), np.zeros((1, 2)), 3)
    assert output.shape == (1, 3)
    assert output[0].tolist() == pytest.approx([0.25, 0.0, 0.75])


def test_validation_threshold_selection_is_deterministic() -> None:
    first = select_f1_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    second = select_f1_threshold([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert first == second
    assert first["f1"] == 1.0
    assert 0.3 < first["threshold"] <= 0.7


def test_formats_supported_and_unavailable_tactics_without_decision_role() -> None:
    result = format_classical_multitask_outputs(
        MANIFEST,
        binary_score=0.8,
        scam_type_scores=[0.1, 0.9],
        tactic_scores={"authority_impersonation": 0.7},
        stage_scores=[0.2, 0.8],
    )
    assert result["binary"]["is_scam"] is True
    assert result["scam_type"]["label"] == "digital_arrest"
    assert result["tactics"]["authority_impersonation"]["present"] is True
    assert result["tactics"]["phantom_riches"]["available"] is False
    assert result["stage"]["label"] == "payment_extraction"
    assert result["used_as_api_decision_source"] is False
    assert result["llm_used"] is False
