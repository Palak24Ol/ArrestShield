import pytest

from arrestshield.transformer_inference import format_multitask_outputs


MANIFEST = {
    "threshold": 0.7,
    "selection_role": "feasibility_only",
    "labels": {
        "scam_types": ["non_scam", "digital_arrest"],
        "tactics": ["authority_impersonation", "financial_demand"],
        "stages": ["contact", "payment_extraction"],
    },
}


def test_formats_all_multitask_heads_without_becoming_decision_source() -> None:
    result = format_multitask_outputs(
        MANIFEST,
        binary_score=0.8,
        scam_type_scores=[0.1, 0.9],
        tactic_scores=[0.7, 0.2],
        stage_scores=[0.25, 0.75],
    )
    assert result["binary"] == {"score": 0.8, "threshold": 0.7, "is_scam": True}
    assert result["scam_type"]["label"] == "digital_arrest"
    assert result["tactics"]["authority_impersonation"]["present_at_0_5"] is True
    assert result["stage"]["label"] == "payment_extraction"
    assert result["used_as_api_decision_source"] is False
    assert result["llm_used"] is False


def test_rejects_output_manifest_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Tactic output size"):
        format_multitask_outputs(
            MANIFEST,
            binary_score=0.5,
            scam_type_scores=[0.5, 0.5],
            tactic_scores=[0.5],
            stage_scores=[0.5, 0.5],
        )
