import numpy as np
import pytest

from arrestshield.risk import RISK_FEATURE_NAMES, build_risk_matrix, risk_feature_row, risk_scores


def test_risk_feature_contract_is_fixed_and_multisignal() -> None:
    safe = risk_feature_row("[ROLE=agent] Welcome to customer support", 0.1)
    risky = risk_feature_row(
        "[ROLE=caller] Main CBI officer bol raha hoon. Kisi ko mat batana. "
        "Abhi Rs 50,000 transfer karo to shield@ybl and share OTP 445566.",
        0.8,
    )
    assert safe.shape == risky.shape == (len(RISK_FEATURE_NAMES),)
    assert risky[RISK_FEATURE_NAMES.index("entity_upi_id")] > 0
    assert risky[RISK_FEATURE_NAMES.index("lexical_authority_impersonation")] > 0
    assert risky[RISK_FEATURE_NAMES.index("lexical_secrecy")] > 0
    assert risky[RISK_FEATURE_NAMES.index("lexical_financial_demand")] > 0
    assert risky[RISK_FEATURE_NAMES.index("lexical_stage_progress")] == pytest.approx(5 / 6)


def test_build_matrix_checks_lengths_and_score_range() -> None:
    matrix = build_risk_matrix(["one", "two"], [0.1, 0.9])
    assert matrix.shape == (2, len(RISK_FEATURE_NAMES))
    with pytest.raises(ValueError, match="equal length"):
        build_risk_matrix(["one"], [0.1, 0.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        risk_feature_row("one", 1.1)


def test_risk_scores_enforces_non_llm_bundle() -> None:
    class FakeModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, matrix):
            return np.asarray([[0.2, 0.8]] * len(matrix))

    matrix = np.zeros((1, len(RISK_FEATURE_NAMES)), dtype=np.float32)
    assert risk_scores({"model": FakeModel(), "llm_used_for_detection": False}, matrix)[0] == 0.8
    with pytest.raises(ValueError, match="prohibit LLM"):
        risk_scores({"model": FakeModel()}, matrix)
