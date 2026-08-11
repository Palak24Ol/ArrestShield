import numpy as np

from arrestshield.inference import DetectorEngine, InferencePolicy


class FakeRepresentation:
    def transform(self, texts):
        return np.asarray(
            [[0.9 if any(token in text.casefold() for token in ("transfer", "otp", "cbi")) else 0.1]
             for text in texts],
            dtype=np.float32,
        )


class FakeBinaryModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix):
        scores = np.asarray(matrix)[:, 0]
        return np.column_stack([1.0 - scores, scores])


class FakeFusionModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix):
        scores = np.clip(np.asarray(matrix)[:, 0] + 0.05, 0, 1)
        return np.column_stack([1.0 - scores, scores])


def base_bundle() -> dict:
    return {
        "model_family": "fake_sgd",
        "seed": 42,
        "threshold": 0.5,
        "feature_union": FakeRepresentation(),
        "svd": None,
        "model": FakeBinaryModel(),
        "llm_used_for_detection": False,
    }


def fusion_bundle(status: str = "research_only_not_promoted") -> dict:
    return {
        "model_family": "xgboost_risk_fusion",
        "seed": 42,
        "threshold": 0.6,
        "model": FakeFusionModel(),
        "promotion_status": status,
        "llm_used_for_detection": False,
    }


def test_engine_uses_trained_fusion_but_blocks_research_handoff() -> None:
    engine = DetectorEngine(
        base_bundle(),
        InferencePolicy(
            detector_status="research_only_not_promoted",
            allow_research_fusion=True,
            enable_honeypot_handoff=True,
        ),
        fusion_bundle(),
    )
    result = engine.detect(
        [{"speaker_role": "caller", "text": "CBI officer: transfer to shield@ybl"}],
        conversation_id="demo",
    )
    assert result["conversation_id"] == "demo"
    assert result["is_scam"] is True
    assert result["decision_source"] == "trained_xgboost_risk_fusion"
    assert result["production_eligible"] is False
    assert result["honeypot"]["invoked"] is False
    assert result["honeypot"]["handoff_recommended"] is False
    assert result["honeypot"]["handoff_blocked_reason"] == "detector_is_research_only"
    assert result["llm_used_for_detection"] is False
    assert "shield@ybl" not in repr(result["entities"])


def test_engine_does_not_use_unpromoted_fusion_without_opt_in() -> None:
    engine = DetectorEngine(
        base_bundle(),
        InferencePolicy(allow_research_fusion=False),
        fusion_bundle(),
    )
    result = engine.detect([{"speaker_role": "caller", "text": "transfer now"}])
    assert result["decision_source"] == "trained_base_detector"
    assert result["risk_fusion"]["used"] is False


def test_raw_sensitive_entity_values_require_explicit_opt_in() -> None:
    engine = DetectorEngine(base_bundle(), InferencePolicy())
    turn = [{"speaker_role": "caller", "text": "pay shield@ybl"}]
    redacted = engine.detect(turn)
    raw = engine.detect(turn, include_sensitive_entities=True)
    assert "shield@ybl" not in repr(redacted["entities"])
    assert "shield@ybl" in repr(raw["entities"])


def test_engine_rejects_bad_limits_and_noncompliant_bundles() -> None:
    engine = DetectorEngine(base_bundle(), InferencePolicy(maximum_turns=1))
    try:
        engine.detect([])
        raise AssertionError("empty conversation should fail")
    except ValueError as error:
        assert "At least one turn" in str(error)
    try:
        engine.detect([
            {"speaker_role": "a", "text": "one"},
            {"speaker_role": "b", "text": "two"},
        ])
        raise AssertionError("turn limit should fail")
    except ValueError as error:
        assert "maximum_turns" in str(error)

    invalid = base_bundle()
    invalid.pop("llm_used_for_detection")
    try:
        DetectorEngine(invalid, InferencePolicy())
        raise AssertionError("noncompliant detector should fail")
    except ValueError as error:
        assert "llm_used_for_detection=false" in str(error)
