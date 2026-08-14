"""Single-call inference orchestration with an explicit detector/honeypot boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import uuid

import joblib

from .asr_evaluation import frozen_detector_scores
from .data import format_turn
from .entities import extract_entities
from .risk import build_risk_matrix, lexical_signal_summary, risk_scores


@dataclass(frozen=True)
class InferencePolicy:
    detector_status: str = "research_only_not_promoted"
    allow_research_fusion: bool = False
    enable_honeypot_handoff: bool = False
    maximum_turns: int = 100
    maximum_characters: int = 50_000


class DetectorEngine:
    """Combine trained detector outputs; never invoke or consult an LLM."""

    def __init__(
        self,
        base_bundle: Mapping[str, Any],
        policy: InferencePolicy,
        fusion_bundle: Mapping[str, Any] | None = None,
        auxiliary_predictor: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        if base_bundle.get("llm_used_for_detection") is not False:
            raise ValueError("Base detector must explicitly set llm_used_for_detection=false")
        if fusion_bundle is not None and fusion_bundle.get("llm_used_for_detection") is not False:
            raise ValueError("Fusion detector must explicitly set llm_used_for_detection=false")
        self.base_bundle = dict(base_bundle)
        self.fusion_bundle = dict(fusion_bundle) if fusion_bundle is not None else None
        self.policy = policy
        self.auxiliary_predictor = auxiliary_predictor

    @classmethod
    def from_paths(
        cls,
        base_path: Path,
        policy: InferencePolicy,
        fusion_path: Path | None = None,
        auxiliary_predictor: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> "DetectorEngine":
        base_bundle = joblib.load(base_path)
        fusion_bundle = joblib.load(fusion_path) if fusion_path and fusion_path.exists() else None
        return cls(base_bundle, policy, fusion_bundle, auxiliary_predictor)

    def model_info(self) -> dict[str, Any]:
        fusion = self.fusion_bundle
        fusion_status = fusion.get("promotion_status") if fusion else "not_loaded"
        return {
            "base_detector": {
                "family": self.base_bundle.get("model_family"),
                "seed": self.base_bundle.get("seed"),
                "threshold": self.base_bundle.get("threshold"),
                "calibration": self.base_bundle.get("calibration_method", "none"),
                "feature_variant": self.base_bundle.get("feature_variant", "word_char"),
            },
            "risk_fusion": {
                "loaded": fusion is not None,
                "family": fusion.get("model_family") if fusion else None,
                "seed": fusion.get("seed") if fusion else None,
                "promotion_status": fusion_status,
                "allowed_by_policy": bool(
                    fusion is not None
                    and (fusion_status == "eligible" or self.policy.allow_research_fusion)
                ),
            },
            "detector_status": self.policy.detector_status,
            "honeypot_handoff_enabled": self.policy.enable_honeypot_handoff,
            "llm_used_for_detection": False,
        }

    def detect(
        self,
        turns: Sequence[Mapping[str, str]],
        conversation_id: str | None = None,
        include_sensitive_entities: bool = False,
    ) -> dict[str, Any]:
        if not turns:
            raise ValueError("At least one turn is required")
        if len(turns) > self.policy.maximum_turns:
            raise ValueError("Conversation exceeds maximum_turns")
        formatted: list[str] = []
        for index, turn in enumerate(turns):
            value = str(turn.get("text") or "").strip()
            if not value:
                raise ValueError(f"Turn {index} has empty text")
            formatted.append(format_turn(str(turn.get("speaker_role") or "unknown"), value))
        text = "\n".join(formatted)
        if len(text) > self.policy.maximum_characters:
            raise ValueError("Conversation exceeds maximum_characters")

        base_score = float(frozen_detector_scores(self.base_bundle, [text])[0])
        base_threshold = float(self.base_bundle["threshold"])
        score = base_score
        threshold = base_threshold
        decision_source = "trained_base_detector"
        fusion_payload: dict[str, Any] = {"loaded": self.fusion_bundle is not None, "used": False}
        if self.fusion_bundle is not None:
            feature_matrix = build_risk_matrix([text], [base_score])
            fusion_score = float(risk_scores(self.fusion_bundle, feature_matrix)[0])
            fusion_status = str(self.fusion_bundle.get("promotion_status", "unknown"))
            fusion_allowed = fusion_status == "eligible" or self.policy.allow_research_fusion
            fusion_payload.update(
                {
                    "score": fusion_score,
                    "threshold": float(self.fusion_bundle["threshold"]),
                    "promotion_status": fusion_status,
                    "allowed_by_policy": fusion_allowed,
                }
            )
            if fusion_allowed:
                score = fusion_score
                threshold = float(self.fusion_bundle["threshold"])
                decision_source = "trained_xgboost_risk_fusion"
                fusion_payload["used"] = True

        decision = bool(score >= threshold)
        entities = extract_entities(text)
        if self.auxiliary_predictor is None:
            auxiliary = lexical_signal_summary(text)
        else:
            auxiliary = dict(self.auxiliary_predictor(text))
            if auxiliary.get("llm_used") is not False:
                raise ValueError("Auxiliary predictor must explicitly set llm_used=false")

        production_eligible = self.policy.detector_status == "eligible"
        handoff = bool(
            decision and production_eligible and self.policy.enable_honeypot_handoff
        )
        if not decision:
            blocked_reason = "detector_did_not_cross_threshold"
        elif not production_eligible:
            blocked_reason = "detector_is_research_only"
        elif not self.policy.enable_honeypot_handoff:
            blocked_reason = "handoff_disabled_by_policy"
        else:
            blocked_reason = None
        return {
            "schema_version": "1.0.0",
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "is_scam": decision,
            "scam_score": score,
            "threshold": threshold,
            "decision_source": decision_source,
            "detector_status": self.policy.detector_status,
            "production_eligible": production_eligible,
            "base_detector": {
                "score": base_score,
                "threshold": base_threshold,
                "family": self.base_bundle.get("model_family"),
                "seed": self.base_bundle.get("seed"),
                "calibration": self.base_bundle.get("calibration_method", "none"),
                "feature_variant": self.base_bundle.get("feature_variant", "word_char"),
            },
            "risk_fusion": fusion_payload,
            "auxiliary_signals": auxiliary,
            "entities": [
                entity.public_dict(include_sensitive_entities) for entity in entities
            ],
            "honeypot": {
                "invoked": False,
                "handoff_recommended": handoff,
                "handoff_blocked_reason": blocked_reason,
                "llm_used_for_detection": False,
                "boundary": "An LLM may engage only after this ML response and external policy approval.",
            },
            "llm_used_for_detection": False,
        }
