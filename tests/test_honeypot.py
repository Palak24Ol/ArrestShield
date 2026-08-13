"""Honeypot gate, safety, and boundary tests. No network calls are made."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arrestshield.honeypot import (  # noqa: E402
    HandoffRejected,
    HoneypotError,
    HoneypotPolicy,
    SyntheticIdentity,
    _verhoeff_checksum,
    build_handoff_event,
    evaluate_eligibility,
    redact_model_output,
    start_session,
    verify_handoff,
)

SECRET = "test-secret-not-a-real-key"
CONFIG = json.loads((ROOT / "configs/deployment/honeypot.json").read_text(encoding="utf-8"))


def detection(is_scam=True, production_eligible=False, score=0.97):
    return {
        "conversation_id": "conv-1",
        "is_scam": is_scam,
        "scam_score": score,
        "threshold": 0.5,
        "detector_status": "eligible" if production_eligible else "research_only_not_promoted",
        "production_eligible": production_eligible,
        "decision_source": "trained_base_detector",
        "auxiliary_signals": {"authority_terms": 3, "payment_terms": 2},
    }


def enabled_config(**policy_overrides):
    config = json.loads(json.dumps(CONFIG))
    config["policy"].update(policy_overrides)
    return config


def scripted(reply):
    def transport(messages, settings):
        transport.messages = messages
        return reply

    return transport


# --- signed handoff contract ------------------------------------------------


def test_valid_signature_verifies():
    event = build_handoff_event(detection(), SECRET)
    verify_handoff(event, SECRET, 300)


def test_tampered_score_is_rejected():
    event = build_handoff_event(detection(), SECRET)
    event["scam_score"] = 0.99
    with pytest.raises(HandoffRejected, match="signature"):
        verify_handoff(event, SECRET, 300)


def test_unsigned_event_is_rejected():
    event = build_handoff_event(detection(), SECRET)
    event.pop("signature")
    with pytest.raises(HandoffRejected, match="unsigned"):
        verify_handoff(event, SECRET, 300)


def test_wrong_secret_is_rejected():
    event = build_handoff_event(detection(), SECRET)
    with pytest.raises(HandoffRejected, match="signature"):
        verify_handoff(event, "different-secret", 300)


def test_stale_event_is_rejected():
    event = build_handoff_event(detection(), SECRET)
    future = datetime.now(timezone.utc) + timedelta(seconds=600)
    with pytest.raises(HandoffRejected, match="stale"):
        verify_handoff(event, SECRET, 300, now=future)


# --- eligibility gate -------------------------------------------------------


def test_default_policy_blocks_engagement():
    """The shipped config must refuse to engage until an operator opts in."""
    event = build_handoff_event(detection(), SECRET)
    decision = evaluate_eligibility(event, HoneypotPolicy.from_config(CONFIG))
    assert decision.allowed is False
    assert decision.reason == "honeypot_disabled_by_policy"


def test_research_mode_allows_but_marks_session():
    event = build_handoff_event(detection(), SECRET)
    policy = HoneypotPolicy.from_config(enabled_config(research_mode=True))
    decision = evaluate_eligibility(event, policy)
    assert decision.allowed is True
    assert decision.mode == "research_only"


def test_enabled_policy_still_blocks_unpromoted_detector():
    event = build_handoff_event(detection(production_eligible=False), SECRET)
    policy = HoneypotPolicy.from_config(enabled_config(enabled=True))
    decision = evaluate_eligibility(event, policy)
    assert decision.allowed is False
    assert decision.reason == "detector_is_research_only"


def test_promoted_detector_reaches_live_mode():
    event = build_handoff_event(detection(production_eligible=True), SECRET)
    policy = HoneypotPolicy.from_config(enabled_config(enabled=True))
    assert evaluate_eligibility(event, policy).mode == "live"


def test_non_scam_never_engages_even_when_fully_enabled():
    event = build_handoff_event(detection(is_scam=False, production_eligible=True), SECRET)
    policy = HoneypotPolicy.from_config(enabled_config(enabled=True, research_mode=True))
    decision = evaluate_eligibility(event, policy)
    assert decision.allowed is False
    assert decision.reason == "detector_did_not_flag_scam"


# --- synthetic identity -----------------------------------------------------


def test_identity_values_cannot_be_real():
    identity = SyntheticIdentity.create("Sunita Rao", "conv-1|event-1")
    assert identity.is_synthetic()
    # Aadhaar-shaped value must fail the real Verhoeff check.
    assert _verhoeff_checksum(identity.aadhaar_like.replace(" ", "")) != 0
    # Indian mobile numbers never begin with 0 after the country code.
    assert " 0" in identity.phone
    assert identity.upi_like.endswith("@invalid")
    assert identity.account_like.startswith("0000")


def test_identity_is_deterministic_per_conversation():
    first = SyntheticIdentity.create("Sunita Rao", "conv-1|event-1")
    second = SyntheticIdentity.create("Sunita Rao", "conv-1|event-1")
    third = SyntheticIdentity.create("Sunita Rao", "conv-2|event-2")
    assert first == second
    assert first != third


# --- output redaction -------------------------------------------------------


def test_real_looking_phone_is_redacted():
    identity = SyntheticIdentity.create("Sunita Rao", "seed")
    text, redactions = redact_model_output("Call me on +91 98765 43210 please", identity)
    assert "98765" not in text
    assert redactions


def test_synthetic_identifiers_survive_redaction():
    identity = SyntheticIdentity.create("Sunita Rao", "seed")
    text, _ = redact_model_output(f"My UPI is {identity.upi_like}", identity)
    assert identity.upi_like in text


# --- session behaviour ------------------------------------------------------


def test_session_produces_reply_and_marks_transcript_non_evidential():
    event = build_handoff_event(detection(), SECRET)
    session = start_session(
        event, enabled_config(research_mode=True), SECRET, transport=scripted("Haan beta, boliye")
    )
    result = session.respond("Madam, CBI se bol raha hoon")
    assert result["reply"] == "Haan beta, boliye"
    assert result["llm_used_for_detection"] is False

    transcript = session.transcript()
    assert transcript["excluded_from_detector_training"] is True
    assert transcript["usable_as_detector_evidence"] is False
    assert transcript["synthetic_identity_verified"] is True


def test_disengage_token_stops_engagement():
    event = build_handoff_event(detection(), SECRET)
    session = start_session(
        event, enabled_config(research_mode=True), SECRET, transport=scripted("HONEYPOT_DISENGAGE")
    )
    result = session.respond("Actually I need help, I was robbed")
    assert result["engaged"] is False
    assert result["stop_reason"] == "model_requested_disengage"
    assert session.respond("hello")["engaged"] is False


def test_turn_limit_is_enforced():
    event = build_handoff_event(detection(), SECRET)
    session = start_session(
        event,
        enabled_config(research_mode=True, maximum_turns=2),
        SECRET,
        transport=scripted("theek hai"),
    )
    assert session.respond("one")["engaged"] is True
    assert session.respond("two")["engaged"] is True
    third = session.respond("three")
    assert third["engaged"] is False
    assert third["stop_reason"] == "maximum_turns_reached"


def test_reply_is_truncated_to_policy_limit():
    event = build_handoff_event(detection(), SECRET)
    session = start_session(
        event,
        enabled_config(research_mode=True, maximum_reply_characters=20),
        SECRET,
        transport=scripted("x" * 500),
    )
    assert len(session.respond("hi")["reply"]) <= 23


def test_start_session_refuses_blocked_event():
    event = build_handoff_event(detection(), SECRET)
    with pytest.raises(HandoffRejected, match="honeypot_disabled_by_policy"):
        start_session(event, CONFIG, SECRET, transport=scripted("hi"))


def test_start_session_refuses_forged_event():
    event = build_handoff_event(detection(), SECRET)
    event["conversation_id"] = "someone-elses-call"
    with pytest.raises(HandoffRejected):
        start_session(event, enabled_config(research_mode=True), SECRET, transport=scripted("hi"))


# --- boundary ---------------------------------------------------------------


def test_honeypot_module_does_not_import_detector():
    """The honeypot must not be able to reach detection code at all."""
    source = (ROOT / "src/arrestshield/honeypot.py").read_text(encoding="utf-8")
    for forbidden in ("inference", "risk", "baseline", "ladder", "multitask", "asr_evaluation"):
        assert f"from .{forbidden}" not in source
        assert f"import {forbidden}" not in source


def test_missing_api_key_raises_before_any_network_call(monkeypatch):
    from arrestshield.honeypot import GroqSettings, groq_chat
    from arrestshield.llm_client import LLMError

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        groq_chat([{"role": "user", "content": "hi"}], GroqSettings())
