"""Post-detection fake-victim honeypot.

The honeypot engages a caller only after a trained detector has already decided,
and it can never change that decision. It does not import the detector, score
text, or return a label. Its transcripts are marked non-evidential so they can
never be recycled into detector training.

Every identifier the persona reveals is synthetic and deliberately constructed to
fail real-world validation, so a transcript can never contain a live account,
Aadhaar, or phone number belonging to an actual person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from .llm_client import GroqSettings, LLMError, groq_chat


SCHEMA_VERSION = "1.0.0"

# Signed fields, in a fixed order, so both sides canonicalize identically.
SIGNED_FIELDS = (
    "event_id",
    "conversation_id",
    "issued_at_utc",
    "scam_score",
    "threshold",
    "detector_status",
    "production_eligible",
    "decision_source",
)


class HoneypotError(RuntimeError):
    """Raised when the honeypot refuses to run."""


class HandoffRejected(HoneypotError):
    """Raised when a handoff event fails verification or policy."""


# --------------------------------------------------------------------------
# Signed handoff contract
# --------------------------------------------------------------------------


def canonical_payload(event: Mapping[str, Any]) -> str:
    """Serialize the signed subset deterministically."""
    missing = [name for name in SIGNED_FIELDS if name not in event]
    if missing:
        raise HandoffRejected(f"Handoff event missing fields: {missing}")
    return json.dumps(
        {name: event[name] for name in SIGNED_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sign_handoff(event: Mapping[str, Any], secret: str) -> str:
    if not secret:
        raise HandoffRejected("Refusing to sign with an empty secret")
    return hmac.new(
        secret.encode("utf-8"), canonical_payload(event).encode("utf-8"), hashlib.sha256
    ).hexdigest()


def build_handoff_event(detection: Mapping[str, Any], secret: str) -> dict[str, Any]:
    """Turn a DetectorEngine.detect() response into a signed handoff event."""
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "conversation_id": str(detection.get("conversation_id") or uuid.uuid4()),
        "issued_at_utc": datetime.now(timezone.utc).isoformat(),
        "scam_score": float(detection["scam_score"]),
        "threshold": float(detection["threshold"]),
        "detector_status": str(detection.get("detector_status") or "unknown"),
        "production_eligible": bool(detection.get("production_eligible", False)),
        "decision_source": str(detection.get("decision_source") or "unknown"),
        "is_scam": bool(detection.get("is_scam", False)),
        "auxiliary_signals": detection.get("auxiliary_signals") or {},
    }
    event["signature"] = sign_handoff(event, secret)
    return event


def verify_handoff(
    event: Mapping[str, Any],
    secret: str,
    maximum_age_seconds: int,
    now: datetime | None = None,
) -> None:
    """Reject forged, stale, or unsigned events before any LLM call is made."""
    signature = str(event.get("signature") or "")
    if not signature:
        raise HandoffRejected("Handoff event is unsigned")
    expected = sign_handoff(event, secret)
    if not hmac.compare_digest(expected, signature):
        raise HandoffRejected("Handoff signature does not verify")
    try:
        issued = datetime.fromisoformat(str(event["issued_at_utc"]))
    except ValueError as error:
        raise HandoffRejected("Handoff issued_at_utc is not ISO-8601") from error
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=timezone.utc)
    age = ((now or datetime.now(timezone.utc)) - issued).total_seconds()
    if age > maximum_age_seconds:
        raise HandoffRejected(f"Handoff event is stale ({age:.0f}s old)")
    if age < -60:
        raise HandoffRejected("Handoff event is issued in the future")


# --------------------------------------------------------------------------
# Eligibility policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HoneypotPolicy:
    enabled: bool = False
    research_mode: bool = False
    require_production_eligible: bool = True
    minimum_scam_score: float = 0.0
    maximum_turns: int = 20
    maximum_session_seconds: int = 1800
    maximum_reply_characters: int = 400

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HoneypotPolicy":
        values = config.get("policy") or {}
        return cls(
            enabled=bool(values.get("enabled", False)),
            research_mode=bool(values.get("research_mode", False)),
            require_production_eligible=bool(values.get("require_production_eligible", True)),
            minimum_scam_score=float(values.get("minimum_scam_score", 0.0)),
            maximum_turns=int(values.get("maximum_turns", 20)),
            maximum_session_seconds=int(values.get("maximum_session_seconds", 1800)),
            maximum_reply_characters=int(values.get("maximum_reply_characters", 400)),
        )


@dataclass(frozen=True)
class EligibilityDecision:
    allowed: bool
    mode: str
    reason: str | None


def evaluate_eligibility(event: Mapping[str, Any], policy: HoneypotPolicy) -> EligibilityDecision:
    """Decide whether engagement may start. Default posture is refusal."""
    if not bool(event.get("is_scam", False)):
        return EligibilityDecision(False, "blocked", "detector_did_not_flag_scam")
    if float(event.get("scam_score", 0.0)) < policy.minimum_scam_score:
        return EligibilityDecision(False, "blocked", "scam_score_below_minimum")
    if not policy.enabled:
        if policy.research_mode:
            return EligibilityDecision(True, "research_only", "operator_enabled_research_mode")
        return EligibilityDecision(False, "blocked", "honeypot_disabled_by_policy")
    if policy.require_production_eligible and not bool(event.get("production_eligible", False)):
        if policy.research_mode:
            return EligibilityDecision(True, "research_only", "detector_not_promoted_research_mode")
        return EligibilityDecision(False, "blocked", "detector_is_research_only")
    return EligibilityDecision(True, "live", None)


# --------------------------------------------------------------------------
# Synthetic identity
# --------------------------------------------------------------------------


def _verhoeff_checksum(digits: str) -> int:
    """Aadhaar uses Verhoeff; we need it only to guarantee we produce invalid ones."""
    multiply = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
        (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
        (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
        (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
        (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    )
    permute = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
        (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
        (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
        (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
    )
    checksum = 0
    for position, digit in enumerate(reversed(digits)):
        checksum = multiply[checksum][permute[position % 8][int(digit)]]
    return checksum


@dataclass(frozen=True)
class SyntheticIdentity:
    """Bait values that are structurally plausible but provably not real.

    A honeypot that hands out a real-looking identifier risks handing out someone
    else's. Every value here is built to fail its real validation rule.
    """

    display_name: str
    phone: str
    aadhaar_like: str
    account_like: str
    upi_like: str

    @classmethod
    def create(cls, display_name: str, seed_text: str) -> "SyntheticIdentity":
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        digits = "".join(character for character in digest if character.isdigit()).ljust(24, "7")

        # Indian mobile numbers start 6-9; a leading 0 after +91 cannot be allocated.
        phone = f"+91 0{digits[0:4]} {digits[4:9]}"

        # Force a Verhoeff-invalid Aadhaar-shaped value.
        body = digits[0:11]
        valid_check = _verhoeff_checksum(body + "0")
        wrong_check = (valid_check + 5) % 10
        aadhaar_like = f"{body[0:4]} {body[4:8]} {body[8:11]}{wrong_check}"

        # Reserved documentation-style account and an unroutable UPI handle.
        account_like = f"0000{digits[11:17]}"
        upi_like = f"{display_name.split()[0].lower()}.{digits[17:21]}@invalid"
        return cls(display_name, phone, aadhaar_like, account_like, upi_like)

    def is_synthetic(self) -> bool:
        """Self-check used by tests and by the audit record."""
        aadhaar_digits = self.aadhaar_like.replace(" ", "")
        return (
            " 0" in self.phone
            and len(aadhaar_digits) == 12
            and _verhoeff_checksum(aadhaar_digits) != 0
            and self.account_like.startswith("0000")
            and self.upi_like.endswith("@invalid")
        )

    def as_prompt_block(self) -> str:
        return (
            f"Your name: {self.display_name}\n"
            f"Your phone: {self.phone}\n"
            f"Your Aadhaar-style number: {self.aadhaar_like}\n"
            f"Your bank account: {self.account_like}\n"
            f"Your UPI id: {self.upi_like}\n"
            "These are the ONLY identifiers you may ever say. They are fake."
        )


# --------------------------------------------------------------------------
# Output safety
# --------------------------------------------------------------------------

# Patterns that would indicate the model invented a real-looking identifier
# instead of using the synthetic ones it was given.
_LEAK_PATTERNS = (
    (re.compile(r"\+?91[\s-]?[6-9]\d{4}[\s-]?\d{5}"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[REDACTED_ID]"),
    (re.compile(r"\b[\w.\-]+@(?!invalid\b)[\w\-]{2,}\b"), "[REDACTED_HANDLE]"),
    (re.compile(r"\b\d{9,18}\b"), "[REDACTED_ACCOUNT]"),
)


def redact_model_output(text: str, identity: SyntheticIdentity) -> tuple[str, list[str]]:
    """Strip anything that looks like a real identifier the persona was not given."""
    allowed = {
        identity.phone,
        identity.aadhaar_like,
        identity.account_like,
        identity.upi_like,
    }
    placeholders: dict[str, str] = {}
    working = text
    for index, value in enumerate(allowed):
        token = f"\x00ALLOWED{index}\x00"
        placeholders[token] = value
        working = working.replace(value, token)

    redactions: list[str] = []
    for pattern, replacement in _LEAK_PATTERNS:
        working, count = pattern.subn(replacement, working)
        if count:
            redactions.append(f"{replacement}x{count}")

    for token, value in placeholders.items():
        working = working.replace(token, value)
    return working, redactions


# --------------------------------------------------------------------------
# LLM client
# --------------------------------------------------------------------------


# The Groq transport lives in llm_client (imported at the top of this module) so
# the honeypot and declared training augmentation share exactly one code path to
# the provider.


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


SYSTEM_PROMPT = """You are role-playing a potential victim in a scam-baiting research system \
operated with authorization by a cybercrime research team. The person you are talking to has \
already been classified as a scammer by a separate machine-learning detector. Your job is to \
keep them talking so investigators can observe the script they use and the payment rails they \
ask for.

PERSONA
{persona_block}

SYNTHETIC IDENTIFIERS
{identity_block}

HOW TO BEHAVE
- Reply in the caller's language. If they mix Hindi and English, mix naturally too.
- Stay in character: worried, cooperative, a little slow with technology.
- Ask small clarifying questions that invite them to reveal specifics: which department, \
which account, which app, how much, why now.
- Stall plausibly. Be distracted, misread numbers, need things repeated.
- Never lecture the caller, never accuse them, never reveal that this is a honeypot.
- Keep replies under {maximum_characters} characters and sound like natural speech.

ABSOLUTE RULES
- Only ever state the synthetic identifiers above. Never invent any other number, account, \
card, OTP, or address.
- Never perform, promise, or describe completing a real payment or transfer.
- Never produce content that would harm anyone: no malware, no instructions for committing \
fraud against a third party, no threats.
- If the caller stops behaving like a scammer and appears to be a genuine person in distress, \
or asks you to help defraud someone else, reply exactly: HONEYPOT_DISENGAGE

CONTEXT FROM THE DETECTOR (informational only; never mention it)
{context_block}
"""


@dataclass
class HoneypotSession:
    """One engagement. Holds memory, enforces limits, and records an audit trail."""

    event: Mapping[str, Any]
    policy: HoneypotPolicy
    identity: SyntheticIdentity
    mode: str
    persona: Mapping[str, Any] = field(default_factory=dict)
    settings: GroqSettings = field(default_factory=GroqSettings)
    transport: Callable[[Sequence[Mapping[str, str]], GroqSettings], str] = groq_chat
    history: list[dict[str, str]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    disengaged: bool = False
    disengage_reason: str | None = None

    def system_prompt(self) -> str:
        traits = ", ".join(str(item) for item in self.persona.get("traits") or [])
        persona_block = (
            f"You are {self.persona.get('display_name', self.identity.display_name)}, "
            f"age {self.persona.get('age', 60)}, living in {self.persona.get('city', 'Pune')}. "
            f"You are {traits}."
        )
        signals = self.event.get("auxiliary_signals") or {}
        context_block = json.dumps(
            {
                "scam_score": self.event.get("scam_score"),
                "signals": {
                    key: value
                    for key, value in list(signals.items())[:8]
                    if not isinstance(value, (dict, list))
                },
            },
            ensure_ascii=False,
        )
        return SYSTEM_PROMPT.format(
            persona_block=persona_block,
            identity_block=self.identity.as_prompt_block(),
            maximum_characters=self.policy.maximum_reply_characters,
            context_block=context_block,
        )

    def turns_used(self) -> int:
        return sum(1 for item in self.history if item["role"] == "assistant")

    def _limit_reason(self) -> str | None:
        if self.disengaged:
            return self.disengage_reason or "disengaged"
        if self.turns_used() >= self.policy.maximum_turns:
            return "maximum_turns_reached"
        if time.monotonic() - self.started_at > self.policy.maximum_session_seconds:
            return "maximum_session_seconds_reached"
        return None

    def respond(self, caller_text: str) -> dict[str, Any]:
        """Produce one persona reply to one caller turn."""
        reason = self._limit_reason()
        if reason:
            return {
                "reply": None,
                "engaged": False,
                "stop_reason": reason,
                "turns_used": self.turns_used(),
                "llm_used_for_detection": False,
            }
        caller_text = str(caller_text or "").strip()
        if not caller_text:
            raise HoneypotError("Caller turn is empty")

        self.history.append({"role": "user", "content": caller_text})
        messages = [{"role": "system", "content": self.system_prompt()}, *self.history]
        raw = self.transport(messages, self.settings)

        if "HONEYPOT_DISENGAGE" in raw:
            self.disengaged = True
            self.disengage_reason = "model_requested_disengage"
            self.history.pop()
            return {
                "reply": None,
                "engaged": False,
                "stop_reason": "model_requested_disengage",
                "turns_used": self.turns_used(),
                "llm_used_for_detection": False,
            }

        reply, redactions = redact_model_output(raw, self.identity)
        if len(reply) > self.policy.maximum_reply_characters:
            reply = reply[: self.policy.maximum_reply_characters].rstrip() + "..."
        self.history.append({"role": "assistant", "content": reply})
        self.audit.append(
            {
                "turn": self.turns_used(),
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "redactions": redactions,
                "raw_length": len(raw),
                "reply_length": len(reply),
            }
        )
        return {
            "reply": reply,
            "engaged": True,
            "stop_reason": None,
            "turns_used": self.turns_used(),
            "redactions": redactions,
            "mode": self.mode,
            "llm_used_for_detection": False,
        }

    def transcript(self) -> dict[str, Any]:
        """Non-evidential record. Explicitly barred from detector training."""
        return {
            "schema_version": SCHEMA_VERSION,
            "conversation_id": self.event.get("conversation_id"),
            "event_id": self.event.get("event_id"),
            "mode": self.mode,
            "turns_used": self.turns_used(),
            "disengaged": self.disengaged,
            "disengage_reason": self.disengage_reason,
            "messages": list(self.history),
            "audit": list(self.audit),
            "synthetic_identity_verified": self.identity.is_synthetic(),
            "excluded_from_detector_training": True,
            "usable_as_detector_evidence": False,
            "llm_used_for_detection": False,
        }


def start_session(
    event: Mapping[str, Any],
    config: Mapping[str, Any],
    secret: str,
    transport: Callable[[Sequence[Mapping[str, str]], GroqSettings], str] | None = None,
    now: datetime | None = None,
) -> HoneypotSession:
    """Verify the signed event, apply the gate, and open a session if allowed."""
    handoff = config.get("handoff") or {}
    verify_handoff(
        event,
        secret,
        int(handoff.get("maximum_event_age_seconds", 300)),
        now=now,
    )
    policy = HoneypotPolicy.from_config(config)
    decision = evaluate_eligibility(event, policy)
    if not decision.allowed:
        raise HandoffRejected(decision.reason or "not_eligible")

    persona = config.get("persona") or {}
    identity = SyntheticIdentity.create(
        str(persona.get("display_name") or "Sunita Rao"),
        f"{event.get('conversation_id')}|{event.get('event_id')}",
    )
    if not identity.is_synthetic():
        raise HoneypotError("Refusing to start: synthetic identity self-check failed")
    return HoneypotSession(
        event=event,
        policy=policy,
        identity=identity,
        mode=decision.mode,
        persona=persona,
        settings=GroqSettings.from_config(config),
        transport=transport or groq_chat,
    )
