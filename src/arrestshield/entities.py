"""Deterministic, privacy-aware threat-entity extraction for scam transcripts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ExtractedEntity:
    entity_type: str
    start: int
    end: int
    value: str
    normalized_value: str
    redacted_value: str
    confidence: float
    sensitive: bool

    def public_dict(self, include_sensitive_values: bool = False) -> dict[str, object]:
        visible = self.value if include_sensitive_values or not self.sensitive else self.redacted_value
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "value": visible,
            "normalized_value": (
                self.normalized_value
                if include_sensitive_values or not self.sensitive
                else self.redacted_value
            ),
            "confidence": self.confidence,
            "sensitive": self.sensitive,
            "redacted": bool(self.sensitive and not include_sensitive_values),
        }


UPI_HANDLES = {
    "upi", "ybl", "ibl", "axl", "paytm", "okaxis", "okhdfcbank", "okicici",
    "oksbi", "apl", "ptyes", "pthdfc", "pingpay", "waaxis", "wahdfcbank",
    "waicici", "wasbi", "naviaxis", "navihdfc", "navisbi",
}

AUTHORITY_PATTERNS = {
    "CBI": r"\b(?:cbi|central bureau of investigation)\b",
    "RBI": r"\b(?:rbi|reserve bank of india)\b",
    "Police": r"\b(?:police|पुलिस)\b",
    "Cyber Crime": r"\b(?:cyber\s*crime|cyber\s*cell|साइबर\s*क्राइम)\b",
    "Customs": r"\b(?:customs?|custom department|कस्टम)\b",
    "Enforcement Directorate": r"\b(?:enforcement directorate|ईडी|ed officer)\b",
    "Narcotics Bureau": r"\b(?:narcotics?|ncb officer|नारकोटिक्स)\b",
    "Supreme Court": r"\b(?:supreme court|सुप्रीम कोर्ट)\b",
    "Income Tax": r"\b(?:income tax|आयकर)\b",
}

PAYMENT_APP_PATTERNS = {
    "PhonePe": r"\bphone\s*pe\b",
    "Google Pay": r"\b(?:google\s*pay|gpay)\b",
    "Paytm": r"\bpaytm\b",
    "BHIM": r"\bbhim\b",
    "WhatsApp Pay": r"\bwhatsapp\s*pay\b",
}


def _masked_tail(value: str, visible: int = 4) -> str:
    digits = re.sub(r"\D", "", value)
    return f"***{digits[-visible:]}" if digits else "[REDACTED]"


def _context(text: str, start: int, end: int, radius: int = 45) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    for boundary in ".!?;\n":
        previous = text.rfind(boundary, left, start)
        if previous >= 0:
            left = max(left, previous + 1)
        following = text.find(boundary, end, right)
        if following >= 0:
            right = min(right, following)
    return text[left:right].casefold()


def _candidate(
    entity_type: str,
    match: re.Match[str],
    normalized: str,
    redacted: str,
    confidence: float,
    sensitive: bool,
) -> ExtractedEntity:
    return ExtractedEntity(
        entity_type=entity_type,
        start=match.start(),
        end=match.end(),
        value=match.group(0),
        normalized_value=normalized,
        redacted_value=redacted,
        confidence=confidence,
        sensitive=sensitive,
    )


def _overlaps(left: ExtractedEntity, right: ExtractedEntity) -> bool:
    return left.start < right.end and right.start < left.end


def _deduplicate(candidates: Iterable[ExtractedEntity]) -> list[ExtractedEntity]:
    priority = {
        "otp_code": 100,
        "aadhaar_number": 95,
        "phone_number": 90,
        "bank_account_candidate": 85,
        "upi_id": 80,
        "email": 75,
        "url": 70,
        "case_reference": 60,
        "monetary_amount": 55,
        "authority_organisation": 40,
        "payment_app": 35,
    }
    retained: list[ExtractedEntity] = []
    ordered = sorted(
        candidates,
        key=lambda item: (-priority.get(item.entity_type, 0), -item.confidence, item.start),
    )
    for entity in ordered:
        if any(_overlaps(entity, existing) for existing in retained):
            continue
        retained.append(entity)
    return sorted(retained, key=lambda item: (item.start, item.end, item.entity_type))


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract operational entities without using an LLM or network service."""
    candidates: list[ExtractedEntity] = []

    url_pattern = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
    for match in url_pattern.finditer(text):
        raw = match.group(0).rstrip(".,;:!?)]}")
        start, end = match.start(), match.start() + len(raw)
        parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
        hostname = parsed.hostname or "unknown-host"
        candidates.append(
            ExtractedEntity("url", start, end, raw, raw, hostname, 0.98, True)
        )

    email_pattern = re.compile(r"(?i)(?<![\w.-])[\w.+-]{1,64}@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
    for match in email_pattern.finditer(text):
        local, domain = match.group(0).rsplit("@", 1)
        candidates.append(
            _candidate("email", match, f"{local.casefold()}@{domain.casefold()}", f"***@{domain.casefold()}", 0.99, True)
        )

    upi_pattern = re.compile(
        r"(?i)(?<![\w.-])[a-z0-9._-]{2,100}@([a-z][a-z0-9]{1,30})(?![\w-]|\.[a-z0-9])"
    )
    for match in upi_pattern.finditer(text):
        handle = match.group(1).casefold()
        if handle not in UPI_HANDLES:
            continue
        candidates.append(
            _candidate("upi_id", match, match.group(0).casefold(), f"***@{handle}", 0.99, True)
        )

    phone_pattern = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9](?:[-\s]?\d){9}(?!\d)")
    for match in phone_pattern.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        normalized = f"+91{digits[-10:]}"
        candidates.append(
            _candidate("phone_number", match, normalized, _masked_tail(digits), 0.97, True)
        )

    number_pattern = re.compile(r"(?<!\d)\d(?:[ -]?\d){3,17}(?!\d)")
    for match in number_pattern.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        context = _context(text, match.start(), match.end())
        if 4 <= len(digits) <= 8 and re.search(r"\b(?:otp|pin|cvv|code|पासकोड|ओटीपी)\b", context):
            candidates.append(_candidate("otp_code", match, digits, "[REDACTED OTP]", 0.96, True))
        elif len(digits) == 12 and re.search(r"\b(?:aadhaar|aadhar|आधार)\b", context):
            candidates.append(_candidate("aadhaar_number", match, digits, _masked_tail(digits), 0.97, True))
        elif 9 <= len(digits) <= 18 and re.search(
            r"\b(?:account|acct|a/c|bank|खाता|अकाउंट)\b", context
        ):
            candidates.append(
                _candidate("bank_account_candidate", match, digits, _masked_tail(digits), 0.82, True)
            )

    amount_patterns = [
        re.compile(r"(?i)(?:₹|\brs\.?|\binr\b)\s*[0-9][0-9,]*(?:\.\d{1,2})?"),
        re.compile(r"(?i)\b[0-9][0-9,]*(?:\.\d{1,2})?\s*(?:rupees?|रुपये)\b"),
    ]
    for pattern in amount_patterns:
        for match in pattern.finditer(text):
            digits = re.sub(r"[^0-9.]", "", match.group(0).replace(",", ""))
            candidates.append(
                _candidate("monetary_amount", match, digits, match.group(0), 0.95, False)
            )

    case_pattern = re.compile(
        r"(?i)\b(?:case|fir|complaint|reference)\s*(?:no\.?|number|id|#)?\s*[:#-]?\s*([a-z0-9/-]{5,30})"
    )
    for match in case_pattern.finditer(text):
        candidates.append(
            _candidate("case_reference", match, match.group(1).upper(), "[REDACTED CASE]", 0.84, True)
        )

    for normalized, pattern in AUTHORITY_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append(
                _candidate("authority_organisation", match, normalized, normalized, 0.90, False)
            )
    for normalized, pattern in PAYMENT_APP_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidates.append(_candidate("payment_app", match, normalized, normalized, 0.94, False))

    return _deduplicate(candidates)


def entity_type_counts(entities: Sequence[ExtractedEntity]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entity in entities:
        counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
    return counts
