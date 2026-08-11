"""Interpretable feature contract and XGBoost risk-fusion utilities."""

from __future__ import annotations

from math import log1p
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .entities import entity_type_counts, extract_entities


LEXICAL_TACTICS: dict[str, tuple[str, ...]] = {
    "authority_impersonation": (
        r"\b(?:cbi|rbi|police|customs?|narcotics?|court|officer|inspector|department)\b",
        r"(?:पुलिस|अधिकारी|विभाग|अदालत|कस्टम|सीबीआई|आरबीआई)",
    ),
    "accusation": (
        r"\b(?:crime|criminal|money laundering|illegal|narcotics?|drugs?|arrest warrant|case against)\b",
        r"\b(?:jurm|apraadh|case hua|parcel.*drugs)\b",
        r"(?:अपराध|गिरफ्तारी|मनी लॉन्ड्रिंग|नशीले पदार्थ)",
    ),
    "fear_threat": (
        r"\b(?:arrest|jail|freeze|block(?:ed)?|seize|punishment|legal action|warrant)\b",
        r"\b(?:giraftar|jail bhej|account band|nuksan)\b",
        r"(?:गिरफ्तार|जेल|खाता बंद|कानूनी कार्रवाई)",
    ),
    "urgency": (
        r"\b(?:immediately|right now|urgent|within \d+|today only|last chance|without delay)\b",
        r"\b(?:abhi|turant|jaldi|der mat)\b",
        r"(?:अभी|तुरंत|जल्दी|बिना देरी)",
    ),
    "secrecy": (
        r"\b(?:do not|don'?t|never)\s+(?:tell|inform|share|discuss)\b",
        r"\b(?:kisi ko mat|batana mat|secret rakh)\b",
        r"(?:किसी को मत|मत बताना|गुप्त रख)",
    ),
    "isolation": (
        r"\b(?:stay alone|go to a room|lock the door|keep this call connected|do not disconnect)\b",
        r"\b(?:akele|kamre mein|call mat kat|phone mat rakh)\b",
        r"(?:अकेले|कमरे में|कॉल मत काट|फोन मत रख)",
    ),
    "surveillance_control": (
        r"\b(?:screen share|share your screen|video call|camera on|remote access|anydesk|teamviewer)\b",
        r"\b(?:screen dikhao|camera on rakho)\b",
        r"(?:स्क्रीन शेयर|कैमरा चालू)",
    ),
    "financial_demand": (
        r"\b(?:transfer|deposit|pay|payment|send money|safe account|security amount|processing fee|withdraw)\b",
        r"\b(?:paise bhej|paisa transfer|jama karo|payment karo)\b",
        r"(?:पैसे भेज|ट्रांसफर|जमा करो|भुगतान)",
    ),
    "credential_otp_request": (
        r"\b(?:otp|one time password|pin|cvv|password|verification code|login code)\b",
        r"(?:ओटीपी|पासवर्ड|पिन|सीवीवी)",
    ),
}

ENTITY_FEATURE_TYPES = (
    "upi_id",
    "phone_number",
    "bank_account_candidate",
    "otp_code",
    "aadhaar_number",
    "url",
    "email",
    "monetary_amount",
    "case_reference",
    "authority_organisation",
    "payment_app",
)

RISK_FEATURE_NAMES = (
    "base_scam_score",
    "log_turn_count",
    "log_token_count",
    "log_character_count",
    "digit_fraction",
    "uppercase_fraction",
    "entity_total",
    "sensitive_entity_total",
    *(f"entity_{name}" for name in ENTITY_FEATURE_TYPES),
    *(f"lexical_{name}" for name in LEXICAL_TACTICS),
    "lexical_tactic_total",
    "lexical_stage_progress",
)


def _pattern_hit_count(text: str, patterns: Sequence[str]) -> float:
    return float(sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns))


def _stage_progress(tactic_hits: Mapping[str, float]) -> float:
    stage_values = [0.0]
    if tactic_hits["authority_impersonation"]:
        stage_values.append(1.0 / 6.0)
    if tactic_hits["accusation"]:
        stage_values.append(2.0 / 6.0)
    if tactic_hits["fear_threat"] or tactic_hits["urgency"]:
        stage_values.append(3.0 / 6.0)
    if tactic_hits["isolation"] or tactic_hits["secrecy"] or tactic_hits["surveillance_control"]:
        stage_values.append(4.0 / 6.0)
    if tactic_hits["financial_demand"] or tactic_hits["credential_otp_request"]:
        stage_values.append(5.0 / 6.0)
    return max(stage_values)


def lexical_signal_summary(text: str) -> dict[str, Any]:
    """Expose transparent rule signals without calling them model predictions."""
    raw_counts = {
        name: _pattern_hit_count(text, patterns)
        for name, patterns in LEXICAL_TACTICS.items()
    }
    return {
        "signal_source": "deterministic_lexical_rules",
        "tactics": {
            name: {"present": count > 0, "hit_count": int(count)}
            for name, count in raw_counts.items()
        },
        "stage_progress": _stage_progress(raw_counts),
        "not_transformer_predictions": True,
        "llm_used": False,
    }


def risk_feature_row(text: str, base_scam_score: float) -> np.ndarray:
    if not 0.0 <= float(base_scam_score) <= 1.0:
        raise ValueError("base_scam_score must be in [0, 1]")
    entities = extract_entities(text)
    counts = entity_type_counts(entities)
    tactic_hits = {
        name: min(5.0, _pattern_hit_count(text, patterns)) / 5.0
        for name, patterns in LEXICAL_TACTICS.items()
    }
    characters = [character for character in text if not character.isspace()]
    letters = [character for character in characters if character.isalpha()]
    values = [
        float(base_scam_score),
        log1p(text.count("\n") + (1 if text.strip() else 0)),
        log1p(len(text.split())),
        log1p(len(text)),
        sum(character.isdigit() for character in characters) / max(1, len(characters)),
        sum(character.isupper() for character in letters) / max(1, len(letters)),
        min(10.0, float(len(entities))) / 10.0,
        min(10.0, float(sum(entity.sensitive for entity in entities))) / 10.0,
        *(min(3.0, float(counts.get(name, 0))) / 3.0 for name in ENTITY_FEATURE_TYPES),
        *(tactic_hits[name] for name in LEXICAL_TACTICS),
        sum(value > 0 for value in tactic_hits.values()) / len(tactic_hits),
        _stage_progress(tactic_hits),
    ]
    if len(values) != len(RISK_FEATURE_NAMES):
        raise RuntimeError("Risk feature contract length mismatch")
    return np.asarray(values, dtype=np.float32)


def build_risk_matrix(texts: Sequence[str], base_scores: Sequence[float]) -> np.ndarray:
    if len(texts) != len(base_scores):
        raise ValueError("texts and base_scores must have equal length")
    return np.vstack(
        [risk_feature_row(text, float(score)) for text, score in zip(texts, base_scores)]
    ).astype(np.float32)


def risk_scores(bundle: Mapping[str, Any], feature_matrix: np.ndarray) -> np.ndarray:
    if bundle.get("llm_used_for_detection") is not False:
        raise ValueError("Risk-fusion bundle does not explicitly prohibit LLM detection")
    model = bundle["model"]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Risk-fusion model has no positive class 1")
    return np.asarray(model.predict_proba(feature_matrix)[:, classes.index(1)], dtype=np.float64)
