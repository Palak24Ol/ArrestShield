"""Declared LLM augmentation of the Hinglish training pool.

The corpus holds 633 unique Hinglish scam conversations and only 110 legitimate
ones, so the detector has almost never seen a Hinglish call about a bank account
that is *not* a scam. That imbalance, not the model, is what drives the false
positives on unseen Hinglish.

Everything generated here is marked `provenance: llm_synthetic`, carries a lower
sample weight than real data, and is loaded into the training split only. No
generated row may enter validation or test, and no LLM output is ever a feature,
score, threshold, or label in the detector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from .llm_client import GroqSettings, LLMError, groq_chat

DATASET_ID = "llm_augmentation_hinglish_v1"
PROVENANCE = "llm_synthetic"

SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w.\-]+@[\w\-]+\.\w+")

VALID_STAGES = {
    "none_unknown", "contact", "authority_claim", "accusation",
    "threat", "isolation_control", "payment_extraction",
}
VALID_TACTICS = {
    "authority_impersonation", "phantom_riches", "fear_intimidation", "liking",
    "urgency_scarcity", "pretext_trust", "reciprocity", "consistency_commitment",
    "social_proof", "accusation", "secrecy_instruction", "isolation_instruction",
    "surveillance_control", "financial_demand", "credential_request",
}

CITIES = ["Pune", "Indore", "Lucknow", "Jaipur", "Nagpur", "Patna", "Bhopal",
          "Kochi", "Surat", "Ludhiana", "Ranchi", "Coimbatore"]
CALLEE = ["a 63-year-old retired schoolteacher", "a 28-year-old software tester",
          "a 45-year-old shopkeeper", "a 34-year-old nurse", "a 70-year-old pensioner",
          "a 22-year-old student", "a 52-year-old bank clerk", "a 38-year-old homemaker"]
REGISTERS = ["heavily Hinglish, Roman script", "mostly Hindi in Roman script with English terms",
             "mostly English with Hindi discourse markers", "balanced Hindi-English code-mixing"]

# Positive scenarios. scam_type must exist in configs/data/label_map.json.
SCAM_SCENARIOS = [
    ("digital_arrest", "Caller claims to be CBI/Mumbai Police, says a parcel in the callee's name contained contraband, escalates to a video 'digital arrest' and demands the callee stay on the line and move money to a 'verification' account."),
    ("digital_arrest", "Caller claims to be from TRAI/DoT, says the callee's SIM is used in criminal activity, transfers the call to a fake police officer who threatens arrest."),
    ("fake_kyc_bank", "Caller claims the callee's bank KYC has expired and the account will be frozen today unless they share an OTP and install a 'verification' app."),
    ("courier_customs", "Caller claims to be FedEx/DHL customs, says a package with illegal items was intercepted, then hands off to a fake narcotics officer."),
    ("otp_account_takeover", "Caller poses as bank fraud department warning of an unauthorised transaction and walks the callee through 'reversing' it by reading out an OTP."),
    ("tech_support", "Caller claims to be from a telecom or Microsoft support desk, says the callee's device is infected, asks them to install AnyDesk."),
    ("loan_job", "Caller offers a pre-approved loan or a work-from-home job requiring a refundable registration fee paid by UPI."),
]

# Hard negatives: legitimate calls that share the scam vocabulary. This is the
# class the corpus is missing almost entirely.
LEGITIMATE_SCENARIOS = [
    "A genuine bank customer-care agent calls about a card transaction the customer actually made, verifies identity WITHOUT asking for an OTP or password, and explicitly says the bank will never ask for an OTP.",
    "A real courier delivery agent calls to confirm the address and a convenient delivery time for a parcel the callee genuinely ordered.",
    "A genuine police station clerk calls about a routine passport verification appointment and asks the callee to visit the station in person with documents.",
    "The callee calls their bank's helpline themselves to ask why a UPI payment failed; the agent explains the refund timeline.",
    "A telecom customer-service agent calls about a plan renewal and a network outage in the callee's area.",
    "A hospital billing desk calls about an insurance claim document that is missing a signature.",
    "A genuine bank branch manager calls to ask the callee to come to the branch to complete KYC in person, and refuses to take any details over the phone.",
    "An electricity board agent calls about a genuinely pending bill and directs the callee to the official app or a counter, declining to take payment on the call.",
    "A recruiter from a real company calls to schedule an interview and confirms no fee is required at any stage.",
    "A delivery platform support agent calls about a refund for an order that was cancelled.",
    "A family member calls about a money transfer they are genuinely making, discussing account numbers and UPI naturally.",
    "A school administrator calls a parent about pending fee payment, directing them to the official portal.",
]

SYSTEM_PROMPT = """You produce realistic Indian phone-call transcripts for a cybercrime \
detection research dataset at an Indian university. The dataset trains a classifier that \
protects people from "digital arrest" scams.

Return ONLY a JSON object, no prose, with this exact shape:
{"turns": [{"speaker_role": "caller"|"callee", "text": "...", "stage": "...", "tactics": ["..."]}]}

Rules for every transcript:
- Write natural spoken Hinglish in Roman script the way Indians actually speak on the phone.
  Include fillers, interruptions, repetitions, "haan", "achha", "sir/madam", "arre".
- Do NOT write stage directions, narration, or speaker names inside "text".
- Use only clearly fake identifiers: phone numbers starting +91 0, accounts starting 0000,
  UPI handles ending @invalid. Never write a real-looking bank account, card, or Aadhaar.
- "stage" must be one of: none_unknown, contact, authority_claim, accusation, threat,
  isolation_control, payment_extraction.
- "tactics" must be a possibly-empty subset of: authority_impersonation, phantom_riches,
  fear_intimidation, liking, urgency_scarcity, pretext_trust, reciprocity,
  consistency_commitment, social_proof, accusation, secrecy_instruction,
  isolation_instruction, surveillance_control, financial_demand, credential_request.
- For LEGITIMATE calls every turn must have stage "none_unknown" and an empty tactics list.
"""


@dataclass(frozen=True)
class GenerationRequest:
    is_scam: int
    scam_type: str
    scenario: str
    city: str
    callee: str
    register: str
    turn_target: int

    def user_prompt(self) -> str:
        if self.is_scam:
            framing = (
                f"Write a SCAM call transcript. Scenario: {self.scenario}\n"
                "Label each turn's stage and tactics honestly, following the escalation as it happens. "
                "Early turns should NOT carry late-stage labels."
            )
        else:
            framing = (
                f"Write a LEGITIMATE, non-scam call transcript. Scenario: {self.scenario}\n"
                "This call must be genuinely safe, but it must naturally use the same vocabulary a scam "
                "would use (bank, account, KYC, verification, police, courier, payment, Aadhaar, OTP) so "
                "that a classifier cannot separate scam from legitimate on keywords alone. "
                "The legitimate party must never ask for an OTP, password, or a transfer to a 'safe account'."
            )
        return (
            f"{framing}\n\n"
            f"Callee is {self.callee} in {self.city}. Language register: {self.register}. "
            f"Write about {self.turn_target} turns, alternating caller and callee."
        )


def build_requests(
    scam_count: int, legitimate_count: int, seed: int = 42
) -> list[GenerationRequest]:
    """Spread requests across scenarios, cities, personas, and registers."""
    rng = random.Random(seed)
    requests: list[GenerationRequest] = []
    for index in range(scam_count):
        scam_type, scenario = SCAM_SCENARIOS[index % len(SCAM_SCENARIOS)]
        requests.append(
            GenerationRequest(
                is_scam=1,
                scam_type=scam_type,
                scenario=scenario,
                city=rng.choice(CITIES),
                callee=rng.choice(CALLEE),
                register=rng.choice(REGISTERS),
                turn_target=rng.choice([8, 10, 12, 14, 16]),
            )
        )
    for index in range(legitimate_count):
        requests.append(
            GenerationRequest(
                is_scam=0,
                scam_type="non_scam",
                scenario=LEGITIMATE_SCENARIOS[index % len(LEGITIMATE_SCENARIOS)],
                city=rng.choice(CITIES),
                callee=rng.choice(CALLEE),
                register=rng.choice(REGISTERS),
                turn_target=rng.choice([6, 8, 10, 12]),
            )
        )
    rng.shuffle(requests)
    return requests


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("​", "").replace("﻿", "")
    return SPACE_RE.sub(" ", text).strip()


def fingerprint(turns: Sequence[Mapping[str, Any]], template: bool = False) -> str:
    pieces = []
    for turn in turns:
        text = str(turn["normalized_text"]).lower()
        if template:
            text = URL_RE.sub("[url]", text)
            text = EMAIL_RE.sub("[email]", text)
            text = NUMBER_RE.sub("[number]", text)
        pieces.append(f"{turn['speaker_role']}:{text}")
    return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()


def parse_response(payload: str) -> list[dict[str, Any]]:
    """Extract the turn list, tolerating a fenced or prefixed reply."""
    text = payload.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model response")
    parsed = json.loads(text[start : end + 1])
    turns = parsed.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("Response contains no turns")
    return turns


def build_record(
    request: GenerationRequest, raw_turns: Sequence[Mapping[str, Any]], index: int
) -> dict[str, Any] | None:
    """Validate and convert one generation into a canonical conversation record."""
    turns: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_turns):
        body = normalize_text(str(raw.get("text") or ""))
        if not body:
            continue
        role = str(raw.get("speaker_role") or "").strip().lower()
        if role not in {"caller", "callee"}:
            role = "caller" if position % 2 == 0 else "callee"

        stage = str(raw.get("stage") or "none_unknown")
        if stage not in VALID_STAGES or not request.is_scam:
            stage = "none_unknown" if not request.is_scam else "contact"
        raw_tactics = raw.get("tactics") or []
        tactics = {
            str(name): 1
            for name in raw_tactics
            if isinstance(name, str) and name in VALID_TACTICS
        } if request.is_scam else {}

        turns.append(
            {
                "turn_id": len(turns),
                "speaker_role": role,
                "raw_text": body,
                "normalized_text": body,
                "language": "hinglish",
                "labels": {
                    "is_scam": request.is_scam,
                    "scam_type": request.scam_type,
                    "tactics": tactics,
                    "stage": stage,
                    "provenance": PROVENANCE,
                },
            }
        )
    if len(turns) < 4:
        return None

    exact = fingerprint(turns)
    return {
        "schema_version": "1.0.0",
        "conversation_id": f"{DATASET_ID}-{exact[:16]}",
        "source": {
            "dataset_id": DATASET_ID,
            "revision": "v1",
            "license": "Project-owned; generated text, not redistributed source data",
            "record_id": f"gen-{index:05d}",
            "original_split": None,
            "source_type": "llm_generated",
        },
        "language_profile": ["hinglish", "hindi", "english"],
        "conversation_label": {
            "is_scam": request.is_scam,
            "scam_type": request.scam_type,
            "provenance": PROVENANCE,
        },
        "turns": turns,
        "fingerprints": {
            "exact_sha256": exact,
            "template_sha256": fingerprint(turns, template=True),
        },
        "quality": {
            "training_eligible": True,
            "exclusion_reasons": [],
            "label_quality": PROVENANCE,
            "generation": {
                "scenario": request.scenario,
                "city": request.city,
                "callee": request.callee,
                "register": request.register,
            },
        },
        "split_policy": {
            "allowed_splits": ["train"],
            "never_validation_or_test": True,
            "reason": "LLM-generated rows cannot serve as evidence about real-world performance.",
        },
    }


def generate(
    requests: Iterable[GenerationRequest],
    settings: GroqSettings,
    existing_templates: set[str] | None = None,
    transport=groq_chat,
    on_progress=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate records, dropping malformed and duplicate conversations."""
    seen = set(existing_templates or set())
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, request in enumerate(requests):
        try:
            reply = transport(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": request.user_prompt()},
                ],
                settings,
                {"type": "json_object"},
            )
            record = build_record(request, parse_response(reply), index)
        except (LLMError, ValueError, json.JSONDecodeError) as error:
            failures.append({"index": index, "is_scam": request.is_scam, "error": str(error)[:200]})
            if on_progress:
                on_progress(index, None, str(error)[:80])
            continue
        if record is None:
            failures.append({"index": index, "is_scam": request.is_scam, "error": "too few usable turns"})
            if on_progress:
                on_progress(index, None, "too few turns")
            continue
        template = record["fingerprints"]["template_sha256"]
        if template in seen:
            failures.append({"index": index, "is_scam": request.is_scam, "error": "duplicate template"})
            if on_progress:
                on_progress(index, None, "duplicate")
            continue
        seen.add(template)
        records.append(record)
        if on_progress:
            on_progress(index, record, None)
    return records, failures


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scam = [row for row in records if row["conversation_label"]["is_scam"] == 1]
    legitimate = [row for row in records if row["conversation_label"]["is_scam"] == 0]
    by_type: dict[str, int] = {}
    for row in scam:
        key = str(row["conversation_label"]["scam_type"])
        by_type[key] = by_type.get(key, 0) + 1
    turn_counts = [len(row["turns"]) for row in records]
    return {
        "total": len(records),
        "scam": len(scam),
        "legitimate": len(legitimate),
        "scam_types": dict(sorted(by_type.items())),
        "unique_templates": len({row["fingerprints"]["template_sha256"] for row in records}),
        "mean_turns": round(sum(turn_counts) / len(turn_counts), 2) if turn_counts else 0,
        "provenance": PROVENANCE,
        "allowed_splits": ["train"],
    }
