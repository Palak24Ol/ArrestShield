"""Build canonical, deduplicated ArrestShield conversation data and group splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

SCHEMA_VERSION = "1.0.0"
ROLE_RE = re.compile(r"(?i)(?:^|\s)(caller|receiver|innocent|suspect)\s*:\s*")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
UPI_RE = re.compile(r"(?i)\b[a-z0-9._-]{2,}@[a-z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
TITLE_NAME_RE = re.compile(r"(?i)\b(?:mr|mrs|ms|miss|officer|agent|doctor|dr)\.?\s+[a-z]+\b")

TACTICS = [
    "authority_impersonation", "accusation", "fear_threat", "urgency",
    "secrecy", "isolation", "surveillance_control", "financial_demand",
    "credential_otp_request",
]
SCAM_TYPES = {
    "non_scam", "digital_arrest", "fake_kyc_bank", "courier_customs",
    "otp_account_takeover", "investment", "loan_job", "tech_support",
    "impersonation_other", "other_scam",
}
STAGES = {
    "contact", "authority_claim", "accusation", "threat", "isolation_control",
    "payment_extraction", "post_payment", "none_unknown",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return SPACE_RE.sub(" ", text).strip()


def fingerprint_text(turns: list[dict], template: bool = False) -> str:
    pieces = []
    for turn in turns:
        text = turn["normalized_text"].lower()
        if template:
            text = URL_RE.sub("[url]", text)
            text = EMAIL_RE.sub("[email]", text)
            text = PHONE_RE.sub("[phone]", text)
            text = UPI_RE.sub("[upi]", text)
            text = TITLE_NAME_RE.sub("[person]", text)
            text = NUMBER_RE.sub("[number]", text)
        pieces.append(f"{turn['speaker_role']}:{text}")
    return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()


def simhash64(turns: list[dict]) -> int:
    """Similarity fingerprint over unigram/bigram content for synthetic grouping."""
    text = " ".join(t["normalized_text"].lower() for t in turns)
    text = URL_RE.sub(" [url] ", text); text = EMAIL_RE.sub(" [email] ", text)
    text = PHONE_RE.sub(" [phone] ", text); text = UPI_RE.sub(" [upi] ", text)
    text = TITLE_NAME_RE.sub(" [person] ", text); text = NUMBER_RE.sub(" [number] ", text)
    tokens = re.findall(r"[\w\[\]]+", text, flags=re.UNICODE)
    features = Counter(tokens)
    features.update(" ".join(tokens[i:i + 2]) for i in range(max(0, len(tokens) - 1)))
    vector = [0] * 64
    for feature, weight in features.items():
        value = int(hashlib.sha256(feature.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            vector[bit] += weight if value & (1 << bit) else -weight
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def stable_id(dataset_id: str, source_record_id: str) -> str:
    suffix = hashlib.sha256(f"{dataset_id}|{source_record_id}".encode()).hexdigest()[:20]
    return f"{dataset_id}:{suffix}"


def split_role_dialogue(text: str, style: str) -> list[tuple[str, str]]:
    matches = list(ROLE_RE.finditer(text))
    if not matches:
        return [("unknown", normalize_text(text))]
    turns: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw = normalize_text(text[start:end])
        if not raw:
            continue
        role = match.group(1).lower()
        role = {"caller": "caller", "receiver": "recipient", "suspect": "caller", "innocent": "recipient"}[role]
        turns.append((role, raw))
    return turns or [("unknown", normalize_text(text))]


def scam_type_from_text(text: str, source_category: str | None = None) -> str:
    t = normalize_text(text).lower()
    if any(x in t for x in ["digital arrest", "arrest warrant", "money laundering", "narcotics", "cbi", "cyber crime", "police station"]):
        return "digital_arrest"
    if any(x in t for x in ["kyc", "rbi", "bank account", "account block", "atm card", "debit card"]):
        return "otp_account_takeover" if any(x in t for x in ["otp", "cvv", "pin share"]) else "fake_kyc_bank"
    if any(x in t for x in ["parcel", "courier", "customs", "fedex", "delivery charge", "amazon order"]):
        return "courier_customs"
    if any(x in t for x in ["investment", "trading", "guaranteed return", "double your money", "crypto profit"]):
        return "investment"
    if any(x in t for x in ["job offer", "registration fee", "instant loan", "loan approval"]):
        return "loan_job"
    if any(x in t for x in ["technical support", "remote access", "your computer", "microsoft support"]):
        return "tech_support"
    if source_category in {"police_digital_arrest", "police_blackmail", "aadhaar"}:
        return "digital_arrest"
    if source_category == "bank_kyc":
        return "fake_kyc_bank"
    if source_category == "amazon":
        return "courier_customs"
    if source_category == "relative":
        return "impersonation_other"
    return "other_scam"


def weak_tactics(text: str) -> dict[str, int | None]:
    t = normalize_text(text).lower()
    patterns = {
        "authority_impersonation": ["officer", "police", "cbi", "rbi", "social security administration", "cyber cell", "customs officer", "bank manager"],
        "accusation": ["money laundering", "illegal transaction", "illegal activity", "narcotics", "fraudulent activity", "case against", "used in fraud"],
        "fear_threat": ["arrest", "warrant", "jail", "legal action", "blackmail", "severe consequence", "suspended", "account block", "account freeze", "penalty"],
        "urgency": ["urgent", "immediately", "right now", "act now", "24 hours", "2 ghante", "abhi", "turant", "jaldi", "time is of the essence", "every minute"],
        "secrecy": ["do not tell", "don't tell", "do not inform", "kisi ko mat", "secret", "confidential"],
        "isolation": ["stay on the line", "call disconnect mat", "do not disconnect", "don't hang up", "video call par raho"],
        "surveillance_control": ["screen share", "remote access", "download this app", "install app", "video call"],
        "financial_demand": ["transfer", "safe account", "pay ", "payment", "gift card", "security deposit", "processing fee", "₹", "rs."],
        "credential_otp_request": ["otp", "cvv", "password", "pin share", "social security number", "account details", "card number", "confirm your identity"],
    }
    return {label: (1 if any(p in t for p in values) else None) for label, values in patterns.items()}


def weak_stage(tactics: dict[str, int | None], turn_id: int) -> str:
    if tactics.get("financial_demand") or tactics.get("credential_otp_request"):
        return "payment_extraction"
    if tactics.get("isolation") or tactics.get("secrecy") or tactics.get("surveillance_control"):
        return "isolation_control"
    if tactics.get("fear_threat"):
        return "threat"
    if tactics.get("accusation"):
        return "accusation"
    if tactics.get("authority_impersonation"):
        return "authority_claim"
    return "contact" if turn_id == 0 else "none_unknown"


def negative_labels() -> dict:
    return {"is_scam": 0, "scam_type": "non_scam", "tactics": {k: 0 for k in TACTICS}, "stage": "none_unknown", "provenance": "source_silver"}


def positive_weak_labels(text: str, scam_type: str, turn_id: int) -> dict:
    tactics = weak_tactics(text)
    return {"is_scam": 1, "scam_type": scam_type, "tactics": tactics, "stage": weak_stage(tactics, turn_id), "provenance": "weak_rule"}


def unlabeled_turn() -> dict:
    return {"is_scam": None, "scam_type": None, "tactics": {k: None for k in TACTICS}, "stage": None, "provenance": "unlabeled"}


def build_turns(items: list[tuple[str, str]], language: str, conv_is_scam: int | None, scam_type: str | None, allow_weak_positive: bool, allow_negative: bool) -> list[dict]:
    turns = []
    for turn_id, (role, raw) in enumerate(items):
        normalized = normalize_text(raw)
        if not normalized:
            continue
        if conv_is_scam == 0 and allow_negative:
            labels = negative_labels()
        elif conv_is_scam == 1 and allow_weak_positive and role in {"caller", "sender", "unknown"}:
            labels = positive_weak_labels(normalized, scam_type or "other_scam", turn_id)
        else:
            labels = unlabeled_turn()
        turns.append({"turn_id": len(turns), "speaker_role": role, "raw_text": raw, "normalized_text": normalized, "language": language, "labels": labels})
    return turns


def csv_rows(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            yield index, row


def source_info(registry: dict, dataset_id: str, record_id: str, original_split: str | None, source_type: str) -> dict:
    item = registry[dataset_id]
    return {"dataset_id": dataset_id, "repository": item["repository"], "revision": item["revision"], "license": item["license"], "record_id": record_id, "original_split": original_split, "source_type": source_type}


def raw_conversation(registry: dict, dataset_id: str, record_id: str, original_split: str | None, source_type: str, language: str, is_scam: int | None, scam_type: str | None, provenance: str, turns: list[dict], metadata: dict) -> dict:
    if not turns:
        raise ValueError(f"No turns for {dataset_id}/{record_id}")
    return {
        "schema_version": SCHEMA_VERSION,
        "conversation_id": stable_id(dataset_id, record_id),
        "source": source_info(registry, dataset_id, record_id, original_split, source_type),
        "language_profile": [language],
        "conversation_label": {"is_scam": is_scam, "scam_type": scam_type, "provenance": provenance},
        "turns": turns,
        "metadata": metadata,
    }


def parse_indian_scam(raw: Path, registry: dict) -> Iterator[dict]:
    dataset_id = "indian_cyber_scam_phonecall_hinglish"
    path = raw / dataset_id / "India_Cyber_Scam_Hinglish_Dataset.csv"
    for index, row in csv_rows(path):
        is_scam = int(row["label"])
        scam_type = scam_type_from_text(row["text"], row.get("scam_category")) if is_scam else "non_scam"
        turns = build_turns([("caller", row["text"])], "hinglish", is_scam, scam_type, True, True)
        yield raw_conversation(registry, dataset_id, str(index), None, "curated_mixed_unknown", "hinglish", is_scam, scam_type, "source_silver", turns, {k: v for k, v in row.items() if k != "text"})


def parse_indian_messages(raw: Path, registry: dict) -> Iterator[dict]:
    dataset_id = "indian_multilingual_scam_messages"
    path = raw / dataset_id / "ultra_premium_scam_dataset.csv"
    for index, row in csv_rows(path):
        is_scam = 1 if row["label"].strip().lower() == "scam" else 0
        scam_type = scam_type_from_text(row["message"]) if is_scam else "non_scam"
        language = row.get("language", "unknown").strip().lower()
        turns = build_turns([("sender", row["message"])], language, is_scam, scam_type, True, True)
        yield raw_conversation(registry, dataset_id, str(index), None, "real_world_inspired_unknown", language, is_scam, scam_type, "source_silver", turns, {k: v for k, v in row.items() if k != "message"})


def synthetic_type(label: int, kind: str) -> str:
    if not label:
        return "non_scam"
    return {"ssn": "impersonation_other", "refund": "other_scam", "support": "tech_support", "reward": "other_scam"}.get(kind, "other_scam")


def parse_synthetic_dialogues(raw: Path, registry: dict, dataset_id: str, filename: str, label_field: str, style: str) -> Iterator[dict]:
    path = raw / dataset_id / filename
    for index, row in csv_rows(path):
        is_scam = int(row[label_field])
        scam_type = synthetic_type(is_scam, row["type"])
        items = split_role_dialogue(row["dialogue"], style)
        turns = build_turns(items, "english", is_scam, scam_type, True, True)
        metadata = {k: v for k, v in row.items() if k != "dialogue"}
        yield raw_conversation(registry, dataset_id, str(index), None, "synthetic_llm", "english", is_scam, scam_type, "source_silver", turns, metadata)


def parse_banking77(raw: Path, registry: dict) -> Iterator[dict]:
    dataset_id = "banking77"
    for split in ("train", "test"):
        for index, row in csv_rows(raw / dataset_id / f"{split}.csv"):
            turns = build_turns([("customer", row["text"])], "english", 0, "non_scam", False, True)
            yield raw_conversation(registry, dataset_id, f"{split}:{index}", split, "expert_generated_query", "english", 0, "non_scam", "source_gold", turns, {"intent": row["category"]})


def parse_daily_dialog(raw: Path, registry: dict) -> Iterator[dict]:
    dataset_id = "daily_dialog"
    dataset_dir = raw / dataset_id
    for split in ("train", "validation", "test"):
        archive = dataset_dir / f"{split}.zip"
        with zipfile.ZipFile(archive) as zf:
            candidates = [n for n in zf.namelist() if Path(n).name.startswith("dialogues_") and "_act_" not in n and "_emotion_" not in n]
            if len(candidates) != 1:
                raise ValueError(f"Unexpected DailyDialog archive: {archive}: {candidates}")
            with zf.open(candidates[0]) as source:
                for index, line in enumerate(io.TextIOWrapper(source, encoding="utf-8")):
                    utterances = [normalize_text(x) for x in line.split("__eou__") if normalize_text(x)]
                    items = [("speaker_a" if i % 2 == 0 else "speaker_b", text) for i, text in enumerate(utterances)]
                    turns = build_turns(items, "english", 0, "non_scam", False, True)
                    yield raw_conversation(registry, dataset_id, f"{split}:{index}", split, "human_written_dialogue", "english", 0, "non_scam", "source_gold", turns, {})


def parse_sgd(raw: Path, registry: dict) -> Iterator[dict]:
    dataset_id = "schema_guided_dialogue"
    archive = raw / dataset_id / "repository.zip"
    with zipfile.ZipFile(archive) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".json") and "/dialogues_" in n and "/sgd_x/" not in n.lower())
        for name in names:
            parts = Path(name).parts
            split = next((p for p in parts if p in {"train", "dev", "test"}), "unknown")
            records = json.loads(zf.read(name))
            if not isinstance(records, list):
                continue
            for record in records:
                record_id = f"{split}:{record['dialogue_id']}"
                items = []
                for turn in record.get("turns", []):
                    role = "customer" if turn.get("speaker") == "USER" else "service_agent"
                    items.append((role, turn.get("utterance", "")))
                turns = build_turns(items, "english", 0, "non_scam", False, True)
                yield raw_conversation(registry, dataset_id, record_id, split, "simulated_crowdworker_dialogue", "english", 0, "non_scam", "source_gold", turns, {"services": record.get("services", [])})


def load_registry(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in raw["datasets"]}


def conversations(raw: Path, registry: dict) -> Iterable[dict]:
    yield from parse_indian_scam(raw, registry)
    yield from parse_indian_messages(raw, registry)
    yield from parse_synthetic_dialogues(raw, registry, "synthetic_scam_dialogue", "scam-dialogue_all.csv", "label", "caller_receiver")
    yield from parse_synthetic_dialogues(raw, registry, "synthetic_multi_agent_scam_conversation", "agent_conversation_all.csv", "labels", "suspect_innocent")
    yield from parse_banking77(raw, registry)
    yield from parse_daily_dialog(raw, registry)
    yield from parse_sgd(raw, registry)


def assign_split(template_hash: str) -> str:
    value = int(hashlib.sha256(f"arrestshield-split-v1|{template_hash}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < 0.70 else ("validation" if value < 0.85 else "test")


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def validate_conversation(record: dict) -> list[str]:
    errors = []
    if record["schema_version"] != SCHEMA_VERSION:
        errors.append("schema_version")
    if not record["turns"]:
        errors.append("no_turns")
    if [t["turn_id"] for t in record["turns"]] != list(range(len(record["turns"]))):
        errors.append("turn_ids")
    if any(not t["normalized_text"] for t in record["turns"]):
        errors.append("empty_text")
    label = record["conversation_label"]
    if label["scam_type"] not in SCAM_TYPES:
        errors.append("scam_type")
    for turn in record["turns"]:
        if turn["labels"]["stage"] is not None and turn["labels"]["stage"] not in STAGES:
            errors.append("stage")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve(); raw = root / "data" / "raw"; processed = root / "data" / "processed"; splits_dir = root / "data" / "splits"; docs = root / "docs" / "datasets"
    processed.mkdir(parents=True, exist_ok=True); splits_dir.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)
    registry = load_registry(root / "data" / "manifests" / "dataset_registry.json")
    all_path = processed / "conversations.all.jsonl"; kept_path = processed / "conversations.jsonl"; turns_path = processed / "turns.jsonl"
    exact_seen: dict[str, str] = {}; template_members: defaultdict[str, list[str]] = defaultdict(list); eligible: dict[str, dict] = {}; stats = Counter(); sources = Counter(); labels = Counter(); validation_errors = Counter(); exact_duplicate_sources = Counter()
    with all_path.open("w", encoding="utf-8") as all_out, kept_path.open("w", encoding="utf-8") as kept_out, turns_path.open("w", encoding="utf-8") as turns_out:
        for record in conversations(raw, registry):
            stats["input_conversations"] += 1; sources[record["source"]["dataset_id"]] += 1; labels[str((record["conversation_label"]["is_scam"], record["conversation_label"]["scam_type"]))] += 1
            exact = fingerprint_text(record["turns"], template=False); template = fingerprint_text(record["turns"], template=True)
            similarity = simhash64(record["turns"]) if record["source"]["dataset_id"] in {"synthetic_scam_dialogue", "synthetic_multi_agent_scam_conversation"} else int(template[:16], 16)
            record["fingerprints"] = {"exact_sha256": exact, "template_sha256": template, "simhash64": f"{similarity:016x}"}
            reasons = []
            if record["conversation_label"]["is_scam"] not in {0, 1}:
                reasons.append("missing_binary_label")
            if exact in exact_seen:
                reasons.append("exact_duplicate")
                record["duplicate_of"] = exact_seen[exact]
                exact_duplicate_sources[record["source"]["dataset_id"]] += 1
            else:
                exact_seen[exact] = record["conversation_id"]
            errors = validate_conversation(record)
            if errors:
                reasons.extend(f"validation:{e}" for e in errors); validation_errors.update(errors)
            record["quality"] = {"training_eligible": not reasons, "exclusion_reasons": reasons, "label_quality": record["conversation_label"]["provenance"]}
            all_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if reasons:
                stats["excluded_conversations"] += 1
                continue
            kept_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["kept_conversations"] += 1; stats["kept_turns"] += len(record["turns"])
            template_members[template].append(record["conversation_id"])
            eligible[record["conversation_id"]] = {"template": template, "simhash": similarity, "source": record["source"]["dataset_id"], "is_scam": record["conversation_label"]["is_scam"], "scam_type": record["conversation_label"]["scam_type"], "scenario": record.get("metadata", {}).get("type")}
            for turn in record["turns"]:
                flattened = {"schema_version": SCHEMA_VERSION, "conversation_id": record["conversation_id"], "turn_id": turn["turn_id"], "speaker_role": turn["speaker_role"], "raw_text": turn["raw_text"], "normalized_text": turn["normalized_text"], "language": turn["language"], "labels": turn["labels"], "conversation_label": record["conversation_label"], "source": record["source"], "template_group_id": template}
                turns_out.write(json.dumps(flattened, ensure_ascii=False) + "\n")

    # Group template-equal and high-similarity synthetic conversations before splitting.
    union = UnionFind(eligible)
    for ids in template_members.values():
        for other in ids[1:]:
            union.union(ids[0], other)
    synthetic_sources = {"synthetic_scam_dialogue", "synthetic_multi_agent_scam_conversation"}
    buckets: defaultdict[tuple, list[str]] = defaultdict(list)
    for cid, item in eligible.items():
        if item["source"] not in synthetic_sources:
            continue
        for band in range(8):
            buckets[(item["is_scam"], item["scam_type"], item["scenario"], band, (item["simhash"] >> (band * 8)) & 0xFF)].append(cid)
    compared = set(); near_edges = 0
    for ids in buckets.values():
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                pair = (left, right) if left < right else (right, left)
                if pair in compared:
                    continue
                compared.add(pair)
                if (eligible[left]["simhash"] ^ eligible[right]["simhash"]).bit_count() <= 6:
                    union.union(left, right); near_edges += 1
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for cid in eligible:
        groups[union.find(cid)].append(cid)
    group_ids = {root_id: hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() for root_id, ids in groups.items()}
    conversation_to_group = {cid: group_ids[root_id] for root_id, ids in groups.items() for cid in ids}

    split_ids = {"train": [], "validation": [], "test": []}; split_counts = {s: Counter() for s in split_ids}
    for conversation_id, item in eligible.items():
        split = assign_split(conversation_to_group[conversation_id]); split_ids[split].append(conversation_id); split_counts[split][f"source:{item['source']}"] += 1; split_counts[split][f"label:{item['is_scam']}"] += 1; split_counts[split][f"type:{item['scam_type']}"] += 1
    for split, ids in split_ids.items():
        ids.sort(); (splits_dir / f"{split}_conversation_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    conversation_to_split = {cid: split for split, ids in split_ids.items() for cid in ids}
    template_split = {}
    leakage = []
    for template, ids in groups.items():
        observed = {conversation_to_split[i] for i in ids}
        if len(observed) != 1:
            leakage.append({"template": template, "splits": sorted(observed), "ids": ids[:5]})
        template_split[template] = next(iter(observed))
    manifest = {"schema_version": SCHEMA_VERSION, "method": "exact/template grouping plus synthetic SimHash LSH (Hamming <= 6), then SHA-256 group assignment; 70/15/15", "seed_namespace": "arrestshield-split-v1", "counts": {k: len(v) for k, v in split_ids.items()}, "breakdown": {k: dict(v) for k, v in split_counts.items()}, "dedup_groups": len(groups), "near_similarity_edges": near_edges, "largest_group": max(map(len, groups.values()), default=0), "group_leakage": leakage, "conversation_to_group": conversation_to_group, "conversation_to_split": conversation_to_split}
    (splits_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"schema_version": SCHEMA_VERSION, "stats": dict(stats), "source_conversations": dict(sources), "conversation_labels": dict(labels), "exact_unique": len(exact_seen), "exact_duplicate_sources": dict(exact_duplicate_sources), "raw_template_groups": len(template_members), "dedup_groups": len(groups), "multi_member_dedup_groups": sum(1 for v in groups.values() if len(v) > 1), "largest_dedup_group": max(map(len, groups.values()), default=0), "near_similarity_edges": near_edges, "validation_errors": dict(validation_errors), "split_counts": manifest["counts"], "group_leakage_count": len(leakage), "notes": ["Exact duplicates are excluded from conversations.jsonl and turns.jsonl but retained with flags in conversations.all.jsonl.", "High-similarity synthetic conversations remain included but are clustered into one split to prevent paraphrase leakage.", "Positive turn-level tactics/stages from public data are weak_rule labels, never gold.", "Original public train/test splits are retained only as provenance and ignored for ArrestShield splitting.", "HINMIX remains a separate language-robustness source and is not mislabeled as scam/non-scam."]}
    (docs / "CANONICAL_BUILD_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Canonical Dataset Build Report", "", f"- Input conversations: {stats['input_conversations']:,}", f"- Retained after exact deduplication/validation: {stats['kept_conversations']:,}", f"- Retained turns: {stats['kept_turns']:,}", f"- Exact duplicates excluded: {stats['excluded_conversations']:,}", f"- Leakage-safe groups: {len(groups):,}", f"- Multi-member similarity groups: {report['multi_member_dedup_groups']:,}", f"- Largest similarity group: {report['largest_dedup_group']:,}", f"- Group leakage across splits: {len(leakage)}", "", "## Split counts", "", *[f"- {k}: {len(v):,}" for k, v in split_ids.items()], "", "## Source counts", "", *[f"- {k}: {v:,}" for k, v in sorted(sources.items())], "", "## Guardrails", "", *[f"- {n}" for n in report["notes"]], ""]
    (docs / "CANONICAL_BUILD_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if leakage or validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
