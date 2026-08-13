"""Fetch the colloquial Hinglish BFSI corpus as declared hard negatives.

These are legitimate Hinglish banking/KYC/loan/insurance conversations. The
ArrestShield corpus contains only 110 legitimate Hinglish conversations, so the
detector has effectively never seen Hinglish financial vocabulary outside a scam.

Two properties matter and both are declared, not assumed:

1. The corpus is synthetic, but it is *independently* synthetic. A different
   author and generator means its writing style does not coincide with the style
   that already defines "scam" in our training data.
2. Its Hugging Face card states no licence. It is therefore loaded as
   train-split-only, weighted below real data, and flagged in the registry.

Because it is a single-label source, adding it can in principle reintroduce the
source shortcut. That risk is not argued away here: leave-one-source-out on the
untouched Hinglish test set will show it if it happens.

    python scripts/fetch_colloquial_hinglish.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.augmentation import fingerprint, normalize_text  # noqa: E402

DATASET_ID = "colloquial_hinglish_bfsi"
REPOSITORY = "ankitdhiman/colloquial-hinglish-conversations"
REVISION = "b5dce134606d43dc5573a8b2eb721df628db2a53"
PROVENANCE = "external_unverified_license"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
ROLE_MAP = {"user": "callee", "assistant": "caller", "system": "caller"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="single_turn")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/external/colloquial_hinglish_bfsi"
    )
    return parser.parse_args()


def fetch_page(config: str, split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": REPOSITORY, "config": config, "split": split, "offset": offset, "length": length}
    )
    request = urllib.request.Request(
        f"{ROWS_ENDPOINT}?{query}", headers={"User-Agent": "ArrestShield/1.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def build_record(messages: list[dict], index: int) -> dict | None:
    turns: list[dict] = []
    for message in messages:
        body = normalize_text(str(message.get("content") or ""))
        if not body:
            continue
        role = ROLE_MAP.get(str(message.get("role") or "").lower(), "caller")
        turns.append(
            {
                "turn_id": len(turns),
                "speaker_role": role,
                "raw_text": body,
                "normalized_text": body,
                "language": "hinglish",
                "labels": {
                    "is_scam": 0,
                    "scam_type": "non_scam",
                    "tactics": {},
                    "stage": "none_unknown",
                    "provenance": PROVENANCE,
                },
            }
        )
    if len(turns) < 2:
        return None
    exact = fingerprint(turns)
    return {
        "schema_version": "1.0.0",
        "conversation_id": f"{DATASET_ID}-{exact[:16]}",
        "source": {
            "dataset_id": DATASET_ID,
            "revision": REVISION,
            "license": "UNSPECIFIED_ON_DATASET_CARD",
            "record_id": f"row-{index:05d}",
            "original_split": "train",
            "source_type": "external_synthetic",
        },
        "language_profile": ["hinglish", "hindi", "english"],
        "conversation_label": {"is_scam": 0, "scam_type": "non_scam", "provenance": PROVENANCE},
        "turns": turns,
        "fingerprints": {
            "exact_sha256": exact,
            "template_sha256": fingerprint(turns, template=True),
        },
        "quality": {
            "training_eligible": True,
            "exclusion_reasons": [],
            "label_quality": PROVENANCE,
        },
        "split_policy": {
            "allowed_splits": ["train"],
            "never_validation_or_test": True,
            "reason": "Licence unverified and corpus is synthetic; it cannot serve as evidence of real-world performance.",
        },
    }


def main() -> int:
    args = parse_args()
    records: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    offset = 0
    total = None

    while True:
        page = fetch_page(args.config, args.split, offset, args.page_size)
        total = page.get("num_rows_total", total)
        rows = page.get("rows") or []
        if not rows:
            break
        for row in rows:
            messages = (row.get("row") or {}).get("messages") or []
            record = build_record(messages, offset + len(records) + duplicates)
            if record is None:
                continue
            template = record["fingerprints"]["template_sha256"]
            if template in seen:
                duplicates += 1
                continue
            seen.add(template)
            records.append(record)
        offset += len(rows)
        print(f"  fetched {offset}/{total} rows -> {len(records)} unique conversations")
        if args.limit and len(records) >= args.limit:
            records = records[: args.limit]
            break
        if total is not None and offset >= total:
            break
        time.sleep(0.3)

    if not records:
        print("No conversations were retrieved.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "conversations.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    payload = output_path.read_bytes()
    turn_counts = [len(row["turns"]) for row in records]
    report = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "repository": REPOSITORY,
        "revision": REVISION,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_available": total,
        "conversations_written": len(records),
        "exact_duplicates_dropped": duplicates,
        "mean_turns": round(sum(turn_counts) / len(turn_counts), 2),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "label": {"is_scam": 0, "role": "hard_negative_hinglish_financial"},
        "policy": {
            "allowed_splits": ["train"],
            "never_validation_or_test": True,
            "license": "UNSPECIFIED_ON_DATASET_CARD",
            "license_action_required": "Contact the dataset author for written terms before any release or publication.",
            "llm_used_for_detection": False,
        },
    }
    (args.output_dir / "FETCH_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {len(records)} conversations -> {output_path}")
    print(f"  duplicates dropped: {duplicates}   mean turns: {report['mean_turns']}")
    print(f"\nRe-run training with:\n  python scripts/train_mixed_source_detector.py --extra-negatives {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
