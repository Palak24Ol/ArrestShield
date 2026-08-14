"""Convert approved external sources into frozen evaluation-only manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize(value: str) -> str:
    return " ".join(str(value).split())


def youtube_records(archive: Path) -> list[dict]:
    with zipfile.ZipFile(archive) as handle:
        candidates = [name for name in handle.namelist() if Path(name).name == "FullTranscriptData.csv"]
        if len(candidates) != 1:
            raise ValueError(f"Expected one FullTranscriptData.csv in {archive}; found {candidates}")
        with handle.open(candidates[0]) as source:
            rows = list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig")))

    records: list[dict] = []
    seen_text: set[str] = set()
    for index, row in enumerate(rows):
        text = normalize(row.get("Content", ""))
        if not text:
            continue
        fingerprint = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
        if fingerprint in seen_text:
            continue
        seen_text.add(fingerprint)
        source_url = normalize(row.get("Source", ""))
        source_id = normalize(row.get("ID", "")) or str(index)
        records.append(
            {
                "schema_version": "1.0.0",
                "record_id": f"youtube-scam-{source_id}",
                "conversation_id": f"youtube-scam-{source_id}",
                "text": f"[ROLE=caller] {text}",
                "label": 1,
                "language": "english",
                "source_group": "youtube_scam_phone_call_transcripts",
                "source_url": source_url,
                "rights_basis": "CC0-1.0 public-domain dedication declared by dataset publisher",
                "pii_redacted": True,
                "split": "external_evaluation",
                "training_eligible": False,
                "threshold_selection_eligible": False,
            }
        )
    if not records:
        raise ValueError("No usable YouTube scam transcripts found")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--youtube-archive",
        type=Path,
        default=PROJECT_ROOT / "data/raw/youtube_scam_phone_call_transcripts/dataset.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/interim/external_evaluation/youtube_scam_calls.jsonl",
    )
    args = parser.parse_args()
    records = youtube_records(args.youtube_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
