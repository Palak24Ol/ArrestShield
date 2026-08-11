"""Validate and freeze the independent project-authored human test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.human_test import (  # noqa: E402
    build_frozen_records,
    normalized_transcript,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs/data/human_frozen_test_protocol.json",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=PROJECT_ROOT / "configs/data/manipulation_taxonomy.json",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=PROJECT_ROOT / "data/processed/conversations.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/human_test/frozen/v1",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def canonical_leakage_keys(path: Path) -> tuple[set[str], set[str]]:
    identifiers: set[str] = set()
    hashes: set[str] = set()
    for row in read_jsonl(path):
        identifiers.add(str(row.get("conversation_id") or ""))
        converted = {
            "turns": [
                {
                    "speaker_role": turn.get("speaker_role"),
                    "text": turn.get("normalized_text") or turn.get("raw_text") or "",
                }
                for turn in row.get("turns") or []
            ]
        }
        digest = hashlib.sha256(normalized_transcript(converted).encode("utf-8")).hexdigest()
        hashes.add(digest)
    return identifiers, hashes


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    protocol = read_json(args.protocol)
    taxonomy = read_json(args.taxonomy)
    tactic_names = set(taxonomy["research_psychological_techniques"]) | set(
        taxonomy["digital_arrest_operational_signals"]
    )
    canonical_ids, canonical_hashes = canonical_leakage_keys(args.canonical)
    rows, manifest = build_frozen_records(
        read_jsonl(args.intake),
        read_jsonl(args.annotations),
        read_jsonl(args.adjudications),
        protocol,
        canonical_conversation_ids=canonical_ids,
        canonical_transcript_hashes=canonical_hashes,
        tactic_names=tactic_names,
    )
    if args.validate_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_path = args.output_dir / "conversations.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"Frozen output is write-once and already exists under {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_atomic(output_path, rows)
    manifest["conversations_sha256"] = sha256_file(output_path)
    manifest["protocol_sha256"] = sha256_file(args.protocol)
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
