"""Generate declared Hinglish training augmentation with Groq.

Weighted toward legitimate hard negatives, which is the class the corpus is
missing: 633 unique Hinglish scam conversations exist against 110 legitimate
ones. Output is training-split-only and tagged `llm_synthetic`.

    python scripts/generate_hinglish_augmentation.py --smoke
    python scripts/generate_hinglish_augmentation.py --scam 300 --legitimate 700
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.augmentation import (  # noqa: E402
    DATASET_ID,
    build_requests,
    generate,
    summarize,
)
from arrestshield.llm_client import GroqSettings, load_environment_file  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "data/external/llm_augmentation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scam", type=int, default=300)
    parser.add_argument("--legitimate", type=int, default=700)
    parser.add_argument("--smoke", action="store_true", help="Generate 3 + 5 and stop")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append", action="store_true", help="Add to an existing file instead of replacing it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_environment_file(PROJECT_ROOT / ".env")

    scam_count = 3 if args.smoke else args.scam
    legitimate_count = 5 if args.smoke else args.legitimate
    output_path = args.output_dir / "conversations.jsonl"

    existing: list[dict] = []
    if args.append and output_path.exists():
        existing = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing_templates = {row["fingerprints"]["template_sha256"] for row in existing}

    settings = GroqSettings(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=90,
        maximum_retries=2,
    )
    requests = build_requests(scam_count, legitimate_count, seed=args.seed)
    print(f"Generating {scam_count} scam + {legitimate_count} legitimate conversations")
    print(f"model={args.model} temperature={args.temperature}")
    print(f"output={output_path}\n")

    def progress(index, record, error):
        total = len(requests)
        if record is None:
            print(f"  [{index + 1}/{total}] skipped: {error}")
        else:
            label = "scam" if record["conversation_label"]["is_scam"] else "legit"
            print(f"  [{index + 1}/{total}] {label:5s} {len(record['turns']):2d} turns  {record['conversation_id']}")

    records, failures = generate(
        requests, settings, existing_templates=existing_templates, on_progress=progress
    )
    if not records:
        print("\nNo usable conversations were generated.")
        if failures:
            print(f"First failure: {failures[0]['error']}")
        return 1

    combined = existing + records
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in combined:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "temperature": args.temperature,
        "generation_seed": args.seed,
        "requested": {"scam": scam_count, "legitimate": legitimate_count},
        "summary": summarize(combined),
        "failures": len(failures),
        "failure_examples": failures[:5],
        "policy": {
            "allowed_splits": ["train"],
            "never_validation_or_test": True,
            "sample_weight": 0.5,
            "llm_used_for_detection": False,
            "declaration": (
                "These conversations are LLM-generated and are used only to train. They are "
                "not evidence of real-world performance and must never appear in a reported "
                "validation or test result."
            ),
        },
    }
    (args.output_dir / "GENERATION_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = report["summary"]
    print(f"\nWrote {len(combined)} conversations -> {output_path}")
    print(f"  scam={summary['scam']} legitimate={summary['legitimate']} "
          f"unique_templates={summary['unique_templates']} mean_turns={summary['mean_turns']}")
    print(f"  failures={len(failures)}")
    print(f"\nTrain with:\n  python scripts/train_mixed_source_detector.py --augmentation {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
