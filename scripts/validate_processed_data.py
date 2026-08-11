"""Independent validation of canonical ArrestShield outputs and split safety."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    processed = root / "data" / "processed"; splits = root / "data" / "splits"; docs = root / "docs" / "datasets"
    manifest = json.loads((splits / "split_manifest.json").read_text(encoding="utf-8"))
    split_sets = {name: set((splits / f"{name}_conversation_ids.txt").read_text(encoding="utf-8").splitlines()) for name in ("train", "validation", "test")}
    errors = []
    if split_sets["train"] & split_sets["validation"] or split_sets["train"] & split_sets["test"] or split_sets["validation"] & split_sets["test"]:
        errors.append("split_id_overlap")
    all_split_ids = set().union(*split_sets.values())
    if len(all_split_ids) != sum(map(len, split_sets.values())):
        errors.append("split_count_not_disjoint")
    if manifest["counts"] != {k: len(v) for k, v in split_sets.items()}:
        errors.append("manifest_count_mismatch")
    if set(manifest["conversation_to_split"]) != all_split_ids:
        errors.append("manifest_id_mismatch")
    for cid, split in manifest["conversation_to_split"].items():
        if cid not in split_sets[split]:
            errors.append("manifest_assignment_mismatch"); break

    group_splits = defaultdict(set)
    for cid, group in manifest["conversation_to_group"].items():
        group_splits[group].add(manifest["conversation_to_split"][cid])
    leaking_groups = [group for group, observed in group_splits.items() if len(observed) > 1]
    if leaking_groups:
        errors.append("similarity_group_leakage")

    conversation_ids = set(); exact_hashes = set(); sources = Counter(); labels = Counter(); types = Counter(); label_provenance = Counter(); turn_total = 0; malformed = 0
    conversations_path = processed / "conversations.jsonl"
    with conversations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                cid = record["conversation_id"]
                if cid in conversation_ids: errors.append(f"duplicate_conversation_id:{cid}")
                conversation_ids.add(cid)
                exact = record["fingerprints"]["exact_sha256"]
                if exact in exact_hashes: errors.append(f"duplicate_exact_hash:{exact}")
                exact_hashes.add(exact)
                if not record["quality"]["training_eligible"]: errors.append(f"ineligible_in_kept:{cid}")
                if cid not in all_split_ids: errors.append(f"missing_split:{cid}")
                if [t["turn_id"] for t in record["turns"]] != list(range(len(record["turns"]))): errors.append(f"bad_turn_order:{cid}")
                if any(not t["normalized_text"].strip() for t in record["turns"]): errors.append(f"empty_turn:{cid}")
                turn_total += len(record["turns"])
                sources[record["source"]["dataset_id"]] += 1
                label = record["conversation_label"]
                labels[str(label["is_scam"])] += 1; types[label["scam_type"]] += 1; label_provenance[label["provenance"]] += 1
            except Exception:
                malformed += 1
    if malformed: errors.append(f"malformed_conversations:{malformed}")
    if conversation_ids != all_split_ids: errors.append("kept_conversation_split_set_mismatch")

    flattened_turns = 0; last_cid = None; expected_turn = 0; turn_order_errors = 0
    turns_path = processed / "turns.jsonl"
    with turns_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line); cid = record["conversation_id"]
            if cid not in conversation_ids: errors.append(f"orphan_turn:{cid}"); break
            if cid != last_cid:
                last_cid = cid; expected_turn = 0
            if record["turn_id"] != expected_turn:
                turn_order_errors += 1
            expected_turn += 1; flattened_turns += 1
    if flattened_turns != turn_total: errors.append("flattened_turn_count_mismatch")
    if turn_order_errors: errors.append(f"flattened_turn_order_errors:{turn_order_errors}")

    artifacts = {}
    for path in [conversations_path, turns_path, splits / "split_manifest.json"]:
        artifacts[str(path.relative_to(root)).replace("\\", "/")] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "conversations": len(conversation_ids),
        "turns": turn_total,
        "split_counts": {k: len(v) for k, v in split_sets.items()},
        "source_counts_after_dedup": dict(sources),
        "binary_label_counts_after_dedup": dict(labels),
        "scam_type_counts_after_dedup": dict(types),
        "conversation_label_provenance": dict(label_provenance),
        "similarity_groups": len(group_splits),
        "leaking_similarity_groups": len(leaking_groups),
        "artifacts": artifacts,
    }
    (docs / "PROCESSED_DATA_VALIDATION.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Processed Dataset Validation", "", f"**Status: {report['status'].upper()}**", "", f"- Conversations: {report['conversations']:,}", f"- Turns: {report['turns']:,}", f"- Train / validation / test: {report['split_counts']['train']:,} / {report['split_counts']['validation']:,} / {report['split_counts']['test']:,}", f"- Similarity groups crossing splits: {report['leaking_similarity_groups']}", f"- Errors: {len(errors)}", "", "## Labels after deduplication", "", *[f"- {k}: {v:,}" for k, v in types.most_common()], "", "## Sources after deduplication", "", *[f"- {k}: {v:,}" for k, v in sources.most_common()], "", "## Artifact hashes", "", *[f"- `{k}` — {v['bytes']:,} bytes — `{v['sha256']}`" for k, v in artifacts.items()], ""]
    (docs / "PROCESSED_DATA_VALIDATION.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
