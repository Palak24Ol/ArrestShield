"""Create a non-mutating profile of downloaded ArrestShield raw datasets."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path


def csv_profile(path: Path) -> dict:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeError(f"Unable to decode {path}")
    profile = {"rows": len(rows), "columns": reader.fieldnames or [], "encoding": encoding}
    values = {}
    unique_counts = {}
    for column in profile["columns"]:
        counter = Counter(str(row.get(column, "")).strip() for row in rows)
        unique_counts[column] = len(counter)
        if len(counter) <= 30:
            values[column] = dict(counter.most_common(30))
    profile["categorical_counts"] = values
    profile["unique_counts"] = unique_counts
    return profile


def daily_dialog_profile(dataset_dir: Path) -> dict:
    result = {"archives": {}}
    total_dialogues = 0
    for archive in sorted(dataset_dir.glob("*.zip")):
        archive_result = {"files": [], "dialogues": None}
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                info = zf.getinfo(name)
                archive_result["files"].append({"name": name, "bytes": info.file_size})
                filename = Path(name).name
                if filename.startswith("dialogues_") and "_act_" not in filename and "_emotion_" not in filename:
                    with zf.open(name) as source:
                        count = sum(1 for line in io.TextIOWrapper(source, encoding="utf-8") if line.strip())
                    archive_result["dialogues"] = count
                    total_dialogues += count
        result["archives"][archive.name] = archive_result
    result["total_dialogues"] = total_dialogues
    return result


def sgd_profile(path: Path) -> dict:
    result = {"json_files": 0, "dialogues": 0, "turns": 0, "services": Counter(), "splits": Counter()}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json") or "/dialogues_" not in name:
                continue
            result["json_files"] += 1
            split = name.split("/")[1] if "/" in name else "unknown"
            try:
                records = json.loads(zf.read(name))
            except json.JSONDecodeError:
                continue
            if not isinstance(records, list):
                continue
            result["dialogues"] += len(records)
            result["splits"][split] += len(records)
            for record in records:
                result["turns"] += len(record.get("turns", []))
                result["services"].update(record.get("services", []))
    result["services"] = dict(result["services"].most_common())
    result["splits"] = dict(result["splits"])
    return result


def parquet_profile(path: Path) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        return {"error": f"pyarrow unavailable: {exc}"}
    metadata = pq.ParquetFile(path).metadata
    schema = pq.read_schema(path)
    return {"rows": metadata.num_rows, "row_groups": metadata.num_row_groups, "columns": schema.names}


def to_markdown(profiles: dict) -> str:
    lines = ["# Raw Dataset Profile", "", "Generated from immutable source files. Counts do not imply that records are ready for training.", ""]
    for dataset_id, profile in profiles.items():
        lines.extend([f"## {dataset_id}", "", "```json", json.dumps(profile, indent=2, ensure_ascii=False), "```", ""])
    lines.extend([
        "## Interpretation",
        "",
        "- Public scam corpora are mostly synthetic or weakly documented; they require deduplication and manual label audit.",
        "- BANKING77, DailyDialog, and Schema-Guided Dialogue provide legitimate-domain hard negatives, but they are not phone-call recordings.",
        "- HINMIX is language robustness data only and must not receive scam labels.",
        "- The primary English/Hindi/Hinglish ArrestShield corpus with turn-level tactic/stage labels is still required.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    profiles = {
        "indian_cyber_scam_phonecall_hinglish": csv_profile(raw / "indian_cyber_scam_phonecall_hinglish" / "India_Cyber_Scam_Hinglish_Dataset.csv"),
        "indian_multilingual_scam_messages": csv_profile(raw / "indian_multilingual_scam_messages" / "ultra_premium_scam_dataset.csv"),
        "synthetic_scam_dialogue": csv_profile(raw / "synthetic_scam_dialogue" / "scam-dialogue_all.csv"),
        "synthetic_multi_agent_scam_conversation": csv_profile(raw / "synthetic_multi_agent_scam_conversation" / "agent_conversation_all.csv"),
        "banking77_train": csv_profile(raw / "banking77" / "train.csv"),
        "banking77_test": csv_profile(raw / "banking77" / "test.csv"),
        "daily_dialog": daily_dialog_profile(raw / "daily_dialog"),
        "schema_guided_dialogue": sgd_profile(raw / "schema_guided_dialogue" / "repository.zip"),
        "hinmix_hicmrom_test": parquet_profile(raw / "hinmix_gold_code_mix" / "lcsalign-hicmrom" / "test.parquet"),
        "hinmix_hicmrom_valid": parquet_profile(raw / "hinmix_gold_code_mix" / "lcsalign-hicmrom" / "valid.parquet"),
        "hinmix_noisy_test": parquet_profile(raw / "hinmix_gold_code_mix" / "lcsalign-noisyhicmrom" / "test.parquet"),
        "hinmix_noisy_valid": parquet_profile(raw / "hinmix_gold_code_mix" / "lcsalign-noisyhicmrom" / "valid.parquet"),
    }
    docs = root / "docs" / "datasets"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "RAW_DATA_PROFILE.json").write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    (docs / "RAW_DATA_PROFILE.md").write_text(to_markdown(profiles), encoding="utf-8")
    print(json.dumps(profiles, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
