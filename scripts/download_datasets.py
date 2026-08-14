"""Download approved ArrestShield seed datasets with immutable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "ArrestShield-Dataset-Bootstrap/1.0 (academic research)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int = 3) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    if part.exists():
        part.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            os.replace(part, destination)
            return {
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed after {retries} attempts: {url}") from last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Allow an explicitly selected optional_* dataset; never affects blocked/rejected sources.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    project_root = args.root.resolve() if args.root else registry_path.parents[2]
    raw_root = project_root / "data" / "raw"
    manifest_path = project_root / "data" / "manifests" / "download_manifest.jsonl"
    summary_path = project_root / "data" / "manifests" / "download_summary.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = set(args.dataset)

    receipts: list[dict] = []
    failures: list[dict] = []
    skipped: list[dict] = []
    for dataset in registry["datasets"]:
        dataset_id = dataset["id"]
        if selected and dataset_id not in selected:
            continue
        status = str(dataset["status"])
        optional_allowed = bool(
            args.include_optional
            and dataset_id in selected
            and status.startswith("optional_")
        )
        if status not in registry["policy"]["automatic_download_statuses"] and not optional_allowed:
            skipped.append({"dataset_id": dataset_id, "status": dataset["status"]})
            continue
        for item in dataset.get("files", []):
            destination = raw_root / dataset_id / item["path"]
            try:
                if destination.exists() and not args.force:
                    result = {"bytes": destination.stat().st_size, "sha256": sha256_file(destination)}
                    action = "verified_existing"
                else:
                    result = download(item["url"], destination)
                    action = "downloaded"
                receipt = {
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "dataset_id": dataset_id,
                    "repository": dataset["repository"],
                    "revision": dataset["revision"],
                    "license": dataset["license"],
                    "relative_path": str(destination.relative_to(project_root)).replace("\\", "/"),
                    "url": item["url"],
                    "action": action,
                    **result,
                }
                receipts.append(receipt)
                print(f"[{action}] {dataset_id}/{item['path']} ({result['bytes']:,} bytes)")
            except Exception as exc:  # continue to report every source
                failures.append({"dataset_id": dataset_id, "path": item["path"], "error": repr(exc)})
                print(f"[failed] {dataset_id}/{item['path']}: {exc}", file=sys.stderr)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if selected and manifest_path.exists():
        existing: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing.append(json.loads(line))
        merged = {
            (str(receipt.get("dataset_id")), str(receipt.get("relative_path"))): receipt
            for receipt in existing
        }
        for receipt in receipts:
            merged[(receipt["dataset_id"], receipt["relative_path"])] = receipt
        receipts = [merged[key] for key in sorted(merged)]
    with manifest_path.open("w", encoding="utf-8") as handle:
        for receipt in receipts:
            handle.write(json.dumps(receipt, ensure_ascii=False) + "\n")
    summary = {
        "registry_version": registry["registry_version"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "downloaded_or_verified_files": len(receipts),
        "downloaded_or_verified_bytes": sum(r["bytes"] for r in receipts),
        "failures": failures,
        "deferred_datasets": skipped,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
