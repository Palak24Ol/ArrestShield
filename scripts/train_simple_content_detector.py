"""Train the simple binary transcript-content detector used by the demo path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, sha256_file, write_json  # noqa: E402
from arrestshield.calibration import select_shared_threshold_at_group_fpr  # noqa: E402
from arrestshield.content_training import (  # noqa: E402
    external_as_examples,
    load_project_content_examples,
    partition_external_calls,
    source_label_balanced_weights,
    split_counts,
)
from arrestshield.data import labels, load_examples, texts  # noqa: E402
from arrestshield.evaluation import binary_metrics, grouped_metrics  # noqa: E402
from arrestshield.external_evaluation import load_external_text_manifest  # noqa: E402
from arrestshield.ladder import build_feature_union, build_sgd, positive_scores  # noqa: E402


LOGGER = logging.getLogger("arrestshield.train_simple_content_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/model/simple_content_detector.json",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=PROJECT_ROOT / "data/processed/conversations.jsonl",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "data/splits/split_manifest.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/models/simple_content_detector_v1",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/simple_content_detector_v1",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    canonical, canonical_summary = load_examples(
        args.conversations,
        args.splits,
        config["data"]["allowed_label_provenance"],
        config["data"]["provenance_weights"],
    )
    external_path = PROJECT_ROOT / config["external_adaptation"]["manifest"]
    external = load_external_text_manifest(external_path)
    if any(record.label != 1 for record in external):
        raise ValueError("The external adaptation source is expected to be positive-only")
    partitions = partition_external_calls(
        external,
        float(config["external_adaptation"]["train_fraction"]),
        float(config["external_adaptation"]["validation_fraction"]),
    )
    external_examples = {
        split: external_as_examples(records, split)
        for split, records in partitions.items()
    }
    project_examples = load_project_content_examples(
        PROJECT_ROOT / config["project_authored_examples"]
    )
    train = canonical["train"] + external_examples["train"] + project_examples["train"]
    validation = (
        canonical["validation"]
        + external_examples["validation"]
        + project_examples["validation"]
    )

    LOGGER.info(
        "Training simple content detector with canonical=%d external_train=%d project_train=%d",
        len(canonical["train"]),
        len(external_examples["train"]),
        len(project_examples["train"]),
    )
    feature_groups = tuple(str(value) for value in config["feature_groups"])
    representation = build_feature_union(config, enabled_groups=feature_groups)
    train_matrix = representation.fit_transform(texts(train))
    validation_matrix = representation.transform(texts(validation))
    weights = source_label_balanced_weights(train)

    seeds = [int(seed) for seed in config["seeds"]]
    models = []
    validation_scores = []
    for seed in seeds:
        model = build_sgd(config, seed)
        model.fit(train_matrix, labels(train), sample_weight=weights)
        models.append(model)
        validation_scores.append(positive_scores(model, validation_matrix))

    operating = select_shared_threshold_at_group_fpr(
        labels(validation),
        validation_scores,
        [example.source for example in validation],
        float(config["evaluation"]["maximum_false_positive_rate"]),
        int(config["evaluation"]["minimum_validation_negatives_per_source"]),
    )
    threshold = float(operating["threshold"])
    LOGGER.info("Selected shared threshold %.6f", threshold)

    canonical_test_matrix = representation.transform(texts(canonical["test"]))
    external_test = external_examples["test"]
    external_test_matrix = representation.transform(texts(external_test))
    seed_results = []
    for seed, model in zip(seeds, models):
        canonical_scores = positive_scores(model, canonical_test_matrix)
        external_scores = positive_scores(model, external_test_matrix)
        seed_results.append(
            {
                "seed": seed,
                "canonical_test": binary_metrics(
                    labels(canonical["test"]), canonical_scores, threshold
                ),
                "canonical_test_slices": grouped_metrics(
                    canonical["test"], canonical_scores, threshold
                ),
                "external_adaptation_test": binary_metrics(
                    labels(external_test), external_scores, threshold
                ),
            }
        )

    deployment_seed = int(config["deployment_seed"])
    deployment_index = seeds.index(deployment_seed)
    exit_threshold = threshold * float(config["evaluation"]["exit_threshold_ratio"])
    bundle = {
        "schema_version": "1.0.0",
        "model_family": "simple_transcript_content_sgd",
        "feature_variant": "_".join(feature_groups),
        "seed": deployment_seed,
        "threshold": threshold,
        "exit_threshold": exit_threshold,
        "feature_union": representation,
        "svd": None,
        "model": models[deployment_index],
        "calibrator": None,
        "calibration_method": "none",
        "promotion_status": config["deployment_status"],
        "training_sources": sorted({example.source for example in train}),
        "llm_used_for_detection": False,
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifact_dir / "selected_detector.joblib"
    joblib.dump(bundle, artifact_path, compress=3)

    report = {
        "schema_version": "1.0.0",
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "objective": config["objective"],
        "canonical_summary": asdict(canonical_summary),
        "external_partition_counts": split_counts(partitions),
        "project_authored_counts": {
            split: len(rows) for split, rows in project_examples.items()
        },
        "external_partition_grouped_by": "source_url_or_conversation_id",
        "external_source_previously_inspected": True,
        "test_status": config["external_adaptation"]["test_status"],
        "seeds": seeds,
        "shared_operating_point": operating,
        "seed_results": seed_results,
        "deployment_seed": deployment_seed,
        "promotion_status": config["deployment_status"],
        "llm_used_for_detection": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    metadata = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "model_family": bundle["model_family"],
        "feature_variant": bundle["feature_variant"],
        "threshold": threshold,
        "deployment_seed": deployment_seed,
        "promotion_status": bundle["promotion_status"],
    }
    write_json(args.report_dir / "metrics.json", report)
    write_json(args.report_dir / "run_metadata.json", metadata)
    write_json(args.report_dir / "config.json", config)
    print(json.dumps({"artifact": str(artifact_path), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
