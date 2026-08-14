"""Train, calibrate, and export the leakage-aware mixed-source research detector."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, sha256_file, write_json  # noqa: E402
from arrestshield.calibration import (  # noqa: E402
    fit_score_calibrator,
    select_shared_threshold_at_group_fpr,
    split_calibration_and_threshold,
)
from arrestshield.data import (  # noqa: E402
    ConversationExample,
    labels,
    load_examples,
    sample_weights,
    texts,
)
from arrestshield.evaluation import binary_metrics, grouped_metrics  # noqa: E402
from arrestshield.ladder import (  # noqa: E402
    build_feature_union,
    build_prefix_batch,
    build_sgd,
    positive_scores,
    stable_latency_from_flat_scores,
)


LOGGER = logging.getLogger("arrestshield.train_mixed_source_candidate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/model/mixed_source_detector.json"
    )
    parser.add_argument(
        "--conversations", type=Path, default=PROJECT_ROOT / "data/processed/conversations.jsonl"
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "data/splits/split_manifest.json"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/models/mixed_source_candidate_v2",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports/mixed_source_candidate_v2"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    return parser.parse_args()


def mixed_sources(
    examples: Sequence[ConversationExample], minimum_per_label: int
) -> list[str]:
    counts: dict[str, dict[int, int]] = {}
    for example in examples:
        counts.setdefault(example.source, {0: 0, 1: 0})[example.label] += 1
    return sorted(
        source
        for source, values in counts.items()
        if min(values.values()) >= minimum_per_label
    )


def source_auc(
    config: Mapping[str, Any],
    groups: Sequence[str],
    train: Sequence[ConversationExample],
    held: Sequence[ConversationExample],
    seeds: Sequence[int],
) -> list[float]:
    representation = build_feature_union(config, enabled_groups=groups)
    train_matrix = representation.fit_transform(texts(train))
    held_matrix = representation.transform(texts(held))
    y_train = np.asarray(labels(train), dtype=np.int8)
    y_held = np.asarray(labels(held), dtype=np.int8)
    weights = np.asarray(sample_weights(train), dtype=np.float32)
    scores: list[float] = []
    for seed in seeds:
        model = build_sgd(config, seed)
        model.fit(train_matrix, y_train, sample_weight=weights)
        scores.append(float(roc_auc_score(y_held, positive_scores(model, held_matrix))))
    return scores


def run_feature_ablation(
    config: Mapping[str, Any],
    by_split: Mapping[str, Sequence[ConversationExample]],
    retained_sources: Sequence[str],
    seeds: Sequence[int],
    variants: Mapping[str, Sequence[str]],
) -> tuple[str, dict[str, Any]]:
    results: dict[str, Any] = {}
    for variant, groups in variants.items():
        LOGGER.info("Ablating feature variant %s", variant)
        per_source: dict[str, Any] = {}
        for held_source in retained_sources:
            train = [
                example
                for example in by_split["train"]
                if example.source in retained_sources and example.source != held_source
            ]
            held = [
                example for example in by_split["train"] if example.source == held_source
            ]
            if {example.label for example in train} != {0, 1} or {
                example.label for example in held
            } != {0, 1}:
                per_source[held_source] = {"skipped": "single-class train or held source"}
                continue
            values = source_auc(config, groups, train, held, seeds)
            per_source[held_source] = {
                "seeds": list(seeds),
                "roc_auc": values,
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values)),
                "train_rows": len(train),
                "held_rows": len(held),
            }
        usable = [row for row in per_source.values() if "mean" in row]
        if not usable:
            raise ValueError(f"No evaluable sources for feature variant {variant}")
        by_seed = [
            float(np.mean([row["roc_auc"][seed_index] for row in usable]))
            for seed_index in range(len(seeds))
        ]
        results[variant] = {
            "enabled_groups": list(groups),
            "per_source": per_source,
            "source_macro_roc_auc_by_seed": by_seed,
            "source_macro_roc_auc_mean": float(np.mean(by_seed)),
            "source_macro_roc_auc_standard_deviation": float(np.std(by_seed)),
        }
    selected = max(
        results,
        key=lambda name: (
            results[name]["source_macro_roc_auc_mean"],
            -results[name]["source_macro_roc_auc_standard_deviation"],
            name == "word_char",
        ),
    )
    return selected, results


def evaluate_selected_feature_on_test(
    config: Mapping[str, Any],
    groups: Sequence[str],
    by_split: Mapping[str, Sequence[ConversationExample]],
    retained_sources: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Evaluate the already selected feature family on untouched source holdouts."""
    per_source: dict[str, Any] = {}
    for held_source in retained_sources:
        train = [
            example
            for example in by_split["train"]
            if example.source in retained_sources and example.source != held_source
        ]
        held = [example for example in by_split["test"] if example.source == held_source]
        if {example.label for example in train} != {0, 1} or {
            example.label for example in held
        } != {0, 1}:
            per_source[held_source] = {"skipped": "single-class train or held source"}
            continue
        values = source_auc(config, groups, train, held, seeds)
        per_source[held_source] = {
            "seeds": list(seeds),
            "roc_auc": values,
            "mean": float(np.mean(values)),
            "standard_deviation": float(np.std(values)),
            "train_rows": len(train),
            "held_rows": len(held),
        }
    usable = [row for row in per_source.values() if "mean" in row]
    if not usable:
        raise ValueError("No test source supports selected-feature source holdout")
    by_seed = [
        float(np.mean([row["roc_auc"][seed_index] for row in usable]))
        for seed_index in range(len(seeds))
    ]
    return {
        "selection_role": "supporting_only_after_feature_selection",
        "per_source": per_source,
        "source_macro_roc_auc_by_seed": by_seed,
        "source_macro_roc_auc_mean": float(np.mean(by_seed)),
        "source_macro_roc_auc_standard_deviation": float(np.std(by_seed)),
    }


def calibrate_and_select(
    methods: Sequence[str],
    raw_score_rows: Sequence[np.ndarray],
    validation: Sequence[ConversationExample],
    calibration_indices: Sequence[int],
    threshold_indices: Sequence[int],
    maximum_fpr: float,
    minimum_group_negatives: int,
) -> tuple[str, dict[str, Any], dict[str, list[Any]]]:
    y = np.asarray(labels(validation), dtype=np.int8)
    weights = np.asarray(sample_weights(validation), dtype=np.float64)
    groups = [example.source for example in validation]
    fitted: dict[str, list[Any]] = {}
    results: dict[str, Any] = {}
    for method in methods:
        calibrators = []
        threshold_scores = []
        for raw_scores in raw_score_rows:
            calibrator = fit_score_calibrator(
                method,
                raw_scores[list(calibration_indices)],
                y[list(calibration_indices)],
                weights[list(calibration_indices)],
            )
            calibrators.append(calibrator)
            threshold_scores.append(calibrator.predict(raw_scores[list(threshold_indices)]))
        try:
            selection = select_shared_threshold_at_group_fpr(
                y[list(threshold_indices)],
                threshold_scores,
                [groups[index] for index in threshold_indices],
                maximum_fpr=maximum_fpr,
                minimum_group_negatives=minimum_group_negatives,
            )
        except RuntimeError as error:
            results[method] = {"eligible": False, "reason": str(error)}
            continue
        briers = [
            float(brier_score_loss(y[list(threshold_indices)], scores))
            for scores in threshold_scores
        ]
        results[method] = {
            "eligible": bool(selection["all_seed_source_gates_passed"]),
            "shared_operating_point": selection,
            "threshold_brier_by_seed": briers,
            "threshold_brier_mean": float(np.mean(briers)),
        }
        fitted[method] = calibrators
    eligible = [method for method, row in results.items() if row.get("eligible")]
    if not eligible:
        raise RuntimeError("No calibration method satisfied the shared per-source FPR gate")
    selected = max(
        eligible,
        key=lambda method: (
            results[method]["shared_operating_point"]["mean_recall"],
            results[method]["shared_operating_point"]["minimum_seed_recall"],
            -results[method]["threshold_brier_mean"],
            method == "platt",
        ),
    )
    return selected, results, fitted


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selection_config = config["candidate_selection"]
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    variants = {
        name: groups
        for name, groups in selection_config["feature_variants"].items()
        if args.variants is None or name in set(args.variants)
    }
    if not variants:
        raise ValueError("No requested feature variants exist in the config")

    by_split, summary = load_examples(
        args.conversations,
        args.splits,
        config["data"]["allowed_label_provenance"],
        config["data"]["provenance_weights"],
    )
    all_examples = by_split["train"] + by_split["validation"] + by_split["test"]
    retained_sources = mixed_sources(
        all_examples,
        int(config["training_regime"]["minimum_examples_per_label_per_source"]),
    )
    if len(retained_sources) < 2:
        raise ValueError("At least two mixed-label sources are required for source holdout")

    selected_variant, ablation = run_feature_ablation(
        config, by_split, retained_sources, seeds, variants
    )
    selected_groups = variants[selected_variant]
    LOGGER.info("Selected feature variant %s from source-holdout AUC", selected_variant)
    selected_test_source_holdout = evaluate_selected_feature_on_test(
        config, selected_groups, by_split, retained_sources, seeds
    )

    train = [example for example in by_split["train"] if example.source in retained_sources]
    validation = list(by_split["validation"])
    representation = build_feature_union(config, enabled_groups=selected_groups)
    train_matrix = representation.fit_transform(texts(train))
    validation_matrix = representation.transform(texts(validation))
    y_train = np.asarray(labels(train), dtype=np.int8)
    weights = np.asarray(sample_weights(train), dtype=np.float32)
    models = []
    raw_validation_scores = []
    for seed in seeds:
        model = build_sgd(config, seed)
        model.fit(train_matrix, y_train, sample_weight=weights)
        models.append(model)
        raw_validation_scores.append(positive_scores(model, validation_matrix))

    calibration_indices, threshold_indices = split_calibration_and_threshold(
        validation, float(selection_config["calibration_fraction"])
    )
    selected_calibration, calibration_results, calibrators = calibrate_and_select(
        selection_config["calibration_methods"],
        raw_validation_scores,
        validation,
        calibration_indices,
        threshold_indices,
        float(config["evaluation"]["maximum_false_positive_rate"]),
        int(selection_config["minimum_validation_negatives_per_source"]),
    )
    operating = calibration_results[selected_calibration]["shared_operating_point"]
    threshold = float(operating["threshold"])
    LOGGER.info(
        "Selected calibration=%s shared_threshold=%.6f", selected_calibration, threshold
    )

    test = list(by_split["test"])
    test_matrix = representation.transform(texts(test))
    test_runs = []
    calibrated_test_rows = []
    for seed_index, (model, calibrator) in enumerate(
        zip(models, calibrators[selected_calibration])
    ):
        calibrated_scores = calibrator.predict(positive_scores(model, test_matrix))
        calibrated_test_rows.append(calibrated_scores)
        test_runs.append(
            {
                "seed": seeds[seed_index],
                "metrics": binary_metrics(labels(test), calibrated_scores, threshold),
                "slices": grouped_metrics(test, calibrated_scores, threshold),
            }
        )

    deployment_seed = int(config["deployment_seed"])
    if deployment_seed not in seeds:
        raise ValueError("deployment_seed must be included in the training seeds")
    deployment_index = list(seeds).index(deployment_seed)
    deployment_scores = calibrated_test_rows[deployment_index]
    positives, prefix_texts, owners = build_prefix_batch(test)
    prefix_matrix = representation.transform(prefix_texts)
    prefix_raw = positive_scores(models[deployment_index], prefix_matrix)
    prefix_scores = calibrators[selected_calibration][deployment_index].predict(prefix_raw)
    exit_threshold = threshold * float(selection_config["exit_threshold_ratio"])
    latency = stable_latency_from_flat_scores(
        positives, owners, prefix_scores, threshold, exit_threshold
    )

    bundle = {
        "schema_version": "2.0.0",
        "model_family": "sgd_mixed_source_calibrated",
        "feature_variant": selected_variant,
        "seed": deployment_seed,
        "threshold": threshold,
        "exit_threshold": exit_threshold,
        "feature_union": representation,
        "svd": None,
        "model": models[deployment_index],
        "calibrator": calibrators[selected_calibration][deployment_index],
        "calibration_method": selected_calibration,
        "promotion_status": selection_config["deployment_status"],
        "training_sources": retained_sources,
        "llm_used_for_detection": False,
    }
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.artifact_dir / "selected_detector.joblib"
    joblib.dump(bundle, selected_path, compress=3)

    report = {
        "schema_version": "2.0.0",
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "data_summary": asdict(summary),
        "training_sources": retained_sources,
        "excluded_single_label_sources": sorted(
            {example.source for example in all_examples} - set(retained_sources)
        ),
        "seeds": list(seeds),
        "feature_ablation": ablation,
        "feature_selection_view": "training_partition_leave_one_source_out",
        "selected_feature_variant": selected_variant,
        "selected_feature_test_source_holdout": selected_test_source_holdout,
        "calibration_partition": {
            "method": "deterministic_disjoint_source_label_strata",
            "calibration_rows": len(calibration_indices),
            "threshold_rows": len(threshold_indices),
            "overlap": len(set(calibration_indices) & set(threshold_indices)),
        },
        "calibration_comparison": calibration_results,
        "selected_calibration": selected_calibration,
        "shared_threshold": threshold,
        "shared_threshold_across_seeds": True,
        "test_supporting_only": test_runs,
        "deployment_seed_test": binary_metrics(labels(test), deployment_scores, threshold),
        "deployment_seed_latency": latency,
        "promotion_status": selection_config["deployment_status"],
        "promotion_blockers": [
            "No independently annotated frozen human-gold call set exists.",
            "The external YouTube source is positive-only and cannot establish false-positive rate.",
            "Hinglish multi-turn and Hinglish ASR performance remain unmeasured.",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "llm_used_for_detection": False,
    }
    metadata = {
        "schema_version": "2.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "artifact_sha256": sha256_file(selected_path),
        "artifact_bytes": selected_path.stat().st_size,
        "model_family": bundle["model_family"],
        "feature_variant": selected_variant,
        "calibration_method": selected_calibration,
        "shared_threshold": threshold,
        "deployment_seed": deployment_seed,
        "promotion_status": bundle["promotion_status"],
    }
    write_json(args.report_dir / "metrics.json", report)
    write_json(args.report_dir / "run_metadata.json", metadata)
    write_json(args.report_dir / "config.json", config)
    LOGGER.info("Saved candidate %s", metadata["artifact_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
