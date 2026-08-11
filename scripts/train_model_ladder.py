"""Train and compare SGD and SVD-XGBoost using the pre-registered protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Sequence

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, sha256_file, write_json  # noqa: E402
from arrestshield.data import ConversationExample, labels, load_examples, sample_weights, texts  # noqa: E402
from arrestshield.evaluation import binary_metrics, grouped_metrics  # noqa: E402
from arrestshield.ladder import (  # noqa: E402
    build_feature_union,
    build_prefix_batch,
    build_sgd,
    build_svd,
    build_xgboost,
    choose_family,
    positive_scores,
    stable_latency_from_flat_scores,
)
from arrestshield.protocol import select_threshold_at_fpr, summarize_seed_values  # noqa: E402


LOGGER = logging.getLogger("arrestshield.train_model_ladder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/model/model_ladder.json")
    parser.add_argument("--conversations", type=Path, default=PROJECT_ROOT / "data/processed/conversations.jsonl")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data/splits/split_manifest.json")
    parser.add_argument("--views", type=Path, default=PROJECT_ROOT / "data/splits/evaluation_views.json")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts/models/model_ladder_v1")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/model_ladder_v1")
    parser.add_argument("--max-per-split", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def indices_for_primary(
    examples: Sequence[ConversationExample], hard_negative_ids: set[str]
) -> list[int]:
    return [
        index
        for index, example in enumerate(examples)
        if example.label == 1 or example.conversation_id in hard_negative_ids
    ]


def subset(values: Sequence[Any] | np.ndarray, indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def evaluate_seed(
    model_name: str,
    seed: int,
    model: Any,
    validation_matrix: Any,
    test_matrix: Any,
    validation_examples: Sequence[ConversationExample],
    test_examples: Sequence[ConversationExample],
    validation_primary_indices: Sequence[int],
    test_primary_indices: Sequence[int],
    validation_prefix_matrix: Any,
    validation_positives: Sequence[ConversationExample],
    validation_owners: Sequence[tuple[int, int]],
    test_prefix_matrix: Any,
    test_positives: Sequence[ConversationExample],
    test_owners: Sequence[tuple[int, int]],
    maximum_fpr: float,
    exit_threshold_ratio: float,
) -> dict[str, Any]:
    validation_scores = positive_scores(model, validation_matrix)
    test_scores = positive_scores(model, test_matrix)
    operating = select_threshold_at_fpr(
        subset(labels(validation_examples), validation_primary_indices),
        subset(validation_scores, validation_primary_indices),
        maximum_fpr=maximum_fpr,
    )
    threshold = operating.threshold
    exit_threshold = threshold * exit_threshold_ratio
    validation_prefix_scores = positive_scores(model, validation_prefix_matrix)
    test_prefix_scores = positive_scores(model, test_prefix_matrix)
    return {
        "model": model_name,
        "seed": seed,
        "operating_point": asdict(operating),
        "validation_primary": binary_metrics(
            subset(labels(validation_examples), validation_primary_indices),
            subset(validation_scores, validation_primary_indices),
            threshold,
        ),
        "validation_stable_latency": stable_latency_from_flat_scores(
            validation_positives,
            validation_owners,
            validation_prefix_scores,
            threshold,
            exit_threshold,
        ),
        "test_primary": binary_metrics(
            subset(labels(test_examples), test_primary_indices),
            subset(test_scores, test_primary_indices),
            threshold,
        ),
        "test_full": binary_metrics(labels(test_examples), test_scores, threshold),
        "test_slices": grouped_metrics(test_examples, test_scores, threshold),
        "test_stable_latency": stable_latency_from_flat_scores(
            test_positives,
            test_owners,
            test_prefix_scores,
            threshold,
            exit_threshold,
        ),
    }


def aggregate_runs(runs: Sequence[dict[str, Any]], artifact_bytes: int) -> dict[str, Any]:
    recalls = [run["validation_primary"]["recall"] for run in runs]
    macro_f1 = [run["validation_primary"]["macro_f1"] for run in runs]
    stable_turns = [
        run["validation_stable_latency"]["stable_scammer_turns"]["median_turn"]
        for run in runs
    ]
    if any(value is None for value in stable_turns):
        stable_turns = [float("inf") if value is None else value for value in stable_turns]
    return {
        "validation_recall": summarize_seed_values(recalls),
        "validation_macro_f1": summarize_seed_values(macro_f1),
        "validation_median_stable_scammer_turn": summarize_seed_values(stable_turns),
        "test_recall_supporting_only": summarize_seed_values(
            [run["test_primary"]["recall"] for run in runs]
        ),
        "artifact_bytes": artifact_bytes,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = read_json(args.config)
    protocol = read_json(PROJECT_ROOT / config["protocol_path"])
    if not args.views.exists():
        raise FileNotFoundError(f"Build evaluation views first: {args.views}")
    views = read_json(args.views)
    data_config = config["data"]
    by_split, load_summary = load_examples(
        args.conversations,
        args.splits,
        allowed_provenance=data_config["allowed_label_provenance"],
        provenance_weights=data_config["provenance_weights"],
        max_per_split=args.max_per_split,
    )
    train_examples = by_split["train"]
    validation_examples = by_split["validation"]
    test_examples = by_split["test"]
    LOGGER.info("Loaded split counts %s", load_summary.split_counts)

    validation_primary_indices = indices_for_primary(
        validation_examples, set(views["hard_negative_ids"]["validation"])
    )
    test_primary_indices = indices_for_primary(
        test_examples, set(views["hard_negative_ids"]["test"])
    )
    if not validation_primary_indices or not test_primary_indices:
        raise ValueError("Primary hard-negative evaluation views are empty")

    LOGGER.info("Fitting shared multilingual TF-IDF representation")
    feature_union = build_feature_union(config)
    train_sparse = feature_union.fit_transform(texts(train_examples))
    validation_sparse = feature_union.transform(texts(validation_examples))
    test_sparse = feature_union.transform(texts(test_examples))

    validation_positives, validation_prefix_texts, validation_owners = build_prefix_batch(
        validation_examples
    )
    test_positives, test_prefix_texts, test_owners = build_prefix_batch(test_examples)
    validation_prefix_sparse = feature_union.transform(validation_prefix_texts)
    test_prefix_sparse = feature_union.transform(test_prefix_texts)

    LOGGER.info("Fitting shared %s-dimensional SVD representation", config["svd"]["components"])
    svd = build_svd(config)
    train_dense = svd.fit_transform(train_sparse).astype(np.float32)
    validation_dense = svd.transform(validation_sparse).astype(np.float32)
    test_dense = svd.transform(test_sparse).astype(np.float32)
    validation_prefix_dense = svd.transform(validation_prefix_sparse).astype(np.float32)
    test_prefix_dense = svd.transform(test_prefix_sparse).astype(np.float32)

    y_train = np.asarray(labels(train_examples), dtype=np.int8)
    y_validation = np.asarray(labels(validation_examples), dtype=np.int8)
    weights = np.asarray(sample_weights(train_examples), dtype=np.float32)
    scale_pos_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    maximum_fpr = float(protocol["primary_selection"]["maximum"])
    exit_ratio = float(protocol["hysteresis"]["exit_threshold_ratio"])
    seeds = [int(seed) for seed in config["seeds"]]

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_union, args.artifact_dir / "feature_union.joblib", compress=3)
    joblib.dump(svd, args.artifact_dir / "svd.joblib", compress=3)

    runs: dict[str, list[dict[str, Any]]] = {"sgd": [], "xgboost": []}
    models: dict[tuple[str, int], Any] = {}
    for seed in seeds:
        LOGGER.info("Training SGD seed %s", seed)
        sgd = build_sgd(config, seed)
        sgd.fit(train_sparse, y_train, sample_weight=weights)
        models[("sgd", seed)] = sgd
        model_path = args.artifact_dir / f"sgd_seed_{seed}.joblib"
        joblib.dump(sgd, model_path, compress=3)
        runs["sgd"].append(
            evaluate_seed(
                "sgd", seed, sgd,
                validation_sparse, test_sparse,
                validation_examples, test_examples,
                validation_primary_indices, test_primary_indices,
                validation_prefix_sparse, validation_positives, validation_owners,
                test_prefix_sparse, test_positives, test_owners,
                maximum_fpr, exit_ratio,
            )
        )

        LOGGER.info("Training XGBoost seed %s", seed)
        xgb = build_xgboost(config, seed, scale_pos_weight)
        xgb.fit(
            train_dense,
            y_train,
            sample_weight=weights,
            eval_set=[(validation_dense, y_validation)],
            verbose=False,
        )
        models[("xgboost", seed)] = xgb
        model_path = args.artifact_dir / f"xgboost_seed_{seed}.joblib"
        joblib.dump(xgb, model_path, compress=3)
        runs["xgboost"].append(
            evaluate_seed(
                "xgboost", seed, xgb,
                validation_dense, test_dense,
                validation_examples, test_examples,
                validation_primary_indices, test_primary_indices,
                validation_prefix_dense, validation_positives, validation_owners,
                test_prefix_dense, test_positives, test_owners,
                maximum_fpr, exit_ratio,
            )
        )

    representation_bytes = sum(
        (args.artifact_dir / name).stat().st_size
        for name in ("feature_union.joblib", "svd.joblib")
    )
    aggregates: dict[str, Any] = {}
    selection_input: dict[str, dict[str, float]] = {}
    for name in ("sgd", "xgboost"):
        model_bytes = sum(
            (args.artifact_dir / f"{name}_seed_{seed}.joblib").stat().st_size
            for seed in seeds
        ) // len(seeds)
        aggregate = aggregate_runs(runs[name], representation_bytes + model_bytes)
        aggregates[name] = aggregate
        selection_input[name] = {
            "recall_mean": aggregate["validation_recall"]["mean"],
            "recall_std": aggregate["validation_recall"]["standard_deviation"],
            "median_stable_turn_mean": aggregate["validation_median_stable_scammer_turn"]["mean"],
            "macro_f1_mean": aggregate["validation_macro_f1"]["mean"],
            "artifact_bytes": aggregate["artifact_bytes"],
        }

    selected_family = choose_family(selection_input)
    deployment_seed = int(config["deployment_seed"])
    selected_run = next(run for run in runs[selected_family] if run["seed"] == deployment_seed)
    selected_bundle = {
        "schema_version": "1.0.0",
        "model_family": selected_family,
        "seed": deployment_seed,
        "threshold": selected_run["operating_point"]["threshold"],
        "exit_threshold": selected_run["operating_point"]["threshold"] * exit_ratio,
        "feature_union": feature_union,
        "svd": svd if selected_family == "xgboost" else None,
        "model": models[(selected_family, deployment_seed)],
        "llm_used_for_detection": False,
    }
    selected_path = args.artifact_dir / "selected_detector.joblib"
    joblib.dump(selected_bundle, selected_path, compress=3)

    report = {
        "schema_version": "1.0.0",
        "run_id": config["run_id"],
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selected_family": selected_family,
        "deployment_seed": deployment_seed,
        "data_summary": asdict(load_summary),
        "primary_view_counts": {
            "validation": len(validation_primary_indices),
            "test": len(test_primary_indices),
        },
        "runs": runs,
        "aggregates": aggregates,
        "limitations": [
            "All positive labels in the current corpus are silver; this is a research prototype result.",
            "Synthetic and source-style shortcuts remain possible and are audited separately.",
            "The LLM honeypot is not used by either detector family.",
        ],
    }
    metadata = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "selected_detector_sha256": sha256_file(selected_path),
        "selected_detector_bytes": selected_path.stat().st_size,
        "selected_family": selected_family,
        "deployment_seed": deployment_seed,
    }
    write_json(args.report_dir / "metrics.json", report)
    write_json(args.report_dir / "run_metadata.json", metadata)
    write_json(args.report_dir / "config.json", config)
    LOGGER.info("Selected %s on validation only", selected_family)
    LOGGER.info("Saved selected detector %s", metadata["selected_detector_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
