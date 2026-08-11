"""Run strict leave-one-mixed-source-out evaluation for SGD and XGBoost."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, write_json  # noqa: E402
from arrestshield.data import ConversationExample, labels, load_examples, sample_weights, texts  # noqa: E402
from arrestshield.evaluation import binary_metrics  # noqa: E402
from arrestshield.ladder import (  # noqa: E402
    build_feature_union,
    build_prefix_batch,
    build_sgd,
    build_svd,
    build_xgboost,
    positive_scores,
    stable_latency_from_flat_scores,
)
from arrestshield.protocol import select_threshold_at_fpr, summarize_seed_values  # noqa: E402


LOGGER = logging.getLogger("arrestshield.evaluate_loso")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/model/model_ladder.json")
    parser.add_argument("--conversations", type=Path, default=PROJECT_ROOT / "data/processed/conversations.jsonl")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data/splits/split_manifest.json")
    parser.add_argument("--views", type=Path, default=PROJECT_ROOT / "data/splits/evaluation_views.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/model_ladder_v1/loso_metrics.json")
    parser.add_argument("--max-per-split", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_ids(examples: Sequence[ConversationExample], ids: set[str]) -> list[ConversationExample]:
    return [example for example in examples if example.conversation_id in ids]


def primary_validation_indices(
    examples: Sequence[ConversationExample], hard_negative_ids: set[str]
) -> list[int]:
    return [
        index
        for index, example in enumerate(examples)
        if example.label == 1 or example.conversation_id in hard_negative_ids
    ]


def evaluate_fold_model(
    name: str,
    seed: int,
    model: Any,
    validation_matrix: Any,
    test_matrix: Any,
    validation_examples: Sequence[ConversationExample],
    test_examples: Sequence[ConversationExample],
    validation_indices: Sequence[int],
    prefix_matrix: Any,
    positives: Sequence[ConversationExample],
    owners: Sequence[tuple[int, int]],
    maximum_fpr: float,
    exit_ratio: float,
) -> dict[str, Any]:
    validation_scores = positive_scores(model, validation_matrix)
    operating = select_threshold_at_fpr(
        [validation_examples[index].label for index in validation_indices],
        [validation_scores[index] for index in validation_indices],
        maximum_fpr,
    )
    test_scores = positive_scores(model, test_matrix)
    prefix_scores = positive_scores(model, prefix_matrix)
    return {
        "model": name,
        "seed": seed,
        "operating_point": asdict(operating),
        "held_out_test": binary_metrics(labels(test_examples), test_scores, operating.threshold),
        "stable_latency": stable_latency_from_flat_scores(
            positives,
            owners,
            prefix_scores,
            operating.threshold,
            operating.threshold * exit_ratio,
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = read_json(args.config)
    protocol = read_json(PROJECT_ROOT / config["protocol_path"])
    views = read_json(args.views)
    by_split, _ = load_examples(
        args.conversations,
        args.splits,
        allowed_provenance=config["data"]["allowed_label_provenance"],
        provenance_weights=config["data"]["provenance_weights"],
        max_per_split=args.max_per_split,
    )
    hard_negative_validation = set(views["hard_negative_ids"]["validation"])
    maximum_fpr = float(protocol["primary_selection"]["maximum"])
    exit_ratio = float(protocol["hysteresis"]["exit_threshold_ratio"])
    seeds = [int(seed) for seed in config["seeds"]]
    fold_results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for held_out_source in views["mixed_sources"]:
        LOGGER.info("Strict holdout source: %s", held_out_source)
        fold = views["leave_one_mixed_source_out"][held_out_source]
        train_examples = select_ids(by_split["train"], set(fold["train_ids"]))
        validation_examples = select_ids(by_split["validation"], set(fold["validation_ids"]))
        test_examples = select_ids(by_split["test"], set(fold["test_ids"]))
        if set(labels(train_examples)) != {0, 1} or set(labels(validation_examples)) != {0, 1}:
            raise ValueError(f"Fold {held_out_source} lacks both classes in train or validation")
        if set(labels(test_examples)) != {0, 1}:
            if args.max_per_split:
                LOGGER.warning(
                    "Skipping %s in capped smoke mode because the sampled held-out set lacks both classes",
                    held_out_source,
                )
                continue
            raise ValueError(f"Held-out test source {held_out_source} lacks both classes")

        feature_union = build_feature_union(config)
        train_sparse = feature_union.fit_transform(texts(train_examples))
        validation_sparse = feature_union.transform(texts(validation_examples))
        test_sparse = feature_union.transform(texts(test_examples))
        positives, prefix_texts, owners = build_prefix_batch(test_examples)
        prefix_sparse = feature_union.transform(prefix_texts)

        svd = build_svd(config)
        train_dense = svd.fit_transform(train_sparse).astype(np.float32)
        validation_dense = svd.transform(validation_sparse).astype(np.float32)
        test_dense = svd.transform(test_sparse).astype(np.float32)
        prefix_dense = svd.transform(prefix_sparse).astype(np.float32)

        y_train = np.asarray(labels(train_examples), dtype=np.int8)
        y_validation = np.asarray(labels(validation_examples), dtype=np.int8)
        weights = np.asarray(sample_weights(train_examples), dtype=np.float32)
        scale_pos_weight = float(np.sum(y_train == 0) / np.sum(y_train == 1))
        validation_indices = primary_validation_indices(
            validation_examples, hard_negative_validation
        )
        fold_results[held_out_source] = {"sgd": [], "xgboost": []}

        for seed in seeds:
            sgd = build_sgd(config, seed)
            sgd.fit(train_sparse, y_train, sample_weight=weights)
            fold_results[held_out_source]["sgd"].append(
                evaluate_fold_model(
                    "sgd", seed, sgd,
                    validation_sparse, test_sparse,
                    validation_examples, test_examples, validation_indices,
                    prefix_sparse, positives, owners,
                    maximum_fpr, exit_ratio,
                )
            )

            xgb = build_xgboost(config, seed, scale_pos_weight)
            xgb.fit(
                train_dense,
                y_train,
                sample_weight=weights,
                eval_set=[(validation_dense, y_validation)],
                verbose=False,
            )
            fold_results[held_out_source]["xgboost"].append(
                evaluate_fold_model(
                    "xgboost", seed, xgb,
                    validation_dense, test_dense,
                    validation_examples, test_examples, validation_indices,
                    prefix_dense, positives, owners,
                    maximum_fpr, exit_ratio,
                )
            )

        del train_sparse, validation_sparse, test_sparse, prefix_sparse
        del train_dense, validation_dense, test_dense, prefix_dense
        gc.collect()

    evaluated_sources = sorted(fold_results)
    if not evaluated_sources:
        raise RuntimeError("No valid leave-one-source-out folds were evaluated")
    macro_by_seed: dict[str, dict[int, dict[str, float]]] = {"sgd": {}, "xgboost": {}}
    for model_name in ("sgd", "xgboost"):
        for seed in seeds:
            seed_runs = [
                next(run for run in fold_results[source][model_name] if run["seed"] == seed)
                for source in evaluated_sources
            ]
            macro_by_seed[model_name][seed] = {
                "source_macro_recall": float(np.mean([run["held_out_test"]["recall"] for run in seed_runs])),
                "source_macro_fpr": float(np.mean([run["held_out_test"]["false_positive_rate"] for run in seed_runs])),
                "source_macro_f1": float(np.mean([run["held_out_test"]["macro_f1"] for run in seed_runs])),
            }

    aggregates: dict[str, Any] = {}
    for model_name in ("sgd", "xgboost"):
        seed_values = list(macro_by_seed[model_name].values())
        aggregates[model_name] = {
            "source_macro_recall": summarize_seed_values([value["source_macro_recall"] for value in seed_values]),
            "source_macro_fpr": summarize_seed_values([value["source_macro_fpr"] for value in seed_values]),
            "source_macro_f1": summarize_seed_values([value["source_macro_f1"] for value in seed_values]),
        }

    report = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "protocol": "strict feature refit and classifier retraining after excluding each mixed-label source channel",
        "test_used_for_threshold_selection": False,
        "mixed_sources": views["mixed_sources"],
        "evaluated_sources": evaluated_sources,
        "folds": fold_results,
        "macro_by_seed": macro_by_seed,
        "aggregates": aggregates,
        "interpretation": (
            "This is the primary dataset-origin shortcut audit. Results are not production claims "
            "because the held sources remain mostly synthetic or silver-labeled."
        ),
    }
    write_json(args.report, report)
    LOGGER.info("Wrote %s", args.report)
    LOGGER.info("LOSO aggregates: %s", aggregates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
