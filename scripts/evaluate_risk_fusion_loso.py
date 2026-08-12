"""Strictly refit and audit XGBoost fusion after excluding each source channel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, write_json  # noqa: E402
from arrestshield.data import ConversationExample, labels, load_examples, sample_weights, texts  # noqa: E402
from arrestshield.evaluation import binary_metrics  # noqa: E402
from arrestshield.ladder import (  # noqa: E402
    build_feature_union,
    build_prefix_batch,
    build_sgd,
    positive_scores,
    stable_latency_from_flat_scores,
)
from arrestshield.protocol import select_threshold_at_fpr, summarize_seed_values  # noqa: E402
from arrestshield.risk import build_risk_matrix  # noqa: E402


LOGGER = logging.getLogger("arrestshield.evaluate_risk_fusion_loso")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/evaluation/risk_fusion_loso.json",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=PROJECT_ROOT / "data/processed/conversations.jsonl",
    )
    parser.add_argument(
        "--splits", type=Path, default=PROJECT_ROOT / "data/splits/split_manifest.json"
    )
    parser.add_argument(
        "--views", type=Path, default=PROJECT_ROOT / "data/splits/evaluation_views.json"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports/risk_fusion_v1/loso_metrics.json",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_ids(
    examples: Sequence[ConversationExample], values: set[str]
) -> list[ConversationExample]:
    return [example for example in examples if example.conversation_id in values]


def primary_indices(
    examples: Sequence[ConversationExample], hard_negative_ids: set[str]
) -> list[int]:
    return [
        index
        for index, example in enumerate(examples)
        if example.label == 1 or example.conversation_id in hard_negative_ids
    ]


def subset(values: Sequence[Any] | np.ndarray, indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def oof_base_scores(
    examples: Sequence[ConversationExample],
    ladder_config: Mapping[str, Any],
    folds: int,
    seed: int,
    base_seed: int,
) -> np.ndarray:
    y = np.asarray(labels(examples), dtype=np.int8)
    weights = np.asarray(sample_weights(examples), dtype=np.float32)
    all_texts = texts(examples)
    scores = np.full(len(examples), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_indices, held_indices) in enumerate(splitter.split(np.zeros(len(y)), y)):
        representation = build_feature_union(ladder_config)
        fit_matrix = representation.fit_transform([all_texts[index] for index in fit_indices])
        held_matrix = representation.transform([all_texts[index] for index in held_indices])
        model = build_sgd(ladder_config, base_seed + fold)
        model.fit(fit_matrix, y[fit_indices], sample_weight=weights[fit_indices])
        scores[held_indices] = positive_scores(model, held_matrix)
        del representation, fit_matrix, held_matrix, model
        gc.collect()
    if np.isnan(scores).any():
        raise RuntimeError("OOF scores are incomplete")
    return scores


def build_xgboost(
    risk_config: Mapping[str, Any], seed: int, scale_pos_weight: float
) -> XGBClassifier:
    values = risk_config["xgboost"]
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=int(values["n_estimators"]),
        learning_rate=float(values["learning_rate"]),
        max_depth=int(values["max_depth"]),
        min_child_weight=float(values["min_child_weight"]),
        subsample=float(values["subsample"]),
        colsample_bytree=float(values["colsample_bytree"]),
        reg_alpha=float(values["reg_alpha"]),
        reg_lambda=float(values["reg_lambda"]),
        early_stopping_rounds=int(values["early_stopping_rounds"]),
        tree_method=str(values["tree_method"]),
        n_jobs=int(values["n_jobs"]),
        scale_pos_weight=float(scale_pos_weight),
        random_state=seed,
    )


def aggregate_source_results(
    fold_results: Mapping[str, Sequence[Mapping[str, Any]]], seeds: Sequence[int]
) -> tuple[dict[str, Any], dict[int, Any]]:
    by_seed: dict[int, Any] = {}
    for seed in seeds:
        runs = [
            next(run for run in source_runs if int(run["seed"]) == seed)
            for source_runs in fold_results.values()
        ]
        by_seed[seed] = {
            "source_macro_recall": float(
                np.mean([run["held_out_test"]["recall"] for run in runs])
            ),
            "source_macro_fpr": float(
                np.mean([run["held_out_test"]["false_positive_rate"] for run in runs])
            ),
            "source_macro_f1": float(
                np.mean([run["held_out_test"]["macro_f1"] for run in runs])
            ),
            "maximum_source_fpr": float(
                max(run["held_out_test"]["false_positive_rate"] for run in runs)
            ),
        }
    aggregate = {
        "source_macro_recall": summarize_seed_values(
            [row["source_macro_recall"] for row in by_seed.values()]
        ),
        "source_macro_fpr": summarize_seed_values(
            [row["source_macro_fpr"] for row in by_seed.values()]
        ),
        "source_macro_f1": summarize_seed_values(
            [row["source_macro_f1"] for row in by_seed.values()]
        ),
        "maximum_source_fpr": summarize_seed_values(
            [row["maximum_source_fpr"] for row in by_seed.values()]
        ),
    }
    return aggregate, by_seed


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    audit_config = read_json(args.config)
    ladder_config = read_json(PROJECT_ROOT / audit_config["model_ladder_config_path"])
    risk_config = read_json(PROJECT_ROOT / audit_config["risk_fusion_config_path"])
    protocol = read_json(PROJECT_ROOT / ladder_config["protocol_path"])
    views = read_json(args.views)
    data_config = risk_config["data"]
    by_split, load_summary = load_examples(
        args.conversations,
        args.splits,
        allowed_provenance=data_config["allowed_label_provenance"],
        provenance_weights=data_config["provenance_weights"],
    )
    hard_negative_validation = set(views["hard_negative_ids"]["validation"])
    maximum_fpr = float(audit_config["gate"]["maximum_fpr_each_held_source"])
    exit_ratio = float(protocol["hysteresis"]["exit_threshold_ratio"])
    seeds = [int(value) for value in audit_config["fusion_seeds"]]
    fold_results: dict[str, list[dict[str, Any]]] = {}

    for held_source in views["mixed_sources"]:
        LOGGER.info("Strict risk-fusion holdout: %s", held_source)
        fold = views["leave_one_mixed_source_out"][held_source]
        train_examples = select_ids(by_split["train"], set(fold["train_ids"]))
        validation_examples = select_ids(by_split["validation"], set(fold["validation_ids"]))
        test_examples = select_ids(by_split["test"], set(fold["test_ids"]))
        for name, examples in (
            ("train", train_examples),
            ("validation", validation_examples),
            ("held_test", test_examples),
        ):
            if set(labels(examples)) != {0, 1}:
                raise ValueError(f"{held_source} {name} partition lacks both classes")

        train_oof_scores = oof_base_scores(
            train_examples,
            ladder_config,
            folds=int(audit_config["oof_folds"]),
            seed=int(audit_config["oof_seed"]),
            base_seed=int(audit_config["base_deployment_seed"]),
        )
        representation = build_feature_union(ladder_config)
        train_sparse = representation.fit_transform(texts(train_examples))
        validation_sparse = representation.transform(texts(validation_examples))
        test_sparse = representation.transform(texts(test_examples))
        positives, prefix_texts, owners = build_prefix_batch(test_examples)
        prefix_sparse = representation.transform(prefix_texts)
        base_model = build_sgd(ladder_config, int(audit_config["base_deployment_seed"]))
        y_train = np.asarray(labels(train_examples), dtype=np.int8)
        weights = np.asarray(sample_weights(train_examples), dtype=np.float32)
        base_model.fit(train_sparse, y_train, sample_weight=weights)
        validation_base = positive_scores(base_model, validation_sparse)
        test_base = positive_scores(base_model, test_sparse)
        prefix_base = positive_scores(base_model, prefix_sparse)

        train_matrix = build_risk_matrix(texts(train_examples), train_oof_scores)
        validation_matrix = build_risk_matrix(texts(validation_examples), validation_base)
        test_matrix = build_risk_matrix(texts(test_examples), test_base)
        prefix_matrix = build_risk_matrix(prefix_texts, prefix_base)
        y_validation = np.asarray(labels(validation_examples), dtype=np.int8)
        validation_primary = primary_indices(validation_examples, hard_negative_validation)
        if {validation_examples[index].label for index in validation_primary} != {0, 1}:
            raise ValueError(f"{held_source} validation primary view lacks both classes")
        raw_scale = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
        scale_pos_weight = min(
            raw_scale, float(risk_config["xgboost"]["maximum_scale_pos_weight"])
        )
        source_runs: list[dict[str, Any]] = []
        for seed in seeds:
            model = build_xgboost(risk_config, seed, scale_pos_weight)
            model.fit(
                train_matrix,
                y_train,
                sample_weight=weights,
                eval_set=[(validation_matrix, y_validation)],
                verbose=False,
            )
            validation_scores = positive_scores(model, validation_matrix)
            operating = select_threshold_at_fpr(
                subset(labels(validation_examples), validation_primary),
                subset(validation_scores, validation_primary),
                maximum_fpr,
            )
            test_scores = positive_scores(model, test_matrix)
            prefix_scores = positive_scores(model, prefix_matrix)
            source_runs.append(
                {
                    "seed": seed,
                    "best_iteration": int(model.best_iteration),
                    "operating_point": asdict(operating),
                    "held_out_test": binary_metrics(
                        labels(test_examples), test_scores, operating.threshold
                    ),
                    "stable_latency": stable_latency_from_flat_scores(
                        positives,
                        owners,
                        prefix_scores,
                        operating.threshold,
                        operating.threshold * exit_ratio,
                    ),
                }
            )
        fold_results[held_source] = source_runs
        del representation, train_sparse, validation_sparse, test_sparse, prefix_sparse
        del train_matrix, validation_matrix, test_matrix, prefix_matrix, base_model
        gc.collect()

    aggregate, by_seed = aggregate_source_results(fold_results, seeds)
    allowed = float(audit_config["gate"]["maximum_fpr_each_held_source"])
    all_source_runs_pass = all(
        run["held_out_test"]["false_positive_rate"] <= allowed + 1e-12
        for runs in fold_results.values()
        for run in runs
    )
    report = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "audit_config": audit_config,
        "data_summary": asdict(load_summary),
        "protocol": (
            "For each mixed-label source, exclude it before fitting every TF-IDF representation, "
            "base detector, OOF fusion feature and XGBoost model; select threshold only on the "
            "source-excluded validation hard-negative view."
        ),
        "held_source_used_for_threshold_selection": False,
        "folds": fold_results,
        "macro_by_seed": by_seed,
        "aggregate": aggregate,
        "gate": {
            "maximum_fpr_each_held_source": allowed,
            "all_source_seed_runs_pass": all_source_runs_pass,
            "passed": all_source_runs_pass,
        },
        "interpretation": (
            "This audit is more credible than mixed-source test performance but remains a "
            "silver/synthetic corpus result, not a human deployment claim."
        ),
        "llm_used_for_detection": False,
    }
    write_json(args.report, report)
    LOGGER.info("Strict risk-fusion aggregate: %s", aggregate)
    LOGGER.info("Strict per-source FPR gate passed: %s", all_source_runs_pass)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
