"""Train an out-of-fold XGBoost fusion layer over deterministic risk signals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, sha256_file, write_json  # noqa: E402
from arrestshield.data import (  # noqa: E402
    ConversationExample,
    labels,
    load_examples,
    sample_weights,
    texts,
)
from arrestshield.evaluation import binary_metrics, grouped_metrics  # noqa: E402
from arrestshield.ladder import (  # noqa: E402
    build_prefix_batch,
    positive_scores,
    stable_latency_from_flat_scores,
)
from arrestshield.protocol import select_threshold_at_fpr, summarize_seed_values  # noqa: E402
from arrestshield.risk import (  # noqa: E402
    RISK_FEATURE_NAMES,
    build_risk_matrix,
    risk_scores,
)


LOGGER = logging.getLogger("arrestshield.train_risk_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/model/risk_fusion.json"
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
        "--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts/models/risk_fusion_v1"
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports/risk_fusion_v1"
    )
    parser.add_argument("--max-per-split", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def subset(values: Sequence[Any] | np.ndarray, indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def primary_indices(
    examples: Sequence[ConversationExample], hard_negative_ids: set[str]
) -> list[int]:
    return [
        index
        for index, example in enumerate(examples)
        if example.label == 1 or example.conversation_id in hard_negative_ids
    ]


def detector_matrix(bundle: Mapping[str, Any], values: Sequence[str]) -> Any:
    matrix = bundle["feature_union"].transform(list(values))
    if bundle.get("svd") is not None:
        matrix = bundle["svd"].transform(matrix).astype(np.float32)
    return matrix


def detector_scores(bundle: Mapping[str, Any], values: Sequence[str]) -> np.ndarray:
    return positive_scores(bundle["model"], detector_matrix(bundle, values))


def out_of_fold_detector_scores(
    detector_bundle: Mapping[str, Any],
    training_texts: Sequence[str],
    training_labels: Sequence[int],
    training_weights: Sequence[float],
    folds: int,
    seed: int,
) -> np.ndarray:
    """Prevent the fusion learner from seeing its row in model or representation fitting."""
    y = np.asarray(training_labels, dtype=np.int8)
    weights = np.asarray(training_weights, dtype=np.float32)
    output = np.full(len(y), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_indices, held_indices) in enumerate(splitter.split(np.zeros(len(y)), y)):
        representation = clone(detector_bundle["feature_union"])
        fit_texts = [training_texts[index] for index in fit_indices]
        held_texts = [training_texts[index] for index in held_indices]
        fit_matrix = representation.fit_transform(fit_texts)
        held_matrix = representation.transform(held_texts)
        if detector_bundle.get("svd") is not None:
            reducer = clone(detector_bundle["svd"])
            fit_matrix = reducer.fit_transform(fit_matrix).astype(np.float32)
            held_matrix = reducer.transform(held_matrix).astype(np.float32)
        model = clone(detector_bundle["model"])
        parameters = model.get_params(deep=False)
        if "random_state" in parameters:
            model.set_params(random_state=seed + fold)
        model.fit(fit_matrix, y[fit_indices], sample_weight=weights[fit_indices])
        output[held_indices] = positive_scores(model, held_matrix)
    if np.isnan(output).any():
        raise RuntimeError("OOF detector scores were not produced for every training row")
    return output


def build_xgboost(config: Mapping[str, Any], seed: int, scale_pos_weight: float) -> XGBClassifier:
    values = config["xgboost"]
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


def aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "validation_recall": summarize_seed_values(
            [float(run["validation_primary"]["recall"]) for run in runs]
        ),
        "validation_false_positive_rate": summarize_seed_values(
            [float(run["validation_primary"]["false_positive_rate"]) for run in runs]
        ),
        "validation_macro_f1": summarize_seed_values(
            [float(run["validation_primary"]["macro_f1"]) for run in runs]
        ),
        "test_recall_supporting_only": summarize_seed_values(
            [float(run["test_primary"]["recall"]) for run in runs]
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = read_json(args.config)
    protocol = read_json(PROJECT_ROOT / config["protocol_path"])
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
    validation_primary = primary_indices(
        validation_examples, set(views["hard_negative_ids"]["validation"])
    )
    test_primary = primary_indices(test_examples, set(views["hard_negative_ids"]["test"]))
    if {validation_examples[index].label for index in validation_primary} != {0, 1}:
        raise ValueError("Validation primary view must contain both classes")

    detector_path = PROJECT_ROOT / config["base_detector_path"]
    detector_bundle = joblib.load(detector_path)
    if detector_bundle.get("llm_used_for_detection") is not False:
        raise ValueError("Base detector does not explicitly prohibit LLM detection")

    LOGGER.info("Generating leakage-safe out-of-fold base detector scores")
    train_base_scores = out_of_fold_detector_scores(
        detector_bundle,
        texts(train_examples),
        labels(train_examples),
        sample_weights(train_examples),
        folds=int(data_config["oof_folds"]),
        seed=int(data_config["oof_seed"]),
    )
    validation_base_scores = detector_scores(detector_bundle, texts(validation_examples))
    test_base_scores = detector_scores(detector_bundle, texts(test_examples))

    LOGGER.info("Extracting %d fixed risk-fusion features", len(RISK_FEATURE_NAMES))
    train_matrix = build_risk_matrix(texts(train_examples), train_base_scores)
    validation_matrix = build_risk_matrix(texts(validation_examples), validation_base_scores)
    test_matrix = build_risk_matrix(texts(test_examples), test_base_scores)
    y_train = np.asarray(labels(train_examples), dtype=np.int8)
    y_validation = np.asarray(labels(validation_examples), dtype=np.int8)
    weights = np.asarray(sample_weights(train_examples), dtype=np.float32)

    validation_positives, validation_prefix_texts, validation_owners = build_prefix_batch(
        validation_examples
    )
    test_positives, test_prefix_texts, test_owners = build_prefix_batch(test_examples)
    validation_prefix_base = detector_scores(detector_bundle, validation_prefix_texts)
    test_prefix_base = detector_scores(detector_bundle, test_prefix_texts)
    validation_prefix_matrix = build_risk_matrix(validation_prefix_texts, validation_prefix_base)
    test_prefix_matrix = build_risk_matrix(test_prefix_texts, test_prefix_base)

    maximum_fpr = float(protocol["primary_selection"]["maximum"])
    exit_ratio = float(protocol["hysteresis"]["exit_threshold_ratio"])
    raw_scale = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    scale_pos_weight = min(
        raw_scale, float(config["xgboost"]["maximum_scale_pos_weight"])
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    models: dict[int, Any] = {}
    for seed in [int(value) for value in config["seeds"]]:
        LOGGER.info("Training risk-fusion XGBoost seed %d", seed)
        model = build_xgboost(config, seed, scale_pos_weight)
        model.fit(
            train_matrix,
            y_train,
            sample_weight=weights,
            eval_set=[(validation_matrix, y_validation)],
            verbose=False,
        )
        models[seed] = model
        joblib.dump(model, args.artifact_dir / f"risk_fusion_seed_{seed}.joblib", compress=3)
        validation_scores = positive_scores(model, validation_matrix)
        test_scores = positive_scores(model, test_matrix)
        operating = select_threshold_at_fpr(
            subset(labels(validation_examples), validation_primary),
            subset(validation_scores, validation_primary),
            maximum_fpr,
        )
        threshold = operating.threshold
        validation_prefix_scores = positive_scores(model, validation_prefix_matrix)
        test_prefix_scores = positive_scores(model, test_prefix_matrix)
        runs.append(
            {
                "seed": seed,
                "best_iteration": int(model.best_iteration),
                "operating_point": asdict(operating),
                "validation_primary": binary_metrics(
                    subset(labels(validation_examples), validation_primary),
                    subset(validation_scores, validation_primary),
                    threshold,
                ),
                "validation_stable_latency": stable_latency_from_flat_scores(
                    validation_positives,
                    validation_owners,
                    validation_prefix_scores,
                    threshold,
                    threshold * exit_ratio,
                ),
                "test_primary": binary_metrics(
                    subset(labels(test_examples), test_primary),
                    subset(test_scores, test_primary),
                    threshold,
                ),
                "test_full": binary_metrics(labels(test_examples), test_scores, threshold),
                "test_slices": grouped_metrics(test_examples, test_scores, threshold),
                "test_stable_latency": stable_latency_from_flat_scores(
                    test_positives,
                    test_owners,
                    test_prefix_scores,
                    threshold,
                    threshold * exit_ratio,
                ),
            }
        )

    deployment_seed = int(config["deployment_seed"])
    selected_run = next(run for run in runs if run["seed"] == deployment_seed)
    selected_model = models[deployment_seed]
    selected_bundle = {
        "schema_version": "1.0.0",
        "model_family": "xgboost_risk_fusion",
        "seed": deployment_seed,
        "threshold": selected_run["operating_point"]["threshold"],
        "exit_threshold": selected_run["operating_point"]["threshold"] * exit_ratio,
        "feature_names": list(RISK_FEATURE_NAMES),
        "base_detector_sha256": sha256_file(detector_path),
        "model": selected_model,
        "promotion_status": "research_only_not_promoted",
        "llm_used_for_detection": False,
    }
    selected_path = args.artifact_dir / "risk_fusion.joblib"
    joblib.dump(selected_bundle, selected_path, compress=3)

    base_threshold = float(detector_bundle["threshold"])
    base_validation = binary_metrics(
        subset(labels(validation_examples), validation_primary),
        subset(validation_base_scores, validation_primary),
        base_threshold,
    )
    base_test = binary_metrics(
        subset(labels(test_examples), test_primary),
        subset(test_base_scores, test_primary),
        base_threshold,
    )
    selected_validation = selected_run["validation_primary"]
    human_status_path = PROJECT_ROOT / "data/human_test/COLLECTION_STATUS.json"
    human_status = read_json(human_status_path)
    promotion_checks = {
        "validation_fpr_at_most_5_percent": bool(
            selected_validation["false_positive_rate"] <= maximum_fpr + 1e-12
        ),
        "validation_recall_not_below_base": bool(
            selected_validation["recall"] >= base_validation["recall"]
        ),
        "frozen_human_gold_available": bool(human_status.get("human_gold", False)),
    }
    promotion_passed = all(promotion_checks.values())
    feature_importance = sorted(
        (
            {"feature": name, "importance": float(importance)}
            for name, importance in zip(RISK_FEATURE_NAMES, selected_model.feature_importances_)
        ),
        key=lambda row: (-row["importance"], row["feature"]),
    )
    report = {
        "schema_version": "1.0.0",
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "method": "XGBoost over OOF base-detector score plus deterministic lexical/entity signals",
        "oof_base_score_generation": {
            "folds": int(data_config["oof_folds"]),
            "seed": int(data_config["oof_seed"]),
            "every_training_row_scored_by_model_not_fitted_on_that_row": True,
            "text_representation_refit_inside_each_fold": True,
        },
        "data_summary": asdict(load_summary),
        "feature_names": list(RISK_FEATURE_NAMES),
        "scale_pos_weight": scale_pos_weight,
        "base_detector": {
            "sha256": sha256_file(detector_path),
            "validation_primary": base_validation,
            "test_primary_supporting_only": base_test,
        },
        "runs": runs,
        "aggregate": aggregate(runs),
        "deployment_seed": deployment_seed,
        "feature_importance": feature_importance,
        "promotion": {
            "checks": promotion_checks,
            "passed": promotion_passed,
            "status": "eligible" if promotion_passed else "research_only_not_promoted",
        },
        "limitations": [
            "All current positive supervision is silver and 71% is synthetic.",
            "The frozen human gold test set is not collected, so this model cannot be promoted.",
            "Lexical tactic features are deterministic indicators, not substitutes for the trained transformer tactic head.",
            "No LLM is used for a feature, score, threshold, or scam decision.",
        ],
    }
    metadata = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "artifact_sha256": sha256_file(selected_path),
        "artifact_bytes": selected_path.stat().st_size,
        "base_detector_sha256": sha256_file(detector_path),
        "promotion_status": report["promotion"]["status"],
        "llm_used_for_detection": False,
    }
    write_json(args.report_dir / "metrics.json", report)
    write_json(args.report_dir / "run_metadata.json", metadata)
    write_json(args.report_dir / "config.json", config)
    LOGGER.info("Risk fusion saved with status %s", metadata["promotion_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
