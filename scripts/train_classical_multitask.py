"""Train CPU-feasible XGBoost auxiliary heads over causal multilingual windows."""

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
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, sha256_file, write_json  # noqa: E402
from arrestshield.classical_multitask import (  # noqa: E402
    aligned_probabilities,
    balanced_sample_weights,
    positive_probabilities,
    select_f1_threshold,
)
from arrestshield.multitask import (  # noqa: E402
    MultiTaskExample,
    create_context_windows,
    load_multitask_examples,
    source_balanced_training_sample,
)
from arrestshield.protocol import summarize_seed_values  # noqa: E402


LOGGER = logging.getLogger("arrestshield.train_classical_multitask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/model/classical_multitask.json")
    parser.add_argument("--conversations", type=Path, default=PROJECT_ROOT / "data/processed/conversations.jsonl")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data/splits/split_manifest.json")
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts/models/classical_multitask_v1")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/classical_multitask_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-train-conversations", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_from_artifact(artifact_dir: Path, target: Path) -> str:
    import os

    return os.path.relpath(target.resolve(), artifact_dir.resolve()).replace("\\", "/")


def class_counts(values: Sequence[int], names: Sequence[str]) -> dict[str, int]:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=len(names))
    return {name: int(counts[index]) for index, name in enumerate(names)}


def observed_class_mapping(values: Sequence[int]) -> tuple[list[int], dict[int, int]]:
    observed = sorted(int(value) for value in np.unique(np.asarray(values, dtype=np.int64)))
    if len(observed) < 2:
        raise ValueError("A multi-class head requires at least two observed training classes")
    return observed, {global_id: local_id for local_id, global_id in enumerate(observed)}


def remap_labels(values: Sequence[int], mapping: Mapping[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(value)] for value in values], dtype=np.int64)


def build_model(config: Mapping[str, Any], seed: int, objective: str, classes: int) -> XGBClassifier:
    values = config["xgboost"]
    kwargs: dict[str, Any] = {
        "objective": objective,
        "eval_metric": "mlogloss" if objective == "multi:softprob" else "logloss",
        "n_estimators": int(values["n_estimators"]),
        "learning_rate": float(values["learning_rate"]),
        "max_depth": int(values["max_depth"]),
        "min_child_weight": float(values["min_child_weight"]),
        "subsample": float(values["subsample"]),
        "colsample_bytree": float(values["colsample_bytree"]),
        "reg_alpha": float(values["reg_alpha"]),
        "reg_lambda": float(values["reg_lambda"]),
        "early_stopping_rounds": int(values["early_stopping_rounds"]),
        "tree_method": str(values["tree_method"]),
        "n_jobs": int(values["n_jobs"]),
        "random_state": int(seed),
    }
    if objective == "multi:softprob":
        kwargs["num_class"] = int(classes)
    return XGBClassifier(**kwargs)


def transform_examples(feature_union: Any, svd: Any, examples: Sequence[MultiTaskExample]) -> np.ndarray:
    sparse = feature_union.transform([item.text for item in examples])
    return svd.transform(sparse).astype(np.float32)


def multiclass_metrics(
    labels: Sequence[int], scores: np.ndarray, names: Sequence[str]
) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.argmax(scores, axis=1)
    report = classification_report(
        truth,
        predicted,
        labels=list(range(len(names))),
        target_names=list(names),
        zero_division=0,
        output_dict=True,
    )
    return {
        "examples": int(len(truth)),
        "macro_f1_all_manifest_classes": float(f1_score(truth, predicted, labels=list(range(len(names))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "per_class": {name: report[name] for name in names},
    }


def tactic_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, Any]:
    from sklearn.metrics import precision_recall_fscore_support

    truth = np.asarray(labels, dtype=np.int8)
    predicted = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0
    )
    return {
        "examples": int(len(truth)),
        "positives": int(truth.sum()),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positives": int(predicted.sum()),
    }


def subset_rows(examples: Sequence[MultiTaskExample], mask: Sequence[bool]) -> list[MultiTaskExample]:
    return [item for item, keep in zip(examples, mask) if keep]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.perf_counter()
    args = parse_args()
    config = read_json(args.config)
    label_config = read_json(PROJECT_ROOT / config["label_config_path"])
    if config.get("llm_used_for_detection") is not False:
        raise ValueError("Classical multi-task config must prohibit LLM detection")
    # Fail before expensive fitting if the selected output location is not writable.
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    data = load_multitask_examples(args.conversations, args.splits, label_config)
    sampled = source_balanced_training_sample(
        data["train"],
        float(label_config["data"]["maximum_negative_to_positive_ratio"]),
        float(label_config["data"]["mixed_source_negative_ratio"]),
    )
    if args.max_train_conversations:
        sampled = sampled[: int(args.max_train_conversations)]
    train = create_context_windows(
        sampled,
        label_config["data"]["positive_prefix_fractions"],
        label_config["data"]["negative_prefix_fractions"],
        int(label_config["data"]["maximum_windows_per_conversation"]),
        label_config["labels"]["stages"],
        float(label_config["data"]["unknown_tactic_negative_weight"]),
    )
    validation = data["validation"]
    test = data["test"]
    representation = config["shared_representation"]
    feature_path = PROJECT_ROOT / representation["feature_union_path"]
    svd_path = PROJECT_ROOT / representation["svd_path"]
    binary_path = PROJECT_ROOT / representation["binary_detector_path"]
    feature_union = joblib.load(feature_path)
    svd = joblib.load(svd_path)
    LOGGER.info("Transforming %s causal train windows with frozen train-only representation", len(train))
    train_matrix = transform_examples(feature_union, svd, train)
    validation_matrix = transform_examples(feature_union, svd, validation)
    test_matrix = transform_examples(feature_union, svd, test)

    labels = label_config["labels"]
    scam_types = list(labels["scam_types"])
    tactics = list(labels["tactics"])
    stages = list(labels["stages"])
    seeds = args.seeds or [int(value) for value in config["seeds"]]
    deployment_seed = int(config["deployment_seed"])
    if deployment_seed not in seeds:
        deployment_seed = seeds[0]
    maximum_class_weight = float(config["xgboost"]["maximum_class_weight"])
    base_train_weights = [item.sample_weight for item in train]
    type_train_labels = [item.scam_type_label for item in train]
    type_validation_labels = [item.scam_type_label for item in validation]
    type_test_labels = [item.scam_type_label for item in test]
    type_weights = balanced_sample_weights(type_train_labels, base_train_weights, maximum_class_weight)
    type_class_ids, type_class_map = observed_class_mapping(type_train_labels)
    type_train_local = remap_labels(type_train_labels, type_class_map)
    type_validation_fit_mask = np.asarray(
        [value in type_class_map for value in type_validation_labels], dtype=bool
    )
    type_validation_local = remap_labels(
        np.asarray(type_validation_labels)[type_validation_fit_mask], type_class_map
    )

    stage_train_mask = np.asarray([item.stage_mask > 0 for item in train], dtype=bool)
    stage_validation_mask = np.asarray([item.stage_mask > 0 for item in validation], dtype=bool)
    stage_test_mask = np.asarray([item.stage_mask > 0 for item in test], dtype=bool)
    stage_train = subset_rows(train, stage_train_mask)
    stage_validation = subset_rows(validation, stage_validation_mask)
    stage_test = subset_rows(test, stage_test_mask)
    stage_train_labels = [item.stage_label for item in stage_train]
    stage_validation_labels = [item.stage_label for item in stage_validation]
    stage_test_labels = [item.stage_label for item in stage_test]
    stage_weights = balanced_sample_weights(
        stage_train_labels,
        [item.sample_weight for item in stage_train],
        maximum_class_weight,
    )
    stage_class_ids, stage_class_map = observed_class_mapping(stage_train_labels)
    stage_train_local = remap_labels(stage_train_labels, stage_class_map)
    stage_validation_fit_mask = np.asarray(
        [value in stage_class_map for value in stage_validation_labels], dtype=bool
    )
    stage_validation_local = remap_labels(
        np.asarray(stage_validation_labels)[stage_validation_fit_mask], stage_class_map
    )

    tactic_support = {
        name: int(sum(item.tactic_labels[index] > 0.5 and item.tactic_mask[index] > 0 for item in train))
        for index, name in enumerate(tactics)
    }
    minimum_tactic_positives = int(config["tactics"]["minimum_positive_training_examples"])
    supported_tactics = [name for name in tactics if tactic_support[name] >= minimum_tactic_positives]
    unsupported_tactics = [name for name in tactics if name not in supported_tactics]
    LOGGER.info("Supported tactic heads: %s; unavailable: %s", supported_tactics, unsupported_tactics)

    runs: list[dict[str, Any]] = []
    deployment_models: dict[str, Any] | None = None
    deployment_thresholds: dict[str, float] | None = None
    for seed in seeds:
        LOGGER.info("Training classical multi-task seed %s", seed)
        type_model = build_model(config, seed, "multi:softprob", len(type_class_ids))
        type_model.fit(
            train_matrix,
            type_train_local,
            sample_weight=type_weights,
            eval_set=[(validation_matrix[type_validation_fit_mask], type_validation_local)],
            verbose=False,
        )
        type_validation_scores = aligned_probabilities(
            type_model, validation_matrix, len(scam_types), type_class_ids
        )
        type_test_scores = aligned_probabilities(
            type_model, test_matrix, len(scam_types), type_class_ids
        )

        stage_model = build_model(config, seed, "multi:softprob", len(stage_class_ids))
        stage_model.fit(
            train_matrix[stage_train_mask],
            stage_train_local,
            sample_weight=stage_weights,
            eval_set=[
                (
                    validation_matrix[stage_validation_mask][stage_validation_fit_mask],
                    stage_validation_local,
                )
            ],
            verbose=False,
        )
        stage_validation_scores = aligned_probabilities(
            stage_model,
            validation_matrix[stage_validation_mask],
            len(stages),
            stage_class_ids,
        )
        stage_test_scores = aligned_probabilities(
            stage_model,
            test_matrix[stage_test_mask],
            len(stages),
            stage_class_ids,
        )

        tactic_models: dict[str, Any] = {}
        tactic_thresholds: dict[str, float] = {}
        tactic_results: dict[str, Any] = {}
        for tactic_index, name in enumerate(tactics):
            if name not in supported_tactics:
                tactic_results[name] = {
                    "available": False,
                    "reason": "insufficient_positive_training_supervision",
                    "training_positives": tactic_support[name],
                }
                continue
            train_mask = np.asarray([item.tactic_mask[tactic_index] > 0 for item in train], dtype=bool)
            validation_mask = np.asarray([item.tactic_mask[tactic_index] > 0 for item in validation], dtype=bool)
            test_mask = np.asarray([item.tactic_mask[tactic_index] > 0 for item in test], dtype=bool)
            y_train = np.asarray([int(item.tactic_labels[tactic_index] > 0.5) for item in train])[train_mask]
            y_validation = np.asarray([int(item.tactic_labels[tactic_index] > 0.5) for item in validation])[validation_mask]
            y_test = np.asarray([int(item.tactic_labels[tactic_index] > 0.5) for item in test])[test_mask]
            base_weights = np.asarray([item.sample_weight * item.tactic_mask[tactic_index] for item in train], dtype=np.float32)[train_mask]
            weights = balanced_sample_weights(y_train, base_weights, maximum_class_weight)
            model = build_model(config, seed, "binary:logistic", 2)
            model.fit(
                train_matrix[train_mask],
                y_train,
                sample_weight=weights,
                eval_set=[(validation_matrix[validation_mask], y_validation)],
                verbose=False,
            )
            validation_scores = positive_probabilities(model, validation_matrix[validation_mask])
            test_scores = positive_probabilities(model, test_matrix[test_mask])
            selected = select_f1_threshold(y_validation, validation_scores)
            threshold = float(selected["threshold"])
            tactic_models[name] = model
            tactic_thresholds[name] = threshold
            tactic_results[name] = {
                "available": True,
                "training_positives": tactic_support[name],
                "validation_selection": selected,
                "test_supporting_only": tactic_metrics(y_test, test_scores, threshold),
            }

        run = {
            "seed": int(seed),
            "scam_type": {
                "validation": multiclass_metrics(type_validation_labels, type_validation_scores, scam_types),
                "test_supporting_only": multiclass_metrics(type_test_labels, type_test_scores, scam_types),
            },
            "stage": {
                "validation": multiclass_metrics(stage_validation_labels, stage_validation_scores, stages),
                "test_supporting_only": multiclass_metrics(stage_test_labels, stage_test_scores, stages),
            },
            "tactics": tactic_results,
        }
        runs.append(run)
        if seed == deployment_seed:
            deployment_models = {
                "schema_version": "1.0.0",
                "model_family": "xgboost_classical_multitask_auxiliary",
                "seed": int(seed),
                "scam_type_model": type_model,
                "scam_type_class_ids": type_class_ids,
                "stage_model": stage_model,
                "stage_class_ids": stage_class_ids,
                "tactic_models": tactic_models,
                "llm_used_for_detection": False,
            }
            deployment_thresholds = tactic_thresholds

    if deployment_models is None or deployment_thresholds is None:
        raise RuntimeError("Deployment seed was not trained")
    heads_path = args.artifact_dir / "classical_multitask_heads.joblib"
    joblib.dump(deployment_models, heads_path, compress=3)
    binary_bundle = joblib.load(binary_path)
    manifest = {
        "schema_version": "1.0.0",
        "model_family": "sgd_binary_plus_xgboost_auxiliary_heads",
        "selection_role": config["selection_role"],
        "seed": deployment_seed,
        "heads_file": heads_path.name,
        "binary_detector": relative_from_artifact(args.artifact_dir, binary_path),
        "binary_family": binary_bundle["model_family"],
        "binary_threshold": float(binary_bundle["threshold"]),
        "representation": {
            "feature_union": relative_from_artifact(args.artifact_dir, feature_path),
            "svd": relative_from_artifact(args.artifact_dir, svd_path),
            "fit_scope": "training_split_only",
        },
        "labels": labels,
        "supported_tactics": supported_tactics,
        "unsupported_tactics": unsupported_tactics,
        "tactic_thresholds": deployment_thresholds,
        "used_as_api_decision_source": False,
        "llm_used_for_detection": False,
    }
    manifest_path = args.artifact_dir / "manifest.json"
    write_json(manifest_path, manifest)
    aggregate = {
        "scam_type_validation_macro_f1": summarize_seed_values([run["scam_type"]["validation"]["macro_f1_all_manifest_classes"] for run in runs]),
        "stage_validation_macro_f1": summarize_seed_values([run["stage"]["validation"]["macro_f1_all_manifest_classes"] for run in runs]),
        "supported_tactic_validation_macro_f1": summarize_seed_values([
            float(np.mean([run["tactics"][name]["validation_selection"]["f1"] for name in supported_tactics]))
            for run in runs
        ]),
    }
    report = {
        "schema_version": "1.0.0",
        "run_id": config["run_id"],
        "selection_role": config["selection_role"],
        "selection_split": "validation",
        "test_used_for_selection": False,
        "data": {
            "train_conversations_after_source_balance": len(sampled),
            "causal_train_windows": len(train),
            "validation_conversations": len(validation),
            "test_conversations": len(test),
            "conversation_safe_split": True,
            "representation_fit_scope": "training_split_only",
            "scam_type_train_counts": class_counts(type_train_labels, scam_types),
            "stage_observed_train_counts": class_counts(stage_train_labels, stages),
            "tactic_positive_train_counts": tactic_support,
        },
        "supported_tactics": supported_tactics,
        "unsupported_tactics": unsupported_tactics,
        "runs": runs,
        "aggregate": aggregate,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "All current positive supervision is silver and 71% is synthetic.",
            "Six tactic labels have no positive supervision and are explicitly unavailable.",
            "Strict unseen-source binary and fusion audits fail the 5% FPR gate.",
            "This auxiliary artifact never supplies the API scam decision.",
        ],
        "llm_used_for_detection": False,
    }
    metadata = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "heads_sha256": sha256_file(heads_path),
        "heads_bytes": heads_path.stat().st_size,
        "manifest_sha256": sha256_file(manifest_path),
        "deployment_seed": deployment_seed,
        "llm_used_for_detection": False,
    }
    write_json(args.report_dir / "metrics.json", report)
    write_json(args.report_dir / "run_metadata.json", metadata)
    write_json(args.report_dir / "config.json", config)
    LOGGER.info("Completed classical multi-task run in %.1f seconds", report["runtime_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
