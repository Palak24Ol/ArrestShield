"""CPU-feasible multilingual auxiliary heads for scam type, tactics, and stage."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

from .asr_evaluation import frozen_detector_scores


def balanced_sample_weights(
    labels: Sequence[int],
    base_weights: Sequence[float],
    maximum_class_weight: float = 12.0,
) -> np.ndarray:
    """Return square-root inverse-frequency weights with a defensive cap."""
    values = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(base_weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("labels and base_weights must be aligned one-dimensional arrays")
    classes, counts = np.unique(values, return_counts=True)
    if len(classes) < 2:
        raise ValueError("At least two observed classes are required")
    total = float(len(values))
    class_count = float(len(classes))
    factors = {
        int(label): min(
            float(maximum_class_weight),
            max(0.5, float(np.sqrt(total / (class_count * float(count))))),
        )
        for label, count in zip(classes, counts)
    }
    return np.asarray(
        [weight * factors[int(label)] for label, weight in zip(values, weights)],
        dtype=np.float32,
    )


def aligned_probabilities(
    model: Any,
    matrix: Any,
    class_count: int,
    output_class_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Align predict_proba columns to manifest class indices."""
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    classes = (
        [int(value) for value in output_class_ids]
        if output_class_ids is not None
        else [int(value) for value in model.classes_]
    )
    if raw.shape[1] != len(classes):
        raise ValueError("Probability columns do not match the supplied class mapping")
    output = np.zeros((raw.shape[0], class_count), dtype=np.float64)
    for source_index, class_index in enumerate(classes):
        if class_index < 0 or class_index >= class_count:
            raise ValueError(f"Model exposes class outside manifest: {class_index}")
        output[:, class_index] = raw[:, source_index]
    row_sums = output.sum(axis=1, keepdims=True)
    return np.divide(output, row_sums, out=output, where=row_sums > 0)


def positive_probabilities(model: Any, matrix: Any) -> np.ndarray:
    """Extract the positive class from one binary auxiliary head."""
    classes = [int(value) for value in model.classes_]
    if 1 not in classes:
        raise ValueError("Binary auxiliary head has no positive class")
    return np.asarray(model.predict_proba(matrix)[:, classes.index(1)], dtype=np.float64)


def select_f1_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
) -> dict[str, float | int]:
    """Select a deterministic validation-only threshold by F1, then precision."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold selection requires both classes")
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.5, 1.0]),
                np.quantile(y_score, np.linspace(0.0, 1.0, 201)),
            ]
        )
    )
    best: tuple[float, float, float, float] | None = None
    payload: dict[str, float | int] | None = None
    for threshold in candidates:
        predicted = (y_score >= threshold).astype(np.int8)
        f1 = float(f1_score(y_true, predicted, zero_division=0))
        precision = float(precision_score(y_true, predicted, zero_division=0))
        recall = float(recall_score(y_true, predicted, zero_division=0))
        key = (f1, precision, recall, float(threshold))
        if best is None or key > best:
            best = key
            payload = {
                "threshold": float(threshold),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "examples": int(len(y_true)),
                "positives": int(y_true.sum()),
            }
    assert payload is not None
    return payload


def format_classical_multitask_outputs(
    manifest: Mapping[str, Any],
    binary_score: float,
    scam_type_scores: Sequence[float],
    tactic_scores: Mapping[str, float],
    stage_scores: Sequence[float],
) -> dict[str, Any]:
    labels = manifest["labels"]
    scam_types = list(labels["scam_types"])
    tactics = list(labels["tactics"])
    stages = list(labels["stages"])
    if len(scam_type_scores) != len(scam_types):
        raise ValueError("Scam-type output size does not match manifest")
    if len(stage_scores) != len(stages):
        raise ValueError("Stage output size does not match manifest")
    supported = set(manifest["supported_tactics"])
    if set(tactic_scores) != supported:
        raise ValueError("Tactic outputs do not match supported-tactic manifest")
    type_index = int(np.argmax(scam_type_scores))
    stage_index = int(np.argmax(stage_scores))
    thresholds = manifest["tactic_thresholds"]
    tactics_payload: dict[str, Any] = {}
    for name in tactics:
        if name in supported:
            score = float(tactic_scores[name])
            threshold = float(thresholds[name])
            tactics_payload[name] = {
                "available": True,
                "score": score,
                "threshold": threshold,
                "present": bool(score >= threshold),
            }
        else:
            tactics_payload[name] = {
                "available": False,
                "score": None,
                "threshold": None,
                "present": None,
                "reason": "no_positive_training_supervision",
            }
    binary_threshold = float(manifest["binary_threshold"])
    return {
        "signal_source": "trained_multilingual_classical_multitask",
        "selection_role": manifest.get("selection_role", "research_only_auxiliary"),
        "binary": {
            "score": float(binary_score),
            "threshold": binary_threshold,
            "is_scam": bool(float(binary_score) >= binary_threshold),
            "family": manifest["binary_family"],
        },
        "scam_type": {
            "label": scam_types[type_index],
            "score": float(scam_type_scores[type_index]),
            "scores": {
                name: float(score) for name, score in zip(scam_types, scam_type_scores)
            },
        },
        "tactics": tactics_payload,
        "stage": {
            "label": stages[stage_index],
            "score": float(stage_scores[stage_index]),
            "scores": {name: float(score) for name, score in zip(stages, stage_scores)},
        },
        "used_as_api_decision_source": False,
        "llm_used": False,
    }


class ClassicalMultiTaskPredictor:
    """Lazy predictor over shared train-only TF-IDF/SVD and XGBoost heads."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.manifest = json.loads(
            (self.artifact_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("llm_used_for_detection") is not False:
            raise ValueError("Classical multi-task manifest must prohibit LLM detection")
        self._lock = Lock()
        self._bundle: Mapping[str, Any] | None = None
        self._feature_union: Any | None = None
        self._svd: Any | None = None
        self._binary_bundle: Mapping[str, Any] | None = None

    def _resolve(self, value: str) -> Path:
        return (self.artifact_dir / value).resolve()

    def load(self) -> None:
        if self._bundle is not None:
            return
        bundle = joblib.load(self.artifact_dir / self.manifest["heads_file"])
        if bundle.get("llm_used_for_detection") is not False:
            raise ValueError("Classical multi-task heads must prohibit LLM detection")
        self._feature_union = joblib.load(
            self._resolve(self.manifest["representation"]["feature_union"])
        )
        self._svd = joblib.load(
            self._resolve(self.manifest["representation"]["svd"])
        )
        self._binary_bundle = joblib.load(
            self._resolve(self.manifest["binary_detector"])
        )
        if self._binary_bundle.get("llm_used_for_detection") is not False:
            raise ValueError("Linked binary detector must prohibit LLM detection")
        self._bundle = bundle

    def predict(self, text: str) -> dict[str, Any]:
        if not str(text).strip():
            raise ValueError("Classical multi-task input text must be non-empty")
        with self._lock:
            self.load()
            assert self._bundle is not None
            assert self._feature_union is not None
            assert self._svd is not None
            assert self._binary_bundle is not None
            sparse = self._feature_union.transform([text])
            dense = self._svd.transform(sparse).astype(np.float32)
            binary_score = float(frozen_detector_scores(self._binary_bundle, [text])[0])
            scam_type_scores = aligned_probabilities(
                self._bundle["scam_type_model"],
                dense,
                len(self.manifest["labels"]["scam_types"]),
                self._bundle["scam_type_class_ids"],
            )[0]
            stage_scores = aligned_probabilities(
                self._bundle["stage_model"],
                dense,
                len(self.manifest["labels"]["stages"]),
                self._bundle["stage_class_ids"],
            )[0]
            tactic_scores = {
                name: float(positive_probabilities(model, dense)[0])
                for name, model in self._bundle["tactic_models"].items()
            }
            return format_classical_multitask_outputs(
                self.manifest,
                binary_score,
                scam_type_scores,
                tactic_scores,
                stage_scores,
            )
