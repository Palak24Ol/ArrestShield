"""Leakage-aware score calibration and pooled operating-point helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .data import ConversationExample


@dataclass
class ScoreCalibrator:
    """A small serializable wrapper around Platt or isotonic calibration."""

    method: str
    model: Any | None = None

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if self.method == "none":
            return np.clip(values, 0.0, 1.0)
        if self.method == "platt":
            clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            return np.asarray(self.model.predict_proba(logits)[:, 1], dtype=np.float64)
        if self.method == "isotonic":
            return np.asarray(self.model.predict(values), dtype=np.float64)
        raise ValueError(f"Unknown calibration method: {self.method}")


def fit_score_calibrator(
    method: str,
    scores: Sequence[float],
    labels: Sequence[int],
    sample_weights: Sequence[float] | None = None,
) -> ScoreCalibrator:
    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int8)
    weights = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    if len(values) != len(targets):
        raise ValueError("scores and labels must have equal length")
    if set(targets.tolist()) != {0, 1}:
        raise ValueError("Both classes are required for calibration")
    if method == "none":
        return ScoreCalibrator(method="none")
    if method == "platt":
        clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        model = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
        model.fit(logits, targets, sample_weight=weights)
        return ScoreCalibrator(method="platt", model=model)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(values, targets, sample_weight=weights)
        return ScoreCalibrator(method="isotonic", model=model)
    raise ValueError(f"Unsupported calibration method: {method}")


def split_calibration_and_threshold(
    examples: Sequence[ConversationExample], calibration_fraction: float = 0.5
) -> tuple[list[int], list[int]]:
    """Deterministically partition validation rows within each source/label stratum."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    groups: dict[tuple[str, int], list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault((example.source, example.label), []).append(index)

    calibration: list[int] = []
    threshold: list[int] = []
    for key, indices in sorted(groups.items()):
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"arrestshield-calibration-v1|{examples[index].conversation_id}".encode()
            ).hexdigest(),
        )
        if len(ranked) == 1:
            target = calibration if len(calibration) <= len(threshold) else threshold
            target.extend(ranked)
            continue
        cut = max(1, min(len(ranked) - 1, round(len(ranked) * calibration_fraction)))
        calibration.extend(ranked[:cut])
        threshold.extend(ranked[cut:])

    # Sparse source/label strata can leave one side single-class. Move the
    # minimum number of rows deterministically rather than silently reusing rows.
    def ensure_both(destination: list[int], source: list[int]) -> None:
        present = {examples[index].label for index in destination}
        for missing in sorted({0, 1} - present):
            candidate = next(
                (index for index in source if examples[index].label == missing), None
            )
            if candidate is None:
                raise ValueError("Validation data cannot support disjoint calibration and threshold views")
            source.remove(candidate)
            destination.append(candidate)

    ensure_both(calibration, threshold)
    ensure_both(threshold, calibration)
    return sorted(calibration), sorted(threshold)


def mean_seed_scores(score_rows: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(score_rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("score_rows must be a non-empty two-dimensional array")
    return matrix.mean(axis=0)


def select_shared_threshold_at_group_fpr(
    labels: Sequence[int],
    score_rows: Sequence[Sequence[float]],
    groups: Sequence[str],
    maximum_fpr: float,
    minimum_group_negatives: int = 1,
) -> dict[str, Any]:
    """Choose one threshold that satisfies every eligible group for every seed."""
    targets = np.asarray(labels, dtype=np.int8)
    matrix = np.asarray(score_rows, dtype=np.float64)
    names = np.asarray(groups, dtype=object)
    if matrix.ndim != 2 or matrix.shape[1] != len(targets) or len(names) != len(targets):
        raise ValueError("labels, groups, and every score row must have equal length")
    if set(targets.tolist()) != {0, 1}:
        raise ValueError("Both classes are required for threshold selection")
    if not 0.0 <= maximum_fpr < 1.0:
        raise ValueError("maximum_fpr must be in [0, 1)")

    eligible_groups = sorted(
        name
        for name in set(names.tolist())
        if int(np.sum((names == name) & (targets == 0))) >= minimum_group_negatives
    )
    if not eligible_groups:
        raise ValueError("No source group has enough negative examples for the FPR gate")

    required_threshold = 0.0
    constraints: list[dict[str, Any]] = []
    for seed_index, scores in enumerate(matrix):
        for name in eligible_groups:
            negative_scores = np.sort(scores[(names == name) & (targets == 0)])[::-1]
            allowed_false_positives = int(np.floor(maximum_fpr * len(negative_scores) + 1e-12))
            boundary = float(negative_scores[allowed_false_positives])
            safe_threshold = float(np.nextafter(boundary, np.inf))
            if safe_threshold > 1.0:
                safe_threshold = float("inf")
            required_threshold = max(required_threshold, safe_threshold)
            constraints.append(
                {
                    "seed_index": seed_index,
                    "source_group": name,
                    "negative_examples": int(len(negative_scores)),
                    "allowed_false_positives": allowed_false_positives,
                    "minimum_safe_threshold": safe_threshold,
                }
            )
    if not np.isfinite(required_threshold):
        raise RuntimeError("No threshold in [0, 1] satisfies every per-source FPR constraint")

    per_seed: list[dict[str, Any]] = []
    for seed_index, scores in enumerate(matrix):
        predicted = scores >= required_threshold
        positives = targets == 1
        recall = float(np.sum(predicted & positives) / np.sum(positives))
        group_fpr: dict[str, float] = {}
        for name in eligible_groups:
            negatives = (names == name) & (targets == 0)
            group_fpr[name] = float(np.sum(predicted & negatives) / np.sum(negatives))
        per_seed.append(
            {
                "seed_index": seed_index,
                "recall": recall,
                "per_source_fpr": group_fpr,
                "maximum_source_fpr": max(group_fpr.values()),
            }
        )
    return {
        "threshold": required_threshold,
        "maximum_fpr": maximum_fpr,
        "minimum_group_negatives": minimum_group_negatives,
        "eligible_groups": eligible_groups,
        "mean_recall": float(np.mean([row["recall"] for row in per_seed])),
        "minimum_seed_recall": float(min(row["recall"] for row in per_seed)),
        "per_seed": per_seed,
        "constraints": constraints,
        "all_seed_source_gates_passed": all(
            row["maximum_source_fpr"] <= maximum_fpr + 1e-12 for row in per_seed
        ),
    }
