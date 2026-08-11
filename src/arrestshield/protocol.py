"""Pre-registered threshold selection and stable early-detection metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, stdev
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    false_positive_rate: float
    recall: float
    precision: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


def select_threshold_at_fpr(
    labels: Sequence[int], scores: Sequence[float], maximum_fpr: float
) -> OperatingPoint:
    """Maximize recall subject to a validation-only false-positive constraint."""
    if not 0 <= maximum_fpr < 1:
        raise ValueError("maximum_fpr must be in [0, 1)")
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    if set(y_true.tolist()) != {0, 1}:
        raise ValueError("Both classes are required to select an operating point")
    if len(y_true) != len(y_score):
        raise ValueError("labels and scores must have the same length")

    candidates = sorted(set(y_score.tolist()) | {0.0, 1.0}, reverse=True)
    eligible: list[OperatingPoint] = []
    for threshold in candidates:
        predicted = y_score >= threshold
        tp = int(np.sum(predicted & (y_true == 1)))
        fp = int(np.sum(predicted & (y_true == 0)))
        tn = int(np.sum((~predicted) & (y_true == 0)))
        fn = int(np.sum((~predicted) & (y_true == 1)))
        fpr = fp / (fp + tn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        point = OperatingPoint(
            threshold=float(threshold),
            false_positive_rate=float(fpr),
            recall=float(recall),
            precision=float(precision),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )
        if fpr <= maximum_fpr + 1e-12:
            eligible.append(point)
    if not eligible:
        raise RuntimeError("No threshold satisfies the false-positive constraint")
    return max(
        eligible,
        key=lambda point: (
            point.recall,
            point.precision,
            -point.false_positive_rate,
            -point.threshold,
        ),
    )


def stable_detection_turn(
    prefix_scores: Sequence[float], entry_threshold: float, exit_threshold: float
) -> int | None:
    """Return the 1-based first prefix that enters and never breaks hysteresis."""
    if not 0 <= exit_threshold <= entry_threshold <= 1:
        raise ValueError("Require 0 <= exit_threshold <= entry_threshold <= 1")
    values = [float(value) for value in prefix_scores]
    for index, score in enumerate(values):
        if score >= entry_threshold and all(
            remaining >= exit_threshold for remaining in values[index:]
        ):
            return index + 1
    return None


def summarize_stable_detection(turns: Iterable[int | None]) -> dict[str, float | int | None]:
    values = list(turns)
    detected = sorted(value for value in values if value is not None)
    if not values:
        return {"conversations": 0, "detected": 0, "undetected_rate": None, "median_turn": None}
    return {
        "conversations": len(values),
        "detected": len(detected),
        "undetected_rate": (len(values) - len(detected)) / len(values),
        "median_turn": float(median(detected)) if detected else None,
        "minimum_turn": int(detected[0]) if detected else None,
        "maximum_turn": int(detected[-1]) if detected else None,
    }


def summarize_seed_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("At least one seed result is required")
    return {
        "seeds": len(values),
        "mean": float(mean(values)),
        "standard_deviation": float(stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }
