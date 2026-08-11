"""Validation-only threshold tuning and leakage-safe evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import ConversationExample


def choose_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    beta: float = 2.0,
    min_precision: float = 0.0,
    grid_size: int = 197,
) -> dict[str, float]:
    """Choose one operating threshold using validation labels only."""
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    candidates = np.linspace(0.01, 0.99, num=grid_size)
    rows: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predicted = (y_score >= threshold).astype(np.int8)
        precision = precision_score(y_true, predicted, zero_division=0)
        recall = recall_score(y_true, predicted, zero_division=0)
        fbeta = fbeta_score(y_true, predicted, beta=beta, zero_division=0)
        rows.append((float(threshold), float(fbeta), float(precision), float(recall)))

    eligible = [row for row in rows if row[2] >= min_precision]
    pool = eligible or rows
    threshold, score, precision, recall = max(
        pool,
        key=lambda row: (row[1], row[3], row[2], -row[0]),
    )
    return {
        "threshold": threshold,
        "validation_fbeta": score,
        "validation_precision": precision,
        "validation_recall": recall,
        "beta": float(beta),
        "requested_min_precision": float(min_precision),
        "min_precision_satisfied": bool(eligible),
    }


def binary_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    has_both_classes = len(set(y_true.tolist())) == 2
    result: dict[str, Any] = {
        "examples": int(len(y_true)),
        "positive_examples": int(y_true.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred)) if has_both_classes else None
        ),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "matthews_correlation": (
            float(matthews_corrcoef(y_true, y_pred)) if has_both_classes else None
        ),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if has_both_classes:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        result["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def grouped_metrics(
    examples: Sequence[ConversationExample],
    scores: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    source_rows: dict[str, list[int]] = defaultdict(list)
    language_rows: dict[str, list[int]] = defaultdict(list)
    scam_type_rows: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        source_rows[example.source].append(index)
        for language in example.languages:
            language_rows[language].append(index)
        if example.label == 1:
            scam_type_rows[example.scam_type].append(index)

    score_array = np.asarray(scores, dtype=np.float64)

    def evaluate_groups(groups: Mapping[str, Sequence[int]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for name, indices in sorted(groups.items()):
            subset_labels = [examples[index].label for index in indices]
            subset_scores = score_array[list(indices)]
            output[name] = binary_metrics(subset_labels, subset_scores, threshold)
        return output

    scam_type_recall: dict[str, Any] = {}
    for name, indices in sorted(scam_type_rows.items()):
        subset_scores = score_array[list(indices)]
        detected = int((subset_scores >= threshold).sum())
        scam_type_recall[name] = {
            "positive_examples": len(indices),
            "detected": detected,
            "recall": detected / len(indices),
            "mean_score": float(subset_scores.mean()),
        }

    return {
        "by_source": evaluate_groups(source_rows),
        "by_language": evaluate_groups(language_rows),
        "positive_recall_by_scam_type": scam_type_recall,
    }


def early_detection_metrics(
    examples: Sequence[ConversationExample],
    score_function: Callable[[Sequence[str]], np.ndarray],
    threshold: float,
    fractions: Sequence[float],
) -> dict[str, Any]:
    """Measure whether a positive conversation fires using only each prefix."""
    positives = [example for example in examples if example.label == 1]
    if not positives:
        return {"positive_conversations": 0, "detected": 0, "undetected": 0}

    prefix_texts: list[str] = []
    owners: list[tuple[int, int, int]] = []
    for example_index, example in enumerate(positives):
        for prefix_size in range(1, len(example.turn_texts) + 1):
            prefix_texts.append("\n".join(example.turn_texts[:prefix_size]))
            owners.append((example_index, prefix_size, len(example.turn_texts)))

    prefix_scores = score_function(prefix_texts)
    first_detection: dict[int, tuple[int, int]] = {}
    score_lookup: dict[tuple[int, int], float] = {}
    for owner, score in zip(owners, prefix_scores):
        example_index, prefix_size, total_turns = owner
        score_lookup[(example_index, prefix_size)] = float(score)
        if score >= threshold and example_index not in first_detection:
            first_detection[example_index] = (prefix_size, total_turns)

    rates: dict[str, float] = {}
    for fraction in fractions:
        detected = 0
        for example_index, example in enumerate(positives):
            prefix_size = min(len(example.turn_texts), max(1, ceil(len(example.turn_texts) * fraction)))
            if score_lookup[(example_index, prefix_size)] >= threshold:
                detected += 1
        rates[f"at_{int(round(fraction * 100))}_percent"] = detected / len(positives)

    detection_fractions = [turn / total for turn, total in first_detection.values()]
    return {
        "positive_conversations": len(positives),
        "detected": len(first_detection),
        "undetected": len(positives) - len(first_detection),
        "eventual_detection_rate": len(first_detection) / len(positives),
        "median_fraction_to_first_detection": (
            float(median(detection_fractions)) if detection_fractions else None
        ),
        "detection_rate_by_available_conversation": rates,
    }
