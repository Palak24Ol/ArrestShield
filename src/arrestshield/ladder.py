"""Shared feature, model, and latency utilities for the model ladder."""

from __future__ import annotations

from math import sqrt
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from xgboost import XGBClassifier

from .data import ConversationExample
from .protocol import stable_detection_turn, summarize_stable_detection


def build_feature_union(config: Mapping[str, Any]) -> FeatureUnion:
    features = config["features"]
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=tuple(features["word_ngram_range"]),
                    min_df=int(features["word_min_df"]),
                    max_df=float(features["max_df"]),
                    max_features=int(features["word_max_features"]),
                    sublinear_tf=bool(features["sublinear_tf"]),
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=tuple(features["char_ngram_range"]),
                    min_df=int(features["char_min_df"]),
                    max_df=float(features["max_df"]),
                    max_features=int(features["char_max_features"]),
                    sublinear_tf=bool(features["sublinear_tf"]),
                    dtype=np.float32,
                ),
            ),
        ]
    )


def build_svd(config: Mapping[str, Any]) -> TruncatedSVD:
    values = config["svd"]
    return TruncatedSVD(
        n_components=int(values["components"]),
        n_iter=int(values["iterations"]),
        random_state=int(values["random_seed"]),
    )


def build_sgd(config: Mapping[str, Any], seed: int) -> SGDClassifier:
    values = config["sgd"]
    return SGDClassifier(
        loss=str(values["loss"]),
        penalty=str(values["penalty"]),
        alpha=float(values["alpha"]),
        l1_ratio=float(values["l1_ratio"]),
        max_iter=int(values["max_iter"]),
        tol=float(values["tolerance"]),
        class_weight=values["class_weight"],
        average=bool(values["average"]),
        random_state=seed,
    )


def build_xgboost(
    config: Mapping[str, Any], seed: int, scale_pos_weight: float
) -> XGBClassifier:
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


def positive_scores(model: Any, matrix: Any) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model does not expose positive class 1")
    return np.asarray(model.predict_proba(matrix)[:, classes.index(1)], dtype=np.float64)


def build_prefix_batch(
    examples: Sequence[ConversationExample],
) -> tuple[list[ConversationExample], list[str], list[tuple[int, int]]]:
    positives = [example for example in examples if example.label == 1]
    prefix_texts: list[str] = []
    owners: list[tuple[int, int]] = []
    for example_index, example in enumerate(positives):
        for prefix_size in range(1, len(example.turn_texts) + 1):
            prefix_texts.append("\n".join(example.turn_texts[:prefix_size]))
            owners.append((example_index, prefix_size))
    return positives, prefix_texts, owners


def stable_latency_from_flat_scores(
    positives: Sequence[ConversationExample],
    owners: Sequence[tuple[int, int]],
    flat_scores: Sequence[float],
    entry_threshold: float,
    exit_threshold: float,
) -> dict[str, Any]:
    by_example: list[list[float]] = [[] for _ in positives]
    for (example_index, _), score in zip(owners, flat_scores):
        by_example[example_index].append(float(score))

    scammer_turns: list[int | None] = []
    prefix_turns: list[int | None] = []
    for example, scores in zip(positives, by_example):
        stable_prefix = stable_detection_turn(scores, entry_threshold, exit_threshold)
        prefix_turns.append(stable_prefix)
        if stable_prefix is None:
            scammer_turns.append(None)
            continue
        observed = example.turn_texts[:stable_prefix]
        caller_count = sum(
            turn.startswith("[ROLE=caller]")
            or turn.startswith("[ROLE=scammer]")
            or turn.startswith("[ROLE=fraudster]")
            for turn in observed
        )
        scammer_turns.append(caller_count or stable_prefix)

    return {
        "entry_threshold": float(entry_threshold),
        "exit_threshold": float(exit_threshold),
        "stable_scammer_turns": summarize_stable_detection(scammer_turns),
        "stable_all_prefix_turns": summarize_stable_detection(prefix_turns),
    }


def choose_family(validation_aggregates: Mapping[str, Mapping[str, float]]) -> str:
    """Apply the pre-registered validation-only family selection rule."""
    names = sorted(validation_aggregates)
    if not names:
        raise ValueError("No model aggregates supplied")
    winner = names[0]
    for candidate in names[1:]:
        current = validation_aggregates[winner]
        challenger = validation_aggregates[candidate]
        difference = challenger["recall_mean"] - current["recall_mean"]
        pooled = sqrt(
            (challenger["recall_std"] ** 2 + current["recall_std"] ** 2) / 2.0
        )
        if difference > pooled:
            winner = candidate
            continue
        if abs(difference) <= pooled:
            challenger_key = (
                -challenger["median_stable_turn_mean"],
                challenger["macro_f1_mean"],
                -challenger["artifact_bytes"],
            )
            current_key = (
                -current["median_stable_turn_mean"],
                current["macro_f1_mean"],
                -current["artifact_bytes"],
            )
            if challenger_key > current_key:
                winner = candidate
    return winner
