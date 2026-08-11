"""Scalable multilingual TF-IDF baseline for binary scam detection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline


def build_baseline(config: Mapping[str, Any]) -> Pipeline:
    features = config["features"]
    classifier = config["classifier"]
    seed = int(config.get("random_seed", 42))

    union = FeatureUnion(
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
    model = SGDClassifier(
        loss=str(classifier["loss"]),
        penalty=str(classifier["penalty"]),
        alpha=float(classifier["alpha"]),
        l1_ratio=float(classifier["l1_ratio"]),
        max_iter=int(classifier["max_iter"]),
        tol=float(classifier["tolerance"]),
        class_weight=classifier["class_weight"],
        average=bool(classifier["average"]),
        random_state=seed,
    )
    return Pipeline([("features", union), ("classifier", model)])


def fit_baseline(
    model: Pipeline,
    training_texts: Sequence[str],
    training_labels: Sequence[int],
    training_weights: Sequence[float],
) -> Pipeline:
    model.fit(
        list(training_texts),
        np.asarray(training_labels, dtype=np.int8),
        classifier__sample_weight=np.asarray(training_weights, dtype=np.float32),
    )
    return model


def scam_scores(model: Pipeline, values: Sequence[str]) -> np.ndarray:
    probabilities = model.predict_proba(list(values))
    classes = list(model.named_steps["classifier"].classes_)
    try:
        positive_index = classes.index(1)
    except ValueError as error:
        raise ValueError("Classifier does not expose positive class 1") from error
    return np.asarray(probabilities[:, positive_index], dtype=np.float64)
