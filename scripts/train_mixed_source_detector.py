"""Train the detector on mixed-label sources only and audit it honestly.

Sources that contain a single label let the model use source identity as a proxy
for the label. Restricting training to sources holding BOTH labels removes that
shortcut by construction.

The headline metric is held-out-source ROC-AUC, which is threshold-free and so
reports the representation rather than threshold transfer. Recall and FPR are
reported alongside it with Wilson intervals, and any source whose held-out
negative count is below the configured floor is flagged as unmeasurable rather
than quietly averaged in.

    python scripts/train_mixed_source_detector.py
    python scripts/train_mixed_source_detector.py --augmentation data/external/llm_augmentation/conversations.jsonl
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, write_json  # noqa: E402
from arrestshield.data import ConversationExample, format_conversation, iter_jsonl, load_examples  # noqa: E402
from arrestshield.ladder import build_feature_union, build_sgd, positive_scores  # noqa: E402
from arrestshield.protocol import select_threshold_at_fpr  # noqa: E402

LOGGER = logging.getLogger("arrestshield.mixed_source")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/model/mixed_source_detector.json")
    parser.add_argument("--conversations", type=Path, default=PROJECT_ROOT / "data/processed/conversations.jsonl")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data/splits/split_manifest.json")
    parser.add_argument("--augmentation", type=Path, default=None, help="Optional llm_synthetic JSONL; train split only")
    parser.add_argument("--extra-negatives", type=Path, default=None, help="Optional declared external JSONL; train split only")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/mixed_source_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% interval for a rate. Wide intervals are the point, not a defect."""
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, centre - spread), min(1.0, centre + spread)


def load_train_only(
    path: Path, weights: Mapping[str, float], allowed_provenance: set[str]
) -> list[ConversationExample]:
    """Load declared train-only rows.

    These are training-only by construction: the loader assigns split='train' and
    never consults a split manifest, so no row from this file can reach the
    validation or test views regardless of what the file claims.
    """
    rows: list[ConversationExample] = []
    for record in iter_jsonl(path):
        label_data = record.get("conversation_label") or {}
        provenance = str(label_data.get("provenance") or "")
        if provenance not in allowed_provenance:
            raise ValueError(
                f"{path.name} contains provenance {provenance!r}; "
                f"expected one of {sorted(allowed_provenance)}"
            )
        policy = record.get("split_policy") or {}
        if policy and policy.get("allowed_splits") not in (None, ["train"]):
            raise ValueError(f"{path.name} declares non-train splits: {policy.get('allowed_splits')}")
        label = label_data.get("is_scam")
        if label not in (0, 1):
            continue
        text, turn_texts = format_conversation(record.get("turns") or [])
        if not text:
            continue
        source_data = record.get("source") or {}
        rows.append(
            ConversationExample(
                conversation_id=str(record.get("conversation_id")),
                text=text,
                label=int(label),
                scam_type=str(label_data.get("scam_type") or "unknown"),
                split="train",
                source=str(source_data.get("dataset_id") or "llm_augmentation"),
                languages=tuple(sorted({str(item) for item in record.get("language_profile") or ["unknown"]})),
                provenance=provenance,
                turn_texts=turn_texts,
                sample_weight=float(weights.get(provenance, 0.5)),
            )
        )
    return rows


def describe(rows: Sequence[ConversationExample]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "scam": sum(item.label for item in rows),
        "legitimate": sum(1 for item in rows if item.label == 0),
        "sources": sorted({item.source for item in rows}),
    }


def fit_once(
    config: Mapping[str, Any],
    train: Sequence[ConversationExample],
    validation: Sequence[ConversationExample],
    evaluation: Sequence[ConversationExample],
    seed: int,
    maximum_fpr: float,
) -> dict[str, Any]:
    features = build_feature_union(config)
    matrix = features.fit_transform([item.text for item in train])
    model = build_sgd(config, seed)
    model.fit(
        matrix,
        [item.label for item in train],
        sample_weight=[item.sample_weight for item in train],
    )

    validation_scores = positive_scores(model, features.transform([item.text for item in validation]))
    point = select_threshold_at_fpr(
        [item.label for item in validation], validation_scores, maximum_fpr=maximum_fpr
    )

    scores = positive_scores(model, features.transform([item.text for item in evaluation]))
    y = np.asarray([item.label for item in evaluation])
    predicted = (scores >= point.threshold).astype(int)
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    true_positives = int(predicted[y == 1].sum()) if positives else 0
    false_positives = int(predicted[y == 0].sum()) if negatives else 0
    return {
        "seed": seed,
        "threshold": point.threshold,
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else None,
        "recall": true_positives / positives if positives else None,
        "false_positive_rate": false_positives / negatives if negatives else None,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "positives": positives,
        "negatives": negatives,
    }


def aggregate(runs: Sequence[Mapping[str, Any]], minimum_negatives: int) -> dict[str, Any]:
    def mean_of(name: str) -> float | None:
        values = [run[name] for run in runs if run.get(name) is not None]
        return float(np.mean(values)) if values else None

    def std_of(name: str) -> float | None:
        values = [run[name] for run in runs if run.get(name) is not None]
        return float(np.std(values)) if len(values) > 1 else 0.0

    negatives = runs[0]["negatives"]
    false_positives = int(round(float(np.mean([run["false_positives"] for run in runs]))))
    interval = wilson_interval(false_positives, negatives)
    return {
        "seeds": [run["seed"] for run in runs],
        "positives": runs[0]["positives"],
        "negatives": negatives,
        "roc_auc_mean": mean_of("roc_auc"),
        "roc_auc_std": std_of("roc_auc"),
        "recall_mean": mean_of("recall"),
        "false_positive_rate_mean": mean_of("false_positive_rate"),
        "false_positive_rate_wilson_95": list(interval) if interval else None,
        "negative_count_sufficient": negatives >= minimum_negatives,
        "measurement_note": (
            None
            if negatives >= minimum_negatives
            else f"Only {negatives} held-out negatives; the FPR interval is too wide to support any claim."
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluation_config = config["evaluation"]
    maximum_fpr = float(evaluation_config["maximum_false_positive_rate"])
    minimum_negatives = int(evaluation_config["minimum_reliable_negative_count"])
    seeds = args.seeds or [int(value) for value in config["seeds"]]
    started = time.perf_counter()

    by_split, summary = load_examples(
        args.conversations,
        args.splits,
        config["data"]["allowed_label_provenance"],
        config["data"]["provenance_weights"],
    )
    everything = by_split["train"] + by_split["validation"] + by_split["test"]
    labels_by_source: dict[str, set[int]] = {}
    counts: dict[str, dict[int, int]] = {}
    for item in everything:
        labels_by_source.setdefault(item.source, set()).add(item.label)
        counts.setdefault(item.source, {0: 0, 1: 0})[item.label] += 1
    floor = int(config["training_regime"]["minimum_examples_per_label_per_source"])
    mixed = sorted(
        source
        for source, labels in labels_by_source.items()
        if labels == {0, 1} and min(counts[source].values()) >= floor
    )
    excluded = sorted(set(labels_by_source) - set(mixed))
    LOGGER.info("Mixed-label sources retained: %s", mixed)
    LOGGER.info("Single-label sources excluded: %s", excluded)

    weights = config["data"]["provenance_weights"]
    augmentation: list[ConversationExample] = []
    if args.augmentation and args.augmentation.exists():
        augmentation = load_train_only(args.augmentation, weights, {"llm_synthetic"})
        LOGGER.info("Loaded llm_synthetic augmentation: %s", describe(augmentation))
    if args.extra_negatives and args.extra_negatives.exists():
        extra = load_train_only(args.extra_negatives, weights, {"external_unverified_license"})
        LOGGER.info("Loaded external train-only negatives: %s", describe(extra))
        augmentation = augmentation + extra

    def pool(split: str) -> list[ConversationExample]:
        return [item for item in by_split[split] if item.source in mixed]

    results: dict[str, Any] = {}

    # 1. Ordinary mixed-source split (supporting evidence only).
    train = pool("train") + augmentation
    ordinary = [
        fit_once(config, train, pool("validation"), pool("test"), seed, maximum_fpr)
        for seed in seeds
    ]
    results["ordinary_mixed_source_test"] = aggregate(ordinary, minimum_negatives)
    LOGGER.info(
        "Ordinary mixed-source test AUC=%.3f",
        results["ordinary_mixed_source_test"]["roc_auc_mean"],
    )

    # 2. Leave-one-source-out (the honest headline).
    per_source: dict[str, Any] = {}
    for held_source in mixed:
        held = [item for item in by_split["test"] if item.source == held_source]
        if len({item.label for item in held}) < 2:
            per_source[held_source] = {"skipped": "held-out test set is single-class"}
            continue
        fold_train = [item for item in pool("train") if item.source != held_source] + augmentation
        fold_validation = [item for item in pool("validation") if item.source != held_source]
        if len({item.label for item in fold_train}) < 2 or len({item.label for item in fold_validation}) < 2:
            per_source[held_source] = {"skipped": "single-class train or validation after exclusion"}
            continue
        runs = [fit_once(config, fold_train, fold_validation, held, seed, maximum_fpr) for seed in seeds]
        per_source[held_source] = aggregate(runs, minimum_negatives)
        per_source[held_source]["train_rows"] = len(fold_train)
        LOGGER.info(
            "LOSO %-45s AUC=%.3f n_neg=%d",
            held_source,
            per_source[held_source]["roc_auc_mean"],
            per_source[held_source]["negatives"],
        )
    results["leave_one_source_out"] = per_source

    measurable = [
        row for row in per_source.values()
        if isinstance(row, dict) and row.get("negative_count_sufficient") and row.get("roc_auc_mean")
    ]
    results["headline"] = {
        "primary_metric": evaluation_config["primary_metric"],
        "source_macro_roc_auc_all": float(
            np.mean([
                row["roc_auc_mean"] for row in per_source.values()
                if isinstance(row, dict) and row.get("roc_auc_mean") is not None
            ])
        ),
        "sources_with_sufficient_negatives": len(measurable),
        "sources_evaluated": sum(1 for row in per_source.values() if isinstance(row, dict) and "skipped" not in row),
        "fpr_is_measurable": bool(measurable),
    }

    report = {
        "schema_version": "1.0.0",
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "seeds": seeds,
        "training_regime": "mixed_label_sources_only",
        "mixed_sources": mixed,
        "excluded_single_label_sources": excluded,
        "augmentation_rows": len(augmentation),
        "augmentation_is_train_split_only": True,
        "corpus_summary": {
            "retained": summary.retained,
            "split_counts": dict(summary.split_counts),
            "label_counts": dict(summary.label_counts),
        },
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
        "llm_used_for_detection": False,
        "limitations": [
            "All positive labels remain source-silver; no human-gold positives exist yet.",
            "Held-out negative counts are small for the Indic sources, so FPR intervals are wide and no FPR claim is supported.",
            "Any llm_synthetic augmentation enters the training split only and never validation or test.",
            "No LLM produced a feature, score, threshold, or label.",
        ],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "metrics.json", report)
    write_json(args.report_dir / "config.json", config)
    LOGGER.info("Wrote %s", args.report_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
