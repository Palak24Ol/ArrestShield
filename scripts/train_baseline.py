"""Train, tune, evaluate, and package the ArrestShield binary baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import save_training_run  # noqa: E402
from arrestshield.baseline import build_baseline, fit_baseline, scam_scores  # noqa: E402
from arrestshield.data import labels, load_examples, sample_weights, texts  # noqa: E402
from arrestshield.evaluation import (  # noqa: E402
    binary_metrics,
    choose_threshold,
    early_detection_metrics,
    grouped_metrics,
)


LOGGER = logging.getLogger("arrestshield.train_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/model/baseline.json")
    parser.add_argument(
        "--conversations",
        type=Path,
        default=PROJECT_ROOT / "data/processed/conversations.jsonl",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PROJECT_ROOT / "data/splits/split_manifest.json",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/models/baseline_v1",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports/baseline_v1"
    )
    parser.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Deterministic per-split cap for smoke tests; omit for the full training run.",
    )
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0.0":
        raise ValueError(f"Unsupported config schema in {path}")
    return payload


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = read_config(args.config)
    data_config = config["data"]

    LOGGER.info("Loading canonical examples with the fixed conversation-group splits")
    by_split, load_summary = load_examples(
        conversations_path=args.conversations,
        split_manifest_path=args.splits,
        allowed_provenance=data_config["allowed_label_provenance"],
        provenance_weights=data_config["provenance_weights"],
        max_per_split=args.max_per_split,
    )
    LOGGER.info("Split counts: %s", load_summary.split_counts)

    train_examples = by_split["train"]
    validation_examples = by_split["validation"]
    test_examples = by_split["test"]

    LOGGER.info("Fitting multilingual word+character baseline")
    model = build_baseline(config)
    fit_baseline(
        model,
        texts(train_examples),
        labels(train_examples),
        sample_weights(train_examples),
    )

    LOGGER.info("Selecting the decision threshold using validation data only")
    validation_scores = scam_scores(model, texts(validation_examples))
    threshold_config = config["threshold_selection"]
    threshold_selection = choose_threshold(
        labels(validation_examples),
        validation_scores,
        beta=float(threshold_config["beta"]),
        min_precision=float(threshold_config["min_precision"]),
        grid_size=int(threshold_config["grid_size"]),
    )
    threshold = float(threshold_selection["threshold"])

    LOGGER.info("Evaluating the untouched test split at threshold %.4f", threshold)
    test_scores = scam_scores(model, texts(test_examples))
    metrics = {
        "schema_version": "1.0.0",
        "model_id": config["model_id"],
        "evaluation_protocol": (
            "threshold selected on validation; all reported final metrics use the untouched "
            "conversation-group test split"
        ),
        "threshold_selection": threshold_selection,
        "validation": binary_metrics(labels(validation_examples), validation_scores, threshold),
        "test": binary_metrics(labels(test_examples), test_scores, threshold),
        "test_slices": grouped_metrics(test_examples, test_scores, threshold),
        "early_detection": early_detection_metrics(
            test_examples,
            score_function=lambda values: scam_scores(model, values),
            threshold=threshold,
            fractions=config["evaluation"]["early_detection_fractions"],
        ),
        "limitations": [
            "Positive seed labels are predominantly silver or synthetic and are not a substitute for human-reviewed Indian call transcripts.",
            "The baseline estimates binary scam risk only; tactic and stage heads require gold annotations.",
            "The score is not certified as a calibrated probability and must be revalidated before real call routing.",
            "An LLM is not used anywhere in detection or evaluation.",
        ],
    }
    data_summary = asdict(load_summary)
    metadata = save_training_run(
        model=model,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        config=config,
        data_summary=data_summary,
        threshold=threshold_selection,
        metrics=metrics,
        project_root=PROJECT_ROOT,
    )
    LOGGER.info("Saved model %s (%s)", metadata["model_id"], metadata["model_sha256"])
    LOGGER.info("Test F2 %.4f; recall %.4f; precision %.4f", metrics["test"]["f2"], metrics["test"]["recall"], metrics["test"]["precision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
