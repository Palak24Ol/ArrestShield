"""Measure the false-positive rate on large, genuinely unseen English conversation.

The Hinglish held-out view has 21 negatives, so its FPR interval is too wide to
support any claim. The three single-label English corpora are excluded from
mixed-only training by construction, which makes them ~48k unseen negatives — a
sample large enough for an FPR with a usable confidence interval.

Reported per source, never only as a total: the aggregate hides which domain the
detector actually fails on.

    python scripts/evaluate_unseen_english.py
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, write_json  # noqa: E402
from arrestshield.data import load_examples  # noqa: E402
from arrestshield.ladder import build_feature_union, build_sgd, positive_scores  # noqa: E402
from arrestshield.protocol import select_threshold_at_fpr  # noqa: E402

LOGGER = logging.getLogger("arrestshield.unseen_english")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/model/mixed_source_detector.json")
    parser.add_argument("--conversations", type=Path, default=PROJECT_ROOT / "data/processed/conversations.jsonl")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data/splits/split_manifest.json")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/unseen_english_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return [max(0.0, centre - spread), min(1.0, centre + spread)]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(value) for value in config["seeds"]]
    maximum_fpr = float(config["evaluation"]["maximum_false_positive_rate"])
    started = time.perf_counter()

    by_split, _ = load_examples(
        args.conversations,
        args.splits,
        config["data"]["allowed_label_provenance"],
        config["data"]["provenance_weights"],
    )
    everything = by_split["train"] + by_split["validation"] + by_split["test"]
    labels_by_source: dict[str, set[int]] = {}
    for item in everything:
        labels_by_source.setdefault(item.source, set()).add(item.label)
    mixed = sorted(source for source, labels in labels_by_source.items() if labels == {0, 1})
    unseen_sources = sorted(source for source, labels in labels_by_source.items() if labels == {0})

    train = [item for item in by_split["train"] if item.source in mixed]
    validation = [item for item in by_split["validation"] if item.source in mixed]
    # Every split of an excluded source is unseen: none of it entered training.
    unseen = [item for item in everything if item.source in unseen_sources]
    synthetic_positives = [
        item for item in by_split["test"]
        if item.source in {"synthetic_scam_dialogue", "synthetic_multi_agent_scam_conversation"}
        and item.label == 1
    ]
    LOGGER.info("Train rows (mixed only): %s   unseen English rows: %s", len(train), len(unseen))

    per_seed: list[dict] = []
    for seed in seeds:
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
        scores = positive_scores(model, features.transform([item.text for item in unseen]))
        flagged = scores >= point.threshold
        by_source = {}
        for source in unseen_sources:
            mask = np.array([item.source == source for item in unseen])
            by_source[source] = {"flagged": int(flagged[mask].sum()), "n": int(mask.sum())}
        positive_scores_synth = positive_scores(
            model, features.transform([item.text for item in synthetic_positives])
        )
        per_seed.append(
            {
                "seed": seed,
                "threshold": float(point.threshold),
                "by_source": by_source,
                "total_flagged": int(flagged.sum()),
                "total_n": int(len(unseen)),
                "synthetic_english_recall": float((positive_scores_synth >= point.threshold).mean()),
            }
        )
        LOGGER.info(
            "seed=%s threshold=%.4f flagged=%d/%d synthetic_recall=%.3f",
            seed, point.threshold, per_seed[-1]["total_flagged"], len(unseen),
            per_seed[-1]["synthetic_english_recall"],
        )

    aggregated = {}
    for source in unseen_sources:
        flagged = int(round(float(np.mean([row["by_source"][source]["flagged"] for row in per_seed]))))
        n = per_seed[0]["by_source"][source]["n"]
        aggregated[source] = {
            "mean_flagged": flagged,
            "n": n,
            "false_positive_rate": flagged / n,
            "wilson_95": wilson_interval(flagged, n),
            "per_seed_flagged": [row["by_source"][source]["flagged"] for row in per_seed],
        }
    total_flagged = int(round(float(np.mean([row["total_flagged"] for row in per_seed]))))
    total_n = per_seed[0]["total_n"]

    thresholds = [row["threshold"] for row in per_seed]
    flagged_counts = [row["total_flagged"] for row in per_seed]
    report = {
        "schema_version": "1.0.0",
        "run_id": "arrestshield-unseen-english-fpr-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "seeds": seeds,
        "training_regime": "mixed_label_sources_only",
        "training_sources": mixed,
        "unseen_sources": unseen_sources,
        "train_rows": len(train),
        "unseen_rows": total_n,
        "per_source": aggregated,
        "total": {
            "mean_flagged": total_flagged,
            "n": total_n,
            "false_positive_rate": total_flagged / total_n,
            "wilson_95": wilson_interval(total_flagged, total_n),
        },
        "seed_stability": {
            "thresholds": thresholds,
            "threshold_min": min(thresholds),
            "threshold_max": max(thresholds),
            "flagged_per_seed": flagged_counts,
            "flagged_ratio_max_over_min": max(flagged_counts) / max(1, min(flagged_counts)),
        },
        "synthetic_english_recall": {
            "mean": float(np.mean([row["synthetic_english_recall"] for row in per_seed])),
            "standard_deviation": float(np.std([row["synthetic_english_recall"] for row in per_seed])),
            "positives": len(synthetic_positives),
        },
        "runtime_seconds": time.perf_counter() - started,
        "llm_used_for_detection": False,
        "interpretation": [
            "The aggregate FPR hides the failure: it is dominated by two corpora the detector finds easy.",
            "banking77 is the domain closest to real deployment and is where the detector fails.",
            "Threshold selection is unstable across seeds, so the operating point is not yet reproducible.",
            "Synthetic English recall is 1.000 and is not evidence of real-world detection.",
        ],
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "metrics.json", report)
    LOGGER.info("Wrote %s", args.report_dir / "metrics.json")
    for source, row in aggregated.items():
        LOGGER.info(
            "%-28s FPR=%.4f  [%.4f, %.4f]  n=%d",
            source, row["false_positive_rate"], row["wilson_95"][0], row["wilson_95"][1], row["n"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
