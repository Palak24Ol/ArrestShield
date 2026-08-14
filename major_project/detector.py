"""Small binary transcript classifier used by the major-project demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    label: str
    is_scam: bool
    scam_score: float
    threshold: float
    matched_patterns: tuple[str, ...]
    model_family: str
    llm_used_for_detection: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "is_scam": self.is_scam,
            "scam_score": self.scam_score,
            "threshold": self.threshold,
            "matched_patterns": list(self.matched_patterns),
            "model_family": self.model_family,
            "llm_used_for_detection": False,
        }


class SimpleScamDetector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve(strict=True)
        self.bundle = joblib.load(self.model_path)
        if self.bundle.get("llm_used_for_detection") is not False:
            raise ValueError("Refusing a detector artifact without an explicit no-LLM boundary")

    @staticmethod
    def format_text(text: str) -> str:
        clean = " ".join(str(text or "").split())
        if not clean:
            raise ValueError("Transcript is empty")
        return clean if clean.startswith("[ROLE=") else f"[ROLE=caller] {clean}"

    def _patterns(self, matrix, limit: int = 8) -> tuple[str, ...]:
        model = self.bundle["model"]
        representation = self.bundle["feature_union"]
        if not hasattr(model, "coef_") or not hasattr(representation, "get_feature_names_out"):
            return ()
        contributions = matrix.multiply(np.asarray(model.coef_[0])).tocsr()
        if not contributions.nnz:
            return ()
        names = representation.get_feature_names_out()
        row = contributions.getrow(0)
        ranked = sorted(
            zip(row.indices.tolist(), row.data.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        patterns: list[str] = []
        ignored = {
            "role", "caller", "role caller", "aur", "hai", "ko", "mat", "se",
            "the", "a", "an", "to", "is", "are", "will",
        }
        # Word n-grams are understandable to a user. Character n-grams remain
        # valuable to the classifier but are not useful as an explanation.
        for index, contribution in ranked:
            if contribution <= 0:
                continue
            name = str(names[index])
            if not name.startswith("word__"):
                continue
            value = name.removeprefix("word__").strip()
            if value and value not in ignored and value not in patterns:
                patterns.append(value)
            if len(patterns) >= limit:
                break
        return tuple(patterns)

    def predict(self, transcript: str) -> DetectionResult:
        formatted = self.format_text(transcript)
        matrix = self.bundle["feature_union"].transform([formatted])
        if self.bundle.get("svd") is not None:
            matrix = self.bundle["svd"].transform(matrix)
        score = float(self.bundle["model"].predict_proba(matrix)[0, 1])
        calibrator = self.bundle.get("calibrator")
        if calibrator is not None:
            score = float(calibrator.predict([score])[0])
        threshold = float(self.bundle["threshold"])
        is_scam = score >= threshold
        return DetectionResult(
            label="SCAM" if is_scam else "NOT_SCAM",
            is_scam=is_scam,
            scam_score=score,
            threshold=threshold,
            matched_patterns=self._patterns(matrix),
            model_family=str(self.bundle.get("model_family") or "unknown"),
        )
