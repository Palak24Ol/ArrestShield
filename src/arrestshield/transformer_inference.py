"""Reconstruct and query the exported multilingual multi-task detector."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np

from .multitask import build_torch_components, head_tail_token_ids


def format_multitask_outputs(
    manifest: Mapping[str, Any],
    binary_score: float,
    scam_type_scores: Sequence[float],
    tactic_scores: Sequence[float],
    stage_scores: Sequence[float],
) -> dict[str, Any]:
    labels = manifest["labels"]
    scam_types = list(labels["scam_types"])
    tactics = list(labels["tactics"])
    stages = list(labels["stages"])
    if len(scam_type_scores) != len(scam_types):
        raise ValueError("Scam-type output size does not match manifest")
    if len(tactic_scores) != len(tactics):
        raise ValueError("Tactic output size does not match manifest")
    if len(stage_scores) != len(stages):
        raise ValueError("Stage output size does not match manifest")
    type_index = int(np.argmax(scam_type_scores))
    stage_index = int(np.argmax(stage_scores))
    threshold = float(manifest["threshold"])
    return {
        "signal_source": "trained_multilingual_multitask_transformer",
        "selection_role": manifest.get("selection_role", "feasibility_only"),
        "binary": {
            "score": float(binary_score),
            "threshold": threshold,
            "is_scam": bool(float(binary_score) >= threshold),
        },
        "scam_type": {
            "label": scam_types[type_index],
            "score": float(scam_type_scores[type_index]),
            "scores": {
                name: float(score) for name, score in zip(scam_types, scam_type_scores)
            },
        },
        "tactics": {
            name: {"score": float(score), "present_at_0_5": bool(float(score) >= 0.5)}
            for name, score in zip(tactics, tactic_scores)
        },
        "stage": {
            "label": stages[stage_index],
            "score": float(stage_scores[stage_index]),
            "scores": {name: float(score) for name, score in zip(stages, stage_scores)},
        },
        "used_as_api_decision_source": False,
        "llm_used": False,
    }


class MultiTaskPredictor:
    """Lazy CPU inference wrapper for one exported multi-task artifact."""

    def __init__(self, artifact_dir: Path, torch_threads: int = 4) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.manifest_path = self.artifact_dir / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("llm_used_for_detection") is not False:
            raise ValueError("Transformer manifest must prohibit LLM detection")
        self.torch_threads = max(1, int(torch_threads))
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._lock = Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer

        torch.set_num_threads(self.torch_threads)
        backbone_path = self.artifact_dir / self.manifest["backbone_directory"]
        tokenizer_path = self.artifact_dir / self.manifest["tokenizer_directory"]
        heads_path = self.artifact_dir / self.manifest["heads_file"]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        _, ModelClass = build_torch_components()
        labels = self.manifest["labels"]
        model = ModelClass(
            str(backbone_path),
            len(labels["scam_types"]),
            len(labels["tactics"]),
            len(labels["stages"]),
            float(self.manifest["dropout"]),
            self.manifest["loss_weights"],
            False,
        )
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        model.binary_head.load_state_dict(heads["binary_head"])
        model.scam_type_head.load_state_dict(heads["scam_type_head"])
        model.tactic_head.load_state_dict(heads["tactic_head"])
        model.stage_head.load_state_dict(heads["stage_head"])
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch

    def predict(self, text: str) -> dict[str, Any]:
        if not str(text).strip():
            raise ValueError("Transformer input text must be non-empty")
        with self._lock:
            self.load()
            assert self._tokenizer is not None and self._model is not None and self._torch is not None
            input_ids, attention_mask = head_tail_token_ids(
                self._tokenizer,
                text,
                int(self.manifest["max_length"]),
                float(self.manifest["head_tail_ratio"]),
            )
            torch = self._torch
            with torch.inference_mode():
                output = self._model(
                    input_ids=torch.tensor([input_ids], dtype=torch.long),
                    attention_mask=torch.tensor([attention_mask], dtype=torch.long),
                )
                binary_score = float(torch.sigmoid(output["binary_logits"])[0].item())
                scam_type_scores = torch.softmax(output["scam_type_logits"], dim=-1)[0].tolist()
                tactic_scores = torch.sigmoid(output["tactic_logits"])[0].tolist()
                stage_scores = torch.softmax(output["stage_logits"], dim=-1)[0].tolist()
            return format_multitask_outputs(
                self.manifest,
                binary_score,
                scam_type_scores,
                tactic_scores,
                stage_scores,
            )
