"""Multilingual multi-task examples, model heads, and masked weak-label losses."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import format_turn, iter_jsonl, load_split_map


LEGACY_TACTIC_MAP = {
    "fear_threat": "fear_intimidation",
    "urgency": "urgency_scarcity",
    "secrecy": "secrecy_instruction",
    "isolation": "isolation_instruction",
    "credential_otp_request": "credential_request",
}

STAGE_ORDER = {
    "none_unknown": 0,
    "contact": 1,
    "authority_claim": 2,
    "accusation": 3,
    "threat": 4,
    "isolation_control": 5,
    "payment_extraction": 6,
}


@dataclass(frozen=True)
class MultiTaskExample:
    conversation_id: str
    text: str
    turn_texts: tuple[str, ...]
    split: str
    source: str
    languages: tuple[str, ...]
    provenance: str
    sample_weight: float
    binary_label: int
    scam_type_label: int
    tactic_labels: tuple[float, ...]
    tactic_mask: tuple[float, ...]
    stage_label: int
    stage_mask: float
    turn_tactics: tuple[tuple[float, ...], ...] = ()
    turn_stages: tuple[str, ...] = ()
    tactics_annotated: bool = False
    stages_annotated: bool = False
    window_id: str = "full"


def stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_tactic(name: str) -> str:
    return LEGACY_TACTIC_MAP.get(name, name)


def extract_turn_records(
    turns: Sequence[Mapping[str, Any]],
    tactic_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...], tuple[str, ...]]:
    """Extract text and labels in one pass so prefix indices stay aligned."""
    tactic_index = {name: index for index, name in enumerate(tactic_names)}
    texts: list[str] = []
    tactic_rows: list[tuple[float, ...]] = []
    stage_rows: list[str] = []
    for turn in turns:
        body = str(turn.get("normalized_text") or turn.get("raw_text") or "")
        if not body.strip():
            continue
        texts.append(format_turn(str(turn.get("speaker_role", "unknown")), body))
        vector = np.zeros(len(tactic_names), dtype=np.float32)
        turn_labels = turn.get("labels") or {}
        for raw_name, value in (turn_labels.get("tactics") or {}).items():
            name = canonical_tactic(str(raw_name))
            if value in (1, True) and name in tactic_index:
                vector[tactic_index[name]] = 1.0
        stage_name = str(turn_labels.get("stage") or "none_unknown")
        if stage_name not in STAGE_ORDER:
            stage_name = "none_unknown"
        tactic_rows.append(tuple(float(value) for value in vector))
        stage_rows.append(stage_name)
    return tuple(texts), tuple(tactic_rows), tuple(stage_rows)


def aggregate_prefix_labels(
    turn_tactics: Sequence[Sequence[float]],
    turn_stages: Sequence[str],
    is_scam: int,
    stage_names: Sequence[str],
    tactic_count: int,
    tactics_annotated: bool,
    stages_annotated: bool,
    unknown_tactic_negative_weight: float,
) -> tuple[tuple[float, ...], tuple[float, ...], int, float]:
    """Aggregate labels from visible turns only, preserving silver uncertainty."""
    labels = np.zeros(tactic_count, dtype=np.float32)
    for vector in turn_tactics:
        labels = np.maximum(labels, np.asarray(vector, dtype=np.float32))
    observed_stages = [stage for stage in turn_stages if stage != "none_unknown"]
    stage_index = {name: index for index, name in enumerate(stage_names)}
    if is_scam == 0:
        tactic_mask = np.ones(tactic_count, dtype=np.float32)
        stage_name = "none_unknown"
        stage_mask = 1.0
    else:
        if tactics_annotated:
            tactic_mask = np.where(
                labels > 0.0, 1.0, float(unknown_tactic_negative_weight)
            ).astype(np.float32)
        else:
            tactic_mask = np.zeros(tactic_count, dtype=np.float32)
        stage_name = (
            max(observed_stages, key=STAGE_ORDER.get)
            if observed_stages
            else "none_unknown"
        )
        # Once the source is known to carry stage annotation, an early
        # none_unknown prefix is useful negative escalation evidence.
        stage_mask = 1.0 if stages_annotated else 0.0
    return (
        tuple(float(value) for value in labels),
        tuple(float(value) for value in tactic_mask),
        stage_index.get(stage_name, stage_index["none_unknown"]),
        float(stage_mask),
    )


def aggregate_turn_labels(
    turns: Sequence[Mapping[str, Any]],
    is_scam: int,
    tactic_names: Sequence[str],
    stage_names: Sequence[str],
    unknown_tactic_negative_weight: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...], int, float]:
    """Aggregate turn labels without converting unknown tactic zeros to negatives.

    The current positive corpus is silver-labeled. A positive tactic is observed,
    but an unmarked tactic is unknown rather than a reliable negative. Confirmed
    non-scam conversations, however, provide valid all-zero tactic supervision.
    """
    _, turn_tactics, turn_stages = extract_turn_records(turns, tactic_names)
    return aggregate_prefix_labels(
        turn_tactics,
        turn_stages,
        is_scam,
        stage_names,
        len(tactic_names),
        any(any(vector) for vector in turn_tactics),
        any(stage != "none_unknown" for stage in turn_stages),
        unknown_tactic_negative_weight,
    )


def load_multitask_examples(
    conversations_path: Path,
    split_manifest_path: Path,
    config: Mapping[str, Any],
) -> dict[str, list[MultiTaskExample]]:
    split_map = load_split_map(split_manifest_path)
    label_config = config["labels"]
    data_config = config["data"]
    scam_types = list(label_config["scam_types"])
    tactic_names = list(label_config["tactics"])
    stage_names = list(label_config["stages"])
    expected_stage_order = [name for name, _ in sorted(STAGE_ORDER.items(), key=lambda row: row[1])]
    if stage_names != expected_stage_order:
        raise ValueError(
            "Stage labels must follow causal stage order: " + ", ".join(expected_stage_order)
        )
    scam_type_index = {name: index for index, name in enumerate(scam_types)}
    allowed = set(data_config["allowed_label_provenance"])
    provenance_weights = data_config["provenance_weights"]
    unknown_weight = float(data_config.get("unknown_tactic_negative_weight", 0.25))
    output: dict[str, list[MultiTaskExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for record in iter_jsonl(conversations_path):
        conversation_id = str(record.get("conversation_id") or "")
        split = split_map.get(conversation_id)
        quality = record.get("quality") or {}
        conversation_label = record.get("conversation_label") or {}
        provenance = str(conversation_label.get("provenance") or "unlabeled")
        raw_binary = conversation_label.get("is_scam")
        if (
            split not in output
            or not quality.get("training_eligible", False)
            or provenance not in allowed
            or raw_binary not in (0, 1)
        ):
            continue
        turn_texts, turn_tactics, turn_stages = extract_turn_records(
            record.get("turns") or [], tactic_names
        )
        text = "\n".join(turn_texts)
        if not text:
            continue
        binary_label = int(raw_binary)
        raw_scam_type = str(conversation_label.get("scam_type") or "other_scam")
        if binary_label == 0:
            raw_scam_type = "non_scam"
        elif raw_scam_type not in scam_type_index:
            raw_scam_type = "other_scam"
        tactics_annotated = any(any(vector) for vector in turn_tactics)
        stages_annotated = any(stage != "none_unknown" for stage in turn_stages)
        tactic_labels, tactic_mask, stage_label, stage_mask = aggregate_prefix_labels(
            turn_tactics,
            turn_stages,
            binary_label,
            stage_names,
            len(tactic_names),
            tactics_annotated,
            stages_annotated,
            unknown_weight,
        )
        source = str((record.get("source") or {}).get("dataset_id") or "unknown")
        languages = tuple(sorted(str(item) for item in record.get("language_profile") or ["unknown"]))
        output[split].append(
            MultiTaskExample(
                conversation_id=conversation_id,
                text=text,
                turn_texts=turn_texts,
                split=split,
                source=source,
                languages=languages,
                provenance=provenance,
                sample_weight=float(provenance_weights.get(provenance, 1.0)),
                binary_label=binary_label,
                scam_type_label=scam_type_index[raw_scam_type],
                tactic_labels=tactic_labels,
                tactic_mask=tactic_mask,
                stage_label=stage_label,
                stage_mask=stage_mask,
                turn_tactics=turn_tactics,
                turn_stages=turn_stages,
                tactics_annotated=tactics_annotated,
                stages_annotated=stages_annotated,
            )
        )

    for split, examples in output.items():
        if not examples:
            raise ValueError(f"No eligible multi-task examples for {split}")
    if {item.binary_label for item in output["train"]} != {0, 1}:
        raise ValueError("Multi-task training split must contain both binary classes")
    return output


def _take_stable(examples: Iterable[MultiTaskExample], count: int) -> list[MultiTaskExample]:
    return sorted(examples, key=lambda item: stable_rank(item.conversation_id))[: max(0, count)]


def source_balanced_training_sample(
    examples: Sequence[MultiTaskExample],
    maximum_negative_to_positive_ratio: float,
    mixed_source_negative_ratio: float,
) -> list[MultiTaskExample]:
    """Keep all positives and cap negatives without making source equal to label."""
    positives = [item for item in examples if item.binary_label == 1]
    negatives = [item for item in examples if item.binary_label == 0]
    if not positives or not negatives:
        raise ValueError("Balanced sampling requires positive and negative examples")
    positives_by_source: dict[str, list[MultiTaskExample]] = defaultdict(list)
    negatives_by_source: dict[str, list[MultiTaskExample]] = defaultdict(list)
    for item in positives:
        positives_by_source[item.source].append(item)
    for item in negatives:
        negatives_by_source[item.source].append(item)

    maximum_negatives = int(math.ceil(len(positives) * maximum_negative_to_positive_ratio))
    selected: list[MultiTaskExample] = list(positives)
    selected_ids: set[str] = set()
    for source, source_positives in sorted(positives_by_source.items()):
        requested = int(math.ceil(len(source_positives) * mixed_source_negative_ratio))
        picked = _take_stable(negatives_by_source.get(source, []), requested)
        selected.extend(picked)
        selected_ids.update(item.conversation_id for item in picked)

    remaining_budget = max(0, maximum_negatives - len(selected_ids))
    pools = {
        source: [item for item in sorted(items, key=lambda row: stable_rank(row.conversation_id))
                 if item.conversation_id not in selected_ids]
        for source, items in negatives_by_source.items()
    }
    sources = sorted(pools)
    cursor = {source: 0 for source in sources}
    while remaining_budget and sources:
        progressed = False
        for source in sources:
            index = cursor[source]
            if index >= len(pools[source]):
                continue
            item = pools[source][index]
            cursor[source] += 1
            selected.append(item)
            selected_ids.add(item.conversation_id)
            remaining_budget -= 1
            progressed = True
            if remaining_budget == 0:
                break
        if not progressed:
            break

    return sorted(selected, key=lambda item: stable_rank(item.conversation_id))


def prefix_example(
    example: MultiTaskExample,
    size: int,
    stage_names: Sequence[str],
    unknown_tactic_negative_weight: float,
    window_kind: str = "prefix",
) -> MultiTaskExample:
    """Cut a conversation prefix and recompute only past-visible labels."""
    total = len(example.turn_texts)
    size = max(1, min(int(size), total))
    labels, mask, stage_label, stage_mask = aggregate_prefix_labels(
        example.turn_tactics[:size],
        example.turn_stages[:size],
        example.binary_label,
        stage_names,
        len(example.tactic_labels),
        example.tactics_annotated,
        example.stages_annotated,
        unknown_tactic_negative_weight,
    )
    return replace(
        example,
        text="\n".join(example.turn_texts[:size]),
        turn_texts=example.turn_texts[:size],
        tactic_labels=labels,
        tactic_mask=mask,
        stage_label=stage_label,
        stage_mask=stage_mask,
        turn_tactics=example.turn_tactics[:size],
        turn_stages=example.turn_stages[:size],
        window_id=f"{window_kind}_{size}_of_{total}",
    )


def create_context_windows(
    examples: Sequence[MultiTaskExample],
    positive_fractions: Sequence[float],
    negative_fractions: Sequence[float],
    maximum_windows: int,
    stage_names: Sequence[str],
    unknown_tactic_negative_weight: float,
) -> list[MultiTaskExample]:
    """Create deterministic prefix windows carrying causal supervision."""
    output: list[MultiTaskExample] = []
    for example in examples:
        fractions = positive_fractions if example.binary_label else negative_fractions
        seen_sizes: set[int] = set()
        for fraction in fractions:
            size = min(
                len(example.turn_texts),
                max(1, int(math.ceil(len(example.turn_texts) * float(fraction)))),
            )
            if size in seen_sizes:
                continue
            seen_sizes.add(size)
            output.append(
                prefix_example(
                    example,
                    size,
                    stage_names,
                    unknown_tactic_negative_weight,
                )
            )
            if len(seen_sizes) >= maximum_windows:
                break
    return output


def head_tail_token_ids(
    tokenizer: Any,
    text: str,
    max_length: int,
    head_ratio: float,
) -> tuple[list[int], list[int]]:
    """Encode long text as head + boundary + tail instead of right truncation."""
    ids = list(
        tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
    )
    return head_tail_from_ids(tokenizer, ids, max_length, head_ratio)


def head_tail_from_ids(
    tokenizer: Any,
    token_ids: Sequence[int],
    max_length: int,
    head_ratio: float,
) -> tuple[list[int], list[int]]:
    """Apply deterministic head-tail selection to already-tokenized IDs."""
    if not 0.0 < head_ratio < 1.0:
        raise ValueError("head_ratio must be between zero and one")
    ids = list(token_ids)
    budget = max(1, int(max_length) - tokenizer.num_special_tokens_to_add(pair=False))
    if len(ids) > budget:
        marker = getattr(tokenizer, "sep_token_id", None)
        inner = budget - (1 if marker is not None else 0)
        head = max(1, int(round(inner * float(head_ratio))))
        tail = max(0, inner - head)
        ids = ids[:head] + ([marker] if marker is not None else []) + (ids[-tail:] if tail else [])
    encoded = list(tokenizer.build_inputs_with_special_tokens(ids))
    attention = [1] * len(encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padding = max(0, int(max_length) - len(encoded))
    return encoded + [pad_id] * padding, attention + [0] * padding


def set_global_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        return


def configure_backbone_trainability(
    backbone: Any,
    train_last_n_layers: int,
    freeze_embeddings: bool,
) -> dict[str, int]:
    """Freeze large encoders except their final transformer blocks."""
    if train_last_n_layers < 0:
        raise ValueError("train_last_n_layers must be non-negative")
    for parameter in backbone.parameters():
        parameter.requires_grad = True
    if freeze_embeddings and hasattr(backbone, "embeddings"):
        for parameter in backbone.embeddings.parameters():
            parameter.requires_grad = False

    layers = None
    if hasattr(backbone, "transformer") and hasattr(backbone.transformer, "layer"):
        layers = backbone.transformer.layer
    elif hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        layers = backbone.encoder.layer
    elif hasattr(backbone, "albert_layer_groups"):
        # ALBERT shares parameters across logical layers; partial layer freezing
        # is not meaningful, so its encoder stays trainable.
        layers = None

    if layers is not None:
        cutoff = max(0, len(layers) - train_last_n_layers)
        for layer in layers[:cutoff]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    trainable = sum(
        parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in backbone.parameters())
    return {
        "trainable_backbone_parameters": trainable,
        "total_backbone_parameters": total,
    }


def build_torch_components() -> tuple[type[Any], type[Any]]:
    """Create torch classes lazily so data utilities do not require torch imports."""
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from transformers import AutoModel

    class EncodedMultiTaskDataset(Dataset):
        def __init__(
            self,
            examples: Sequence[MultiTaskExample],
            tokenizer: Any,
            max_length: int,
            head_ratio: float = 0.5,
        ):
            self.examples = list(examples)
            encoded_rows: list[tuple[list[int], list[int]]] = []
            batch_size = 512
            for start in range(0, len(self.examples), batch_size):
                batch = self.examples[start : start + batch_size]
                tokenized = tokenizer(
                    [item.text for item in batch],
                    add_special_tokens=False,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
                encoded_rows.extend(
                    head_tail_from_ids(tokenizer, ids, max_length, head_ratio)
                    for ids in tokenized
                )
            self.encodings = {
                "input_ids": torch.tensor([row[0] for row in encoded_rows], dtype=torch.long),
                "attention_mask": torch.tensor([row[1] for row in encoded_rows], dtype=torch.long),
            }

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = self.examples[index]
            row = {key: value[index] for key, value in self.encodings.items()}
            row.update(
                {
                    "binary_labels": torch.tensor(item.binary_label, dtype=torch.float32),
                    "scam_type_labels": torch.tensor(item.scam_type_label, dtype=torch.long),
                    "tactic_labels": torch.tensor(item.tactic_labels, dtype=torch.float32),
                    "tactic_mask": torch.tensor(item.tactic_mask, dtype=torch.float32),
                    "stage_labels": torch.tensor(item.stage_label, dtype=torch.long),
                    "stage_mask": torch.tensor(item.stage_mask, dtype=torch.float32),
                    "sample_weight": torch.tensor(item.sample_weight, dtype=torch.float32),
                    "row_index": torch.tensor(index, dtype=torch.long),
                }
            )
            return row

    class MultiTaskDetector(nn.Module):
        def __init__(
            self,
            backbone_path: str,
            scam_type_count: int,
            tactic_count: int,
            stage_count: int,
            dropout: float,
            loss_weights: Mapping[str, float],
            trust_remote_code: bool = False,
        ) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                backbone_path, trust_remote_code=trust_remote_code
            )
            hidden_size = int(self.backbone.config.hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.binary_head = nn.Linear(hidden_size, 1)
            self.scam_type_head = nn.Linear(hidden_size, scam_type_count)
            self.tactic_head = nn.Linear(hidden_size, tactic_count)
            self.stage_head = nn.Linear(hidden_size, stage_count)
            self.loss_weights = dict(loss_weights)

        def forward(self, **batch: Any) -> dict[str, Any]:
            model_inputs = {
                key: value
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            outputs = self.backbone(**model_inputs)
            pooled = self.dropout(outputs.last_hidden_state[:, 0])
            binary_logits = self.binary_head(pooled).squeeze(-1)
            scam_type_logits = self.scam_type_head(pooled)
            tactic_logits = self.tactic_head(pooled)
            stage_logits = self.stage_head(pooled)
            result = {
                "binary_logits": binary_logits,
                "scam_type_logits": scam_type_logits,
                "tactic_logits": tactic_logits,
                "stage_logits": stage_logits,
            }
            if "binary_labels" not in batch:
                return result

            sample_weight = batch["sample_weight"]
            binary_loss = nn.functional.binary_cross_entropy_with_logits(
                binary_logits, batch["binary_labels"], reduction="none"
            )
            binary_loss = (binary_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            scam_type_loss = nn.functional.cross_entropy(
                scam_type_logits, batch["scam_type_labels"], reduction="none"
            )
            scam_type_loss = (scam_type_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
            tactic_loss = nn.functional.binary_cross_entropy_with_logits(
                tactic_logits, batch["tactic_labels"], reduction="none"
            )
            tactic_weights = batch["tactic_mask"] * sample_weight.unsqueeze(1)
            tactic_loss = (
                tactic_loss * tactic_weights
            ).sum() / tactic_weights.sum().clamp_min(1.0)
            stage_loss = nn.functional.cross_entropy(
                stage_logits, batch["stage_labels"], reduction="none"
            )
            stage_weights = batch["stage_mask"] * sample_weight
            stage_loss = (stage_loss * stage_weights).sum() / stage_weights.sum().clamp_min(1.0)
            losses = {
                "binary": binary_loss,
                "scam_type": scam_type_loss,
                "tactics": tactic_loss,
                "stage": stage_loss,
            }
            result["task_losses"] = losses
            result["loss"] = sum(
                losses[name] * float(self.loss_weights[name]) for name in losses
            )
            return result

    return EncodedMultiTaskDataset, MultiTaskDetector
