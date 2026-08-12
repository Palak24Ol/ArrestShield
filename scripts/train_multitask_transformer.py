"""Train the multilingual multi-task detector under the pre-registered protocol."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arrestshield.artifacts import git_revision, sha256_file, write_json  # noqa: E402
from arrestshield.evaluation import binary_metrics  # noqa: E402
from arrestshield.ladder import stable_latency_from_flat_scores  # noqa: E402
from arrestshield.multitask import (  # noqa: E402
    MultiTaskExample,
    build_torch_components,
    configure_backbone_trainability,
    create_context_windows,
    load_multitask_examples,
    prefix_example,
    set_global_seed,
    sortish_epoch_order,
    source_balanced_training_sample,
    trim_padded_batch,
)
from arrestshield.protocol import select_threshold_at_fpr, summarize_seed_values  # noqa: E402


LOGGER = logging.getLogger("arrestshield.train_multitask_transformer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/model/multitask_transformer.json")
    parser.add_argument("--conversations", type=Path, default=PROJECT_ROOT / "data/processed/conversations.jsonl")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data/splits/split_manifest.json")
    parser.add_argument("--views", type=Path, default=PROJECT_ROOT / "data/splits/evaluation_views.json")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts/models/multitask_transformer_v1")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/multitask_transformer_v1")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-train-conversations", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any local step checkpoint and start the configured run again.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_backbone(config: Mapping[str, Any], override: str | None) -> str:
    if override:
        return override
    local_path = PROJECT_ROOT / config["backbone"]["local_pretrained_path"]
    return str(local_path) if local_path.exists() else str(config["backbone"]["model_id"])


def limit_examples(examples: Sequence[MultiTaskExample], maximum: int | None) -> list[MultiTaskExample]:
    if not maximum or maximum <= 0 or len(examples) <= maximum:
        return list(examples)
    positives = [item for item in examples if item.binary_label == 1]
    negatives = [item for item in examples if item.binary_label == 0]
    each = max(1, maximum // 2)
    limited = positives[:each] + negatives[: max(0, maximum - min(each, len(positives)))]
    return limited[:maximum]


def primary_indices(examples: Sequence[MultiTaskExample], hard_negative_ids: set[str]) -> list[int]:
    return [
        index for index, item in enumerate(examples)
        if item.binary_label == 1 or item.conversation_id in hard_negative_ids
    ]


def subset(values: Sequence[Any] | np.ndarray, indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items() if key != "row_index"}


def trainable_state_dict(model: Any) -> dict[str, Any]:
    """Copy only trainable tensors; frozen backbone weights come from the pinned checkpoint."""
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state_dict(model: Any, state: Mapping[str, Any]) -> None:
    """Restore a partial trainable-only state and reject unknown tensor names."""
    result = model.load_state_dict(dict(state), strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {result.unexpected_keys}")


def training_signature(
    config: Mapping[str, Any],
    backbone: str,
    training_examples: int,
    validation_examples: int,
    seeds: Sequence[int],
    epochs: int,
) -> dict[str, Any]:
    """Fields that must match before a step checkpoint can be resumed."""
    return {
        "run_id": config["run_id"],
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "backbone": str(backbone),
        "training_examples": int(training_examples),
        "validation_examples": int(validation_examples),
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "max_length": int(config["model"]["max_length"]),
        "batch_size": int(config["training"]["batch_size"]),
        "gradient_accumulation_steps": int(
            config["training"]["gradient_accumulation_steps"]
        ),
        "sortish_mega_batch_multiplier": int(
            config["training"]["sortish_mega_batch_multiplier"]
        ),
    }


def save_step_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace the local resumable training checkpoint."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def capture_rng_state() -> dict[str, Any]:
    """Capture stochastic state so a resumed dropout trajectory is reproducible."""
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore the exact state recorded after the last completed optimizer update."""
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def predict(model: Any, loader: Any, device: Any) -> dict[str, np.ndarray]:
    import torch

    model.eval()
    binary: list[np.ndarray] = []
    scam_type: list[np.ndarray] = []
    tactics: list[np.ndarray] = []
    stages: list[np.ndarray] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(**batch)
            binary.append(torch.sigmoid(output["binary_logits"]).cpu().numpy())
            scam_type.append(torch.softmax(output["scam_type_logits"], dim=-1).cpu().numpy())
            tactics.append(torch.sigmoid(output["tactic_logits"]).cpu().numpy())
            stages.append(torch.softmax(output["stage_logits"], dim=-1).cpu().numpy())
    return {
        "binary": np.concatenate(binary),
        "scam_type": np.concatenate(scam_type),
        "tactics": np.concatenate(tactics),
        "stage": np.concatenate(stages),
    }


def auxiliary_metrics(
    examples: Sequence[MultiTaskExample],
    predictions: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    type_true = np.asarray([item.scam_type_label for item in examples])
    type_pred = np.asarray(predictions["scam_type"]).argmax(axis=1)
    positive_indices = np.asarray([item.binary_label == 1 for item in examples])
    result: dict[str, Any] = {
        "scam_type_macro_f1_all": float(f1_score(type_true, type_pred, average="macro", zero_division=0)),
        "scam_type_macro_f1_positive_only": (
            float(f1_score(type_true[positive_indices], type_pred[positive_indices], average="macro", zero_division=0))
            if positive_indices.any() else None
        ),
        "tactic_metric_scope": "observed positives versus confirmed non-scam negatives; not fully supervised tactic classification",
    }

    tactic_names = config["labels"]["tactics"]
    tactic_true = np.asarray([item.tactic_labels for item in examples], dtype=np.float32)
    tactic_mask = np.asarray([item.tactic_mask for item in examples], dtype=np.float32)
    tactic_scores = np.asarray(predictions["tactics"])
    tactic_rows: dict[str, Any] = {}
    for index, name in enumerate(tactic_names):
        valid = tactic_mask[:, index] == 1
        valid_true = tactic_true[valid, index]
        valid_scores = tactic_scores[valid, index]
        has_both = len(set(valid_true.tolist())) == 2
        tactic_rows[name] = {
            "supervised_examples": int(valid.sum()),
            "positive_examples": int(valid_true.sum()),
            "average_precision": float(average_precision_score(valid_true, valid_scores)) if has_both else None,
            "f1_at_0_5": float(f1_score(valid_true, valid_scores >= 0.5, zero_division=0)),
        }
    result["tactics"] = tactic_rows

    stage_mask = np.asarray([item.stage_mask == 1 for item in examples])
    stage_true = np.asarray([item.stage_label for item in examples])
    stage_pred = np.asarray(predictions["stage"]).argmax(axis=1)
    result["stage_supervised_examples"] = int(stage_mask.sum())
    result["stage_macro_f1"] = (
        float(f1_score(stage_true[stage_mask], stage_pred[stage_mask], average="macro", zero_division=0))
        if stage_mask.any() else None
    )
    return result


def make_prefix_examples(
    examples: Sequence[MultiTaskExample],
    stage_names: Sequence[str],
    unknown_tactic_negative_weight: float,
) -> tuple[list[MultiTaskExample], list[MultiTaskExample], list[tuple[int, int]]]:
    positives = [item for item in examples if item.binary_label == 1]
    prefixes: list[MultiTaskExample] = []
    owners: list[tuple[int, int]] = []
    for example_index, item in enumerate(positives):
        for prefix_size in range(1, len(item.turn_texts) + 1):
            prefixes.append(
                prefix_example(
                    item,
                    prefix_size,
                    stage_names,
                    unknown_tactic_negative_weight,
                    window_kind="latency",
                )
            )
            owners.append((example_index, prefix_size))
    return positives, prefixes, owners


def export_selected_model(
    model: Any,
    tokenizer: Any,
    artifact_dir: Path,
    config: Mapping[str, Any],
    threshold: float,
    exit_threshold: float,
    seed: int,
) -> Path:
    import torch

    artifact_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir = artifact_dir / "backbone"
    tokenizer_dir = artifact_dir / "tokenizer"
    model.backbone.save_pretrained(backbone_dir, safe_serialization=True)
    tokenizer.save_pretrained(tokenizer_dir)
    heads_path = artifact_dir / "multitask_heads.pt"
    torch.save(
        {
            "binary_head": model.binary_head.state_dict(),
            "scam_type_head": model.scam_type_head.state_dict(),
            "tactic_head": model.tactic_head.state_dict(),
            "stage_head": model.stage_head.state_dict(),
        },
        heads_path,
    )
    manifest = {
        "schema_version": "1.0.0",
        "model_family": "multilingual_distilbert_multitask",
        "selection_role": config["backbone"]["selection_role"],
        "seed": seed,
        "threshold": threshold,
        "exit_threshold": exit_threshold,
        "max_length": config["model"]["max_length"],
        "context_encoding": config["model"]["context_encoding"],
        "head_tail_ratio": config["model"]["head_tail_ratio"],
        "dropout": config["model"]["dropout"],
        "unknown_tactic_policy": config["data"]["unknown_tactic_policy"],
        "unknown_tactic_negative_weight": config["data"]["unknown_tactic_negative_weight"],
        "loss_weights": config["model"]["loss_weights"],
        "labels": config["labels"],
        "backbone_directory": "backbone",
        "tokenizer_directory": "tokenizer",
        "heads_file": "multitask_heads.pt",
        "llm_used_for_detection": False,
    }
    write_json(artifact_dir / "manifest.json", manifest)
    return heads_path


def main() -> int:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = read_json(args.config)
    protocol = read_json(PROJECT_ROOT / config["protocol_path"])
    views = read_json(args.views)
    data_by_split = load_multitask_examples(args.conversations, args.splits, config)
    data_config = config["data"]
    sampled_train = source_balanced_training_sample(
        data_by_split["train"],
        float(data_config["maximum_negative_to_positive_ratio"]),
        float(data_config["mixed_source_negative_ratio"]),
    )
    if args.max_train_conversations:
        sampled_train = limit_examples(sampled_train, args.max_train_conversations)
    training_examples = create_context_windows(
        sampled_train,
        data_config["positive_prefix_fractions"],
        data_config["negative_prefix_fractions"],
        int(data_config["maximum_windows_per_conversation"]),
        config["labels"]["stages"],
        float(data_config["unknown_tactic_negative_weight"]),
    )
    validation_examples = limit_examples(data_by_split["validation"], args.max_validation)
    test_examples = limit_examples(data_by_split["test"], args.max_test)
    if args.smoke:
        training_examples = limit_examples(training_examples, 128)
        validation_examples = limit_examples(validation_examples, 128)
        test_examples = limit_examples(test_examples, 128)

    backbone = choose_backbone(config, args.backbone)
    LOGGER.info("Loading tokenizer and backbone %s", backbone)
    tokenizer = AutoTokenizer.from_pretrained(
        backbone,
        trust_remote_code=bool(config["backbone"].get("trust_remote_code", False)),
        use_fast=True,
    )
    DatasetClass, ModelClass = build_torch_components()
    max_length = int(config["model"]["max_length"])
    LOGGER.info(
        "Encoding %s train windows, %s validation and %s test conversations",
        len(training_examples), len(validation_examples), len(test_examples),
    )
    head_ratio = float(config["model"]["head_tail_ratio"])
    train_dataset = DatasetClass(training_examples, tokenizer, max_length, head_ratio)
    validation_dataset = DatasetClass(validation_examples, tokenizer, max_length, head_ratio)
    test_dataset = DatasetClass(test_examples, tokenizer, max_length, head_ratio)

    validation_ids = set(views["hard_negative_ids"]["validation"])
    validation_primary = primary_indices(validation_examples, validation_ids)
    test_primary = primary_indices(test_examples, set(views["hard_negative_ids"]["test"]))
    if args.smoke:
        validation_primary = list(range(len(validation_examples)))
        test_primary = list(range(len(test_examples)))
    training_config = config["training"]
    seeds = args.seeds or [int(value) for value in training_config["seeds"]]
    epochs = args.epochs or int(training_config["epochs"])
    if args.smoke:
        seeds = [seeds[0]]
        epochs = 1
    deployment_seed = int(training_config["deployment_seed"])
    if deployment_seed not in seeds:
        deployment_seed = seeds[0]
    torch.set_num_threads(int(training_config["torch_threads"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training on %s with seeds %s", device, seeds)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(training_config["batch_size"])
    evaluation_batch_size = int(training_config["evaluation_batch_size"])
    accumulation = int(training_config["gradient_accumulation_steps"])
    maximum_fpr = float(protocol["primary_selection"]["maximum"])
    exit_ratio = float(protocol["hysteresis"]["exit_threshold_ratio"])
    started = time.perf_counter()
    checkpoint_path = args.artifact_dir / "training_checkpoint.pt"
    signature = training_signature(
        config,
        backbone,
        len(training_examples),
        len(validation_examples),
        seeds,
        epochs,
    )
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    resume_payload: dict[str, Any] | None = None
    if checkpoint_path.exists():
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if loaded.get("signature") != signature:
            raise ValueError(
                "Existing transformer checkpoint does not match this run. "
                "Use --no-resume only if starting over is intentional."
            )
        resume_payload = loaded
        LOGGER.info(
            "Resuming local checkpoint active_seed=%s epoch=%s next_batch=%s",
            loaded.get("active_seed"),
            loaded.get("epoch"),
            loaded.get("next_batch"),
        )
    resumed_from_checkpoint = resume_payload is not None
    elapsed_before_resume = float(
        (resume_payload or {}).get("elapsed_training_seconds", 0.0)
    )
    recovered_optimizer_steps = int(
        (resume_payload or {}).get("optimizer_steps", 0)
        if (resume_payload or {}).get("active_seed") is not None
        else 0
    )
    seed_runs: list[dict[str, Any]] = list((resume_payload or {}).get("seed_runs", []))
    completed_seeds = {
        int(seed) for seed in (resume_payload or {}).get("completed_seeds", [])
    }
    deployment_manifest: dict[str, Any] | None = (resume_payload or {}).get(
        "deployment_manifest"
    )
    batch_count = math.ceil(len(train_dataset) / batch_size)
    optimizer_steps_per_epoch = math.ceil(batch_count / accumulation)
    total_steps = max(1, optimizer_steps_per_epoch * epochs)
    warmup_steps = int(total_steps * float(training_config["warmup_ratio"]))
    progress_interval = max(
        1, int(training_config.get("progress_log_every_optimizer_steps", 5))
    )
    checkpoint_interval = max(
        1, int(training_config.get("checkpoint_every_optimizer_steps", 10))
    )

    for seed in seeds:
        if seed in completed_seeds:
            LOGGER.info("Skipping already completed seed %s", seed)
            continue
        set_global_seed(seed, bool(training_config["deterministic_algorithms"]))
        model = ModelClass(
            backbone,
            len(config["labels"]["scam_types"]),
            len(config["labels"]["tactics"]),
            len(config["labels"]["stages"]),
            float(config["model"]["dropout"]),
            config["model"]["loss_weights"],
            bool(config["backbone"].get("trust_remote_code", False)),
        ).to(device)
        parameter_summary = configure_backbone_trainability(
            model.backbone,
            int(training_config["train_last_n_backbone_layers"]),
            bool(training_config["freeze_embeddings"]),
        )
        LOGGER.info("Backbone parameters %s", parameter_summary)
        head_parameters = (
            list(model.binary_head.parameters())
            + list(model.scam_type_head.parameters())
            + list(model.tactic_head.parameters())
            + list(model.stage_head.parameters())
        )
        optimizer = AdamW(
            [
                {
                    "params": [
                        parameter
                        for parameter in model.backbone.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": float(training_config["backbone_learning_rate"]),
                },
                {
                    "params": head_parameters,
                    "lr": float(training_config["head_learning_rate"]),
                },
            ],
            weight_decay=float(training_config["weight_decay"]),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=evaluation_batch_size,
            shuffle=False,
            collate_fn=trim_padded_batch,
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        best_key: tuple[float, float, float] | None = None
        best_state: dict[str, Any] | None = None
        best_epoch: dict[str, Any] | None = None
        patience = 0
        start_epoch = 1
        resume_batch = 0
        running_loss = 0.0
        optimizer_steps = 0

        if resume_payload is not None and int(resume_payload.get("active_seed", -1)) == seed:
            load_trainable_state_dict(model, resume_payload["trainable_state"])
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            scheduler.load_state_dict(resume_payload["scheduler_state"])
            start_epoch = int(resume_payload["epoch"])
            resume_batch = int(resume_payload["next_batch"])
            running_loss = float(resume_payload.get("running_loss", 0.0))
            optimizer_steps = int(resume_payload.get("optimizer_steps", 0))
            raw_best_key = resume_payload.get("best_key")
            best_key = tuple(raw_best_key) if raw_best_key is not None else None
            best_state = resume_payload.get("best_state")
            best_epoch = resume_payload.get("best_epoch")
            patience = int(resume_payload.get("patience", 0))
            restore_rng_state(resume_payload["rng_state"])
            LOGGER.info(
                "Recovered seed=%s optimizer_steps=%s at epoch=%s batch=%s/%s",
                seed,
                optimizer_steps,
                start_epoch,
                resume_batch,
                batch_count,
            )

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if epoch != start_epoch or resume_batch == 0:
                running_loss = 0.0
            order = sortish_epoch_order(
                train_dataset.token_lengths,
                batch_size,
                seed,
                epoch,
                int(training_config["sortish_mega_batch_multiplier"]),
            )
            first_example = min(len(order), resume_batch * batch_size)
            remaining = Subset(train_dataset, order[first_example:])
            loader_generator = torch.Generator().manual_seed(
                int(seed) * 1_000_003 + int(epoch)
            )
            train_loader = DataLoader(
                remaining,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(training_config["num_workers"]),
                generator=loader_generator,
                collate_fn=trim_padded_batch,
            )
            for local_step, raw_batch in enumerate(train_loader, start=1):
                step = resume_batch + local_step
                batch = move_batch(raw_batch, device)
                output = model(**batch)
                loss = output["loss"] / accumulation
                loss.backward()
                running_loss += float(output["loss"].detach().cpu())
                if step % accumulation == 0 or step == batch_count:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config["gradient_clip_norm"]))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if optimizer_steps % progress_interval == 0:
                        LOGGER.info(
                            "seed=%s epoch=%s batch=%s/%s optimizer_step=%s partial_loss=%.4f",
                            seed,
                            epoch,
                            step,
                            batch_count,
                            optimizer_steps,
                            running_loss / max(1, step),
                        )
                    if optimizer_steps % checkpoint_interval == 0:
                        save_step_checkpoint(
                            checkpoint_path,
                            {
                                "schema_version": "1.0.0",
                                "signature": signature,
                                "elapsed_training_seconds": elapsed_before_resume
                                + time.perf_counter()
                                - started,
                                "active_seed": seed,
                                "completed_seeds": sorted(completed_seeds),
                                "seed_runs": seed_runs,
                                "deployment_manifest": deployment_manifest,
                                "epoch": epoch,
                                "next_batch": step,
                                "optimizer_steps": optimizer_steps,
                                "running_loss": running_loss,
                                "trainable_state": trainable_state_dict(model),
                                "optimizer_state": optimizer.state_dict(),
                                "scheduler_state": scheduler.state_dict(),
                                "best_key": best_key,
                                "best_state": best_state,
                                "best_epoch": best_epoch,
                                "patience": patience,
                                "rng_state": capture_rng_state(),
                            },
                        )
                        LOGGER.info("Saved resumable checkpoint at %s", checkpoint_path)

            validation_predictions = predict(model, validation_loader, device)
            validation_labels = [item.binary_label for item in validation_examples]
            operating = select_threshold_at_fpr(
                subset(validation_labels, validation_primary),
                subset(validation_predictions["binary"], validation_primary),
                maximum_fpr=maximum_fpr,
            )
            metrics = binary_metrics(
                subset(validation_labels, validation_primary),
                subset(validation_predictions["binary"], validation_primary),
                operating.threshold,
            )
            key = (metrics["recall"], metrics["macro_f1"], -operating.threshold)
            epoch_row = {
                "epoch": epoch,
                "mean_training_loss": running_loss / max(1, batch_count),
                "operating_point": asdict(operating),
                "validation_primary": metrics,
            }
            LOGGER.info(
                "seed=%s epoch=%s loss=%.4f val_recall=%.4f val_fpr=%.4f",
                seed, epoch, epoch_row["mean_training_loss"], metrics["recall"], metrics["false_positive_rate"],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = trainable_state_dict(model)
                best_epoch = epoch_row
                patience = 0
            else:
                patience += 1
                if patience > int(training_config["early_stopping_patience"]):
                    break
            resume_batch = 0
            if epoch < epochs:
                save_step_checkpoint(
                    checkpoint_path,
                    {
                        "schema_version": "1.0.0",
                        "signature": signature,
                        "elapsed_training_seconds": elapsed_before_resume
                        + time.perf_counter()
                        - started,
                        "active_seed": seed,
                        "completed_seeds": sorted(completed_seeds),
                        "seed_runs": seed_runs,
                        "deployment_manifest": deployment_manifest,
                        "epoch": epoch + 1,
                        "next_batch": 0,
                        "optimizer_steps": optimizer_steps,
                        "running_loss": 0.0,
                        "trainable_state": trainable_state_dict(model),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_key": best_key,
                        "best_state": best_state,
                        "best_epoch": best_epoch,
                        "patience": patience,
                        "rng_state": capture_rng_state(),
                    },
                )

        if best_state is None or best_epoch is None:
            raise RuntimeError("Training produced no checkpoint")
        load_trainable_state_dict(model, best_state)
        model.to(device)
        run = {"seed": seed, "best": best_epoch, "parameters": parameter_summary}
        seed_runs.append(run)
        if seed == deployment_seed:
            threshold = float(best_epoch["operating_point"]["threshold"])
            heads_path = export_selected_model(
                model, tokenizer, args.artifact_dir, config, threshold, threshold * exit_ratio, seed
            )
            deployment_manifest = {
                "threshold": threshold,
                "exit_threshold": threshold * exit_ratio,
                "heads_path": str(heads_path),
            }
        completed_seeds.add(seed)
        save_step_checkpoint(
            checkpoint_path,
            {
                "schema_version": "1.0.0",
                "signature": signature,
                "elapsed_training_seconds": elapsed_before_resume
                + time.perf_counter()
                - started,
                "active_seed": None,
                "completed_seeds": sorted(completed_seeds),
                "seed_runs": seed_runs,
                "deployment_manifest": deployment_manifest,
                "optimizer_steps": 0,
            },
        )
        resume_payload = None
        del model, optimizer, scheduler

    if deployment_manifest is None:
        raise RuntimeError("Deployment seed was not exported")

    # Reload the exported model so evaluation also tests artifact reconstructability.
    exported_backbone = str(args.artifact_dir / "backbone")
    model = ModelClass(
        exported_backbone,
        len(config["labels"]["scam_types"]),
        len(config["labels"]["tactics"]),
        len(config["labels"]["stages"]),
        float(config["model"]["dropout"]),
        config["model"]["loss_weights"],
        False,
    )
    heads = torch.load(args.artifact_dir / "multitask_heads.pt", map_location="cpu", weights_only=True)
    model.binary_head.load_state_dict(heads["binary_head"])
    model.scam_type_head.load_state_dict(heads["scam_type_head"])
    model.tactic_head.load_state_dict(heads["tactic_head"])
    model.stage_head.load_state_dict(heads["stage_head"])
    model.to(device)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        collate_fn=trim_padded_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        collate_fn=trim_padded_batch,
    )
    validation_predictions = predict(model, validation_loader, device)
    test_predictions = predict(model, test_loader, device)
    threshold = float(deployment_manifest["threshold"])
    exit_threshold = float(deployment_manifest["exit_threshold"])
    validation_labels = [item.binary_label for item in validation_examples]
    test_labels = [item.binary_label for item in test_examples]

    positive_test, prefix_test, prefix_owners = make_prefix_examples(
        test_examples,
        config["labels"]["stages"],
        float(data_config["unknown_tactic_negative_weight"]),
    )
    prefix_dataset = DatasetClass(prefix_test, tokenizer, max_length, head_ratio)
    prefix_loader = DataLoader(
        prefix_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        collate_fn=trim_padded_batch,
    )
    prefix_predictions = predict(model, prefix_loader, device) if prefix_test else {"binary": np.asarray([])}
    latency = stable_latency_from_flat_scores(
        positive_test, prefix_owners, prefix_predictions["binary"], threshold, exit_threshold
    )
    report = {
        "schema_version": "1.0.0",
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(PROJECT_ROOT),
        "device": str(device),
        "backbone": backbone,
        "selection_role": config["backbone"]["selection_role"],
        "seeds": seeds,
        "deployment_seed": deployment_seed,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "data": {
            "eligible_train_conversations": len(data_by_split["train"]),
            "sampled_train_conversations": len(sampled_train),
            "training_context_windows": len(training_examples),
            "validation_conversations": len(validation_examples),
            "test_conversations": len(test_examples),
            "unknown_tactic_policy": data_config["unknown_tactic_policy"],
            "unknown_tactic_negative_weight": float(data_config["unknown_tactic_negative_weight"]),
            "causal_prefix_labels": True,
            "context_encoding": config["model"]["context_encoding"],
            "dynamic_padding_trim": True,
            "sortish_mega_batch_multiplier": int(
                training_config["sortish_mega_batch_multiplier"]
            ),
        },
        "seed_runs": seed_runs,
        "validation_primary": binary_metrics(
            subset(validation_labels, validation_primary),
            subset(validation_predictions["binary"], validation_primary),
            threshold,
        ),
        "test_primary": binary_metrics(
            subset(test_labels, test_primary),
            subset(test_predictions["binary"], test_primary),
            threshold,
        ),
        "test_full": binary_metrics(test_labels, test_predictions["binary"], threshold),
        "test_auxiliary_tasks": auxiliary_metrics(test_examples, test_predictions, config),
        "test_stable_latency": latency,
        "seed_validation_recall": summarize_seed_values(
            [row["best"]["validation_primary"]["recall"] for row in seed_runs]
        ),
        "runtime_seconds": elapsed_before_resume + time.perf_counter() - started,
        "resumable_training": {
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "recovered_optimizer_steps": recovered_optimizer_steps,
            "checkpoint_every_optimizer_steps": checkpoint_interval,
            "progress_log_every_optimizer_steps": progress_interval,
            "checkpoint_removed_after_success": True,
        },
        "limitations": [
            "Positive labels are source-silver; unobserved tactics inside annotated conversations are fixed 0.25-weight negatives, not confirmed absences.",
            "feasibility_only: multilingual DistilBERT with frozen embeddings and one trainable layer is a CPU harness, not evidence about IndicBERT, HingRoBERTa-Mixed, MuRIL, or XLM-R.",
            "Test results are supporting evidence only and were computed after checkpoint selection.",
            "No LLM output or LLM judgment is used for scam detection.",
        ],
    }
    heads_path = args.artifact_dir / "multitask_heads.pt"
    metadata = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(args.artifact_dir / "manifest.json"),
        "heads_sha256": sha256_file(heads_path),
        "heads_bytes": heads_path.stat().st_size,
    }
    write_json(args.report_dir / "metrics.json", report)
    write_json(args.report_dir / "run_metadata.json", metadata)
    write_json(args.report_dir / "config.json", config)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    LOGGER.info("Completed transformer run in %.1f seconds", report["runtime_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
