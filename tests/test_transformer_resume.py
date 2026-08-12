from pathlib import Path
import sys

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_multitask_transformer import (  # noqa: E402
    capture_rng_state,
    deterministic_epoch_order,
    load_trainable_state_dict,
    restore_rng_state,
    save_step_checkpoint,
    trainable_state_dict,
    training_signature,
)


def test_trainable_checkpoint_excludes_frozen_parameters_and_restores() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    saved = trainable_state_dict(model)
    assert set(saved) == {"1.weight", "1.bias"}
    expected = saved["1.weight"].clone()
    with torch.no_grad():
        model[1].weight.add_(10)
    load_trainable_state_dict(model, saved)
    assert torch.equal(model[1].weight, expected)
    with pytest.raises(ValueError, match="Unexpected checkpoint keys"):
        load_trainable_state_dict(model, {"not_a_parameter": torch.tensor(1)})


def test_rng_restore_reproduces_python_numpy_and_torch_streams() -> None:
    import random

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.random()), float(torch.rand(1)))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(1)))
    assert actual == pytest.approx(expected)


def test_epoch_order_and_atomic_checkpoint_are_deterministic(tmp_path: Path) -> None:
    first = deterministic_epoch_order(20, seed=42, epoch=2)
    second = deterministic_epoch_order(20, seed=42, epoch=2)
    assert first == second
    assert sorted(first) == list(range(20))
    assert first != deterministic_epoch_order(20, seed=42, epoch=3)

    path = tmp_path / "checkpoint.pt"
    save_step_checkpoint(path, {"value": torch.tensor([1, 2, 3])})
    assert torch.equal(torch.load(path, weights_only=True)["value"], torch.tensor([1, 2, 3]))
    assert not path.with_suffix(".pt.part").exists()


def test_training_signature_changes_when_training_contract_changes() -> None:
    config = {
        "run_id": "fixture",
        "model": {"max_length": 256},
        "training": {"batch_size": 16, "gradient_accumulation_steps": 2},
    }
    first = training_signature(config, "backbone", 100, 20, [17, 42], 3)
    changed = {**config, "model": {"max_length": 128}}
    second = training_signature(changed, "backbone", 100, 20, [17, 42], 3)
    assert first["config_sha256"] != second["config_sha256"]
    assert first["max_length"] == 256
