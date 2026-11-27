"""Training workload identity, runtime reuse and live state snapshots."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Never

import alphabet_imagenet100_runtime as va
import automatic_training_layout as layout
import effective_batch_runtime as effective
import effective_batch_search as search
import numpy as np
import pytest
import torch
from alphabet_model_builder import model_config
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path


def make_runtime[Runtime: layout.AutomaticTrainingRuntime](
    tmp_path: Path,
    runtime_type: type[Runtime] = layout.AutomaticTrainingRuntime,
) -> Runtime:
    model = nn.Linear(3, 2)
    return runtime_type(
        model,
        torch.optim.AdamW(model.parameters()),
        {"precision": "float32", "batch_size": 32, "effective_batch_size": 32},
        {"output_dim": 2},
        "toy",
        SimpleNamespace(
            build_model=va.default_model_config,
            model_config=model_config,
            build_optimizer=va.default_model_config,
            prepare_model=va.prepare_model,
            train_epoch=va.train_epoch,
        ),
        tmp_path,
        torch.Generator(),
        np.random.default_rng(0),
    )


def test_workload_normalizes_scan_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("ALPHABET_GENERAL_SCANS", "ALPHABET_FUSED_FACTORIZED_READER"):
        monkeypatch.delenv(name, raising=False)
    default = make_runtime(tmp_path)
    monkeypatch.setenv("ALPHABET_GENERAL_SCANS", "1")
    monkeypatch.setenv("ALPHABET_FUSED_FACTORIZED_READER", "1")
    assert make_runtime(tmp_path).identity == default.identity
    monkeypatch.setenv("ALPHABET_GENERAL_SCANS", "0")
    assert make_runtime(tmp_path).identity["scan_env"] == {
        "ALPHABET_GENERAL_SCANS": "0",
        "ALPHABET_FUSED_FACTORIZED_READER": "1",
    }


def test_reuses_training_runtime_and_evaluates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = make_runtime(tmp_path)
    compiled = []

    def build(model: nn.Module, _recipe: dict[str, str]) -> nn.Module:
        compiled.append(model)
        return model

    monkeypatch.setattr(va, "_build_runtime", build)
    x = torch.ones(2, 3)
    y = torch.zeros(2, dtype=torch.long)
    for _ in range(2):
        with layout.training_microbatch(runtime, x, y, y, 1.0, 1, 0) as active:
            assert active is runtime.model
    assert compiled == [runtime.model]
    torch.testing.assert_close(runtime(x), runtime.model(x))
    assert compiled == [runtime.model, runtime.model]


def test_cpu_snapshots_are_independent_and_rng_round_trips(tmp_path: Path) -> None:
    runtime = make_runtime(tmp_path)
    device = torch.device("cpu")
    state = runtime.live_state(device)
    snapshot = layout.cpu_tree(state)
    expected = layout.state_hash(snapshot)
    assert layout.state_hash(state) == expected
    with torch.no_grad():
        runtime.model.weight.add_(1)
    assert layout.state_hash(runtime.live_state(device)) != expected
    assert layout.state_hash(snapshot) == expected

    try:
        sample = torch.rand(3)
        layout.restore_rng(snapshot["rng"], device)
        torch.testing.assert_close(torch.rand(3), sample, rtol=0, atol=0)
    finally:
        layout.restore_rng(snapshot["rng"], device)


@pytest.fixture
def mock_calibration_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHABET_LAUNCH_CACHE", str(tmp_path))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(name="test GPU")
    )
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda _: 1.0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (8 << 30, 8 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", nullcontext)


@pytest.mark.usefixtures("mock_calibration_gpu")
def test_effective_runtime_reuses_receipt_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def make_effective() -> effective.EffectiveTrainingRuntime:
        return make_runtime(tmp_path, effective.EffectiveTrainingRuntime)

    def arm(_work: Path, _label: str, micro: int) -> tuple[dict[str, Any], layout.TensorCapture]:
        assert micro == 32
        capture = {group: {"v": torch.ones(1)} for group in ("outputs", "grads", "updates")}
        return {"status": "ok", "ms": 10, "images_per_second": 3200, "peak_mib": 20}, capture

    monkeypatch.setattr(search, "_run_arm", arm)
    first = make_effective()
    batch = (
        torch.ones(32, 3),
        torch.zeros(32, dtype=torch.long),
        torch.ones(32, dtype=torch.long),
        1.0,
    )
    before = layout.state_hash(first.live_state(torch.device("cpu")))
    key, decision = first.configure(batch)
    assert layout.state_hash(first.live_state(torch.device("cpu"))) == before
    assert decision["state_before"] == decision["state_after"]
    assert decision["microbatch"] == 32
    assert (first.base / (key + ".json")).is_file()

    second = make_effective()

    def unexpected(*_args: object) -> Never:
        raise AssertionError("receipt should be reused")

    monkeypatch.setattr(effective, "measure", unexpected)
    cached_key, cached = second.configure(batch)
    assert cached_key == key
    assert layout.digest(cached) == layout.digest(decision)


@pytest.mark.usefixtures("mock_calibration_gpu")
def test_retryable_fallback_is_retried_next_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unavailable(*_args: object) -> Never:
        raise RuntimeError("worker interrupted")

    monkeypatch.setattr(effective, "measure", unavailable)
    runtime = make_runtime(tmp_path, effective.EffectiveTrainingRuntime)
    batch = (
        torch.ones(32, 3),
        torch.zeros(32, dtype=torch.long),
        torch.ones(32, dtype=torch.long),
        1.0,
    )
    with pytest.warns(RuntimeWarning, match="worker interrupted"):
        key, decision = runtime.configure(batch)
    assert decision == {"status": "retryable-fallback", "microbatch": 32}
    assert runtime.configure(batch) == (key, decision)
    assert not (runtime.base / (key + ".json")).exists()

    runtime.epoch_index += 1
    with pytest.warns(RuntimeWarning, match="worker interrupted"):
        assert runtime.configure(batch) == (key, decision)


@pytest.mark.usefixtures("mock_calibration_gpu")
def test_calibration_rejects_live_rng_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path, effective.EffectiveTrainingRuntime)
    batch = (
        torch.ones(32, 3),
        torch.zeros(32, dtype=torch.long),
        torch.ones(32, dtype=torch.long),
        1.0,
    )
    before = layout.rng(torch.device("cpu"))

    def mutate_rng(*_args: object) -> Never:
        torch.rand(1)
        raise RuntimeError("worker interrupted")

    monkeypatch.setattr(effective, "measure", mutate_rng)
    try:
        with pytest.raises(RuntimeError, match="Calibration changed live"):
            runtime.configure(batch)
        assert not list(runtime.base.glob("*.json"))
    finally:
        layout.restore_rng(before, torch.device("cpu"))


def test_short_update_uses_validated_microbatch_without_calibrating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runtime = make_runtime(tmp_path, effective.EffectiveTrainingRuntime)
    runtime.decisions["validated"] = {"microbatch": 8}
    batch = (
        torch.ones(19, 3),
        torch.zeros(19, dtype=torch.long),
        torch.ones(19, dtype=torch.long),
        1.0,
    )

    def unexpected(*_args: object) -> Never:
        raise AssertionError("short updates must not calibrate")

    monkeypatch.setattr(runtime, "configure", unexpected)
    monkeypatch.setattr(va, "_build_runtime", lambda model, _recipe: model)

    def update(
        model: nn.Module,
        compiled: nn.Module,
        optimizer: torch.optim.Optimizer,
        actual_batch: effective.Batch,
        microbatch: int,
        precision: str,
        *,
        metrics: bool,
    ) -> dict[str, int]:
        assert model is compiled is runtime.model
        assert optimizer is runtime.optimizer
        assert actual_batch is batch
        assert microbatch == 8
        assert precision == "float32"
        assert metrics
        return {"samples": len(actual_batch[1])}

    monkeypatch.setattr(effective, "run_update", update)
    assert runtime.logical_update(batch) == {"samples": 19}
