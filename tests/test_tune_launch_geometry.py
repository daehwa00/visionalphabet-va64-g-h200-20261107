from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import tune_launch_geometry as tuner

from lnet.pac_kernel_launch_config import LaunchGeometry
from lnet.pac_launch_tuning import TuningResult


def test_runner_variant_uses_the_single_declared_default() -> None:
    runner = SimpleNamespace(VARIANT="default", VARIANTS=("default", "alternate"))

    assert tuner._runner_variant(runner, None) == "default"


def test_runner_variant_requires_an_explicit_multi_variant_choice() -> None:
    runner = SimpleNamespace(VARIANTS=("first", "second"))

    with pytest.raises(ValueError, match="--variant"):
        tuner._runner_variant(runner, None)

    assert tuner._runner_variant(runner, "second") == "second"


def test_runner_variant_rejects_an_unknown_choice() -> None:
    runner = SimpleNamespace(VARIANTS=("first", "second"))

    with pytest.raises(ValueError, match="unsupported runner variant"):
        tuner._runner_variant(runner, "unknown")


def test_compiled_cache_rotation_cleans_each_candidate_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = str(tmp_path / "existing-inductor-cache")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", original)
    resets = 0

    def observed_reset() -> None:
        nonlocal resets
        resets += 1

    monkeypatch.setattr(tuner.torch.compiler, "reset", observed_reset)

    with tuner._compiled_cache_rotation(enabled=True) as rotate:
        rotate()
        first = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        assert first.is_dir()
        rotate()
        second = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        assert second.is_dir()
        assert second != first
        assert not first.exists()

    assert resets == 2
    assert not second.exists()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == original


def test_requested_kernel_results_must_all_be_present() -> None:
    geometry = LaunchGeometry.build(num_warps=4, blocks={"BLOCK_MODES": 16})
    results = [TuningResult("forward", geometry, geometry, 1.0, 1.0)]

    with pytest.raises(RuntimeError, match="backward"):
        tuner._require_requested_results(("forward", "backward"), results)

    tuner._require_requested_results(("forward",), results)
    tuner._require_requested_results(None, results)


def test_compiled_tuning_inputs_match_training_memory_format() -> None:
    inputs = tuner._synthetic_inputs(2, 16, device="cpu", channels_last=True)

    assert inputs.shape == (2, 3, 16, 16)
    assert inputs.is_contiguous(memory_format=tuner.torch.channels_last)


def test_eager_tuning_inputs_remain_contiguous_nchw() -> None:
    inputs = tuner._synthetic_inputs(2, 16, device="cpu", channels_last=False)

    assert inputs.is_contiguous()
    assert not inputs.is_contiguous(memory_format=tuner.torch.channels_last)


def test_step_timing_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        tuner._step_time_ms(lambda: None, iterations=0)


def test_repeated_step_timing_reports_the_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = iter((12.0, 8.0, 10.0))

    def fake_step_time(_step: object, *, iterations: int) -> float:
        del iterations
        return next(samples)

    monkeypatch.setattr(tuner, "_step_time_ms", fake_step_time)

    median, measured = tuner._step_time_samples(
        lambda: None,
        iterations=5,
        repeats=3,
    )

    assert median == 10.0
    assert measured == (12.0, 8.0, 10.0)


def test_repeated_step_timing_requires_a_sample() -> None:
    with pytest.raises(ValueError, match="repeats must be positive"):
        tuner._step_time_samples(lambda: None, iterations=1, repeats=0)


def test_calibration_requires_finite_loss_and_gradients() -> None:
    model = tuner.torch.nn.Linear(3, 2)
    output = model(tuner.torch.ones(1, 3))
    loss = output.square().mean()
    loss.backward()

    assert tuner._require_finite_step(loss, model) == 2

    model.weight.grad = tuner.torch.full_like(model.weight, float("nan"))
    with pytest.raises(RuntimeError, match="non-finite parameter gradients"):
        tuner._require_finite_step(loss, model)
