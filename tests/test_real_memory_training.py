"""Training candidates must preserve gradients and optimizer resume structure."""

import copy

import pytest
import run_va128_real_memory_imagenet100 as runner
import torch

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader
from lnet.pac_triton_real_spatial import real_factorized_spatial
from lnet.pac_triton_real_training import real_training_spatial
from lnet.real_memory_backbone import RealFactorizedMemoryReader
from lnet.real_memory_training import enable_real_memory_training_


def check_gradient(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    else:
        a, b = actual.float(), expected.float()
        d = a - b
        assert d.norm() / b.norm().clamp_min(1e-12) <= 0.02
        assert d.abs().max() <= 0.03 * b.abs().max() + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vector", [False, True])
def test_spatial_forward_backward(dtype: torch.dtype, vector: bool) -> None:
    torch.manual_seed(81)
    x = torch.randn(2, 7, 5, 20, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(5, 2, 4, 3, 3, device="cuda", dtype=dtype, requires_grad=True)
    probe = torch.randn(2, 7, 5, 10, device="cuda", dtype=dtype)
    expected = real_factorized_spatial(x, w)
    dx, dw = torch.autograd.grad(expected, (x, w), probe)
    actual = real_training_spatial(x, w, vector)
    ax, aw = torch.autograd.grad(actual, (x, w), probe)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    check_gradient(ax, dx, dtype)
    check_gradient(aw, dw, dtype)


def test_training_enable_preserves_parameters_rng_and_optimizer_resume() -> None:

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(83)
        model = runner._build(runner.VARIANT, runner.runtime.model_config())
        recipe = {
            "learning_rate": 0.003,
            "weight_decay": 0.05,
            "modal_learning_rate_multiplier": 1 / 3,
            "pole_geometry_learning_rate_multiplier": 1.0,
            "fused_optimizer": False,
        }
        optimizer = runner.runtime.build_optimizer(model, recipe)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        parameters = list(model.parameters())
        groups = [list(g["params"]) for g in optimizer.param_groups]
        rng = torch.get_rng_state()
        report = enable_real_memory_training_(model, spatial=True, vector_weight_gradient=True)
        assert report["readers"] == 12
        assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
        for expected, actual in zip(groups, optimizer.param_groups, strict=True):
            assert all(a is b for a, b in zip(expected, actual["params"], strict=True))
        assert torch.equal(torch.get_rng_state(), rng)
        assert before.keys() == model.state_dict().keys()
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key])
        resumed = runner._build(runner.VARIANT, runner.runtime.model_config())
        resumed.load_state_dict(model.state_dict(), strict=True)
        resumed_optimizer = runner.runtime.build_optimizer(resumed, recipe)
        resumed_optimizer.load_state_dict(saved_optimizer)
        for current in (model, resumed):
            for parameter in current.parameters():
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        resumed_optimizer.step()
        for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_opt_in_is_explicit(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:

    reader = RealFactorizedMemoryReader(FactorizedComplexConv2dReader(4, 5, rank=2))
    reader.execution_mode = "factorized_triton"
    model = torch.nn.Sequential(reader)
    parameters = list(model.parameters())
    monkeypatch.setattr(runner.prepare, "_prepare_model", lambda model, _recipe: model)
    monkeypatch.setenv("VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS", "1" if enabled else "0")
    assert runner._prepare_model(model, {}) is model
    assert getattr(reader, "_real_training_autotune", False) is enabled
    assert getattr(reader, "_real_training_vector_gradient", False) is enabled
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))


def test_runner_rejects_invalid_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setenv("VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        runner._training_optimizations_enabled()
