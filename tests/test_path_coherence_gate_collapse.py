from __future__ import annotations

# pyright: reportPrivateUsage=false
import torch

from lnet.pac_phase_gated_transition import (
    PathCoherenceGateCollapse,
    ResidualPathCoherenceGateCollapse,
)
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_path_coherence_imagenet100 as runner
from scripts import run_a2d_r2k3_path_coherence_residual_imagenet100 as residual_runner


def test_coherence_gate_initializes_to_exact_direct_collapse() -> None:
    module = PathCoherenceGateCollapse(7)
    real = torch.randn(2, 3, 2, 4, 7)
    imag = torch.randn_like(real)
    gate = module.gate_values(real, imag)
    torch.testing.assert_close(gate, torch.ones_like(gate), rtol=0, atol=0)


def test_coherence_gate_is_mean_one_and_global_phase_invariant() -> None:
    torch.manual_seed(211)
    module = PathCoherenceGateCollapse(6)
    with torch.no_grad():
        nn_weight = torch.randn_like(module.coherence_weight)
        module.coherence_weight.copy_(nn_weight)
    real = torch.randn(2, 3, 2, 4, 6)
    imag = torch.randn_like(real)
    angle = torch.tensor(0.83)
    cosine, sine = angle.cos(), angle.sin()
    rotated = cosine * real - sine * imag, sine * real + cosine * imag
    gate = module.gate_values(real, imag)
    rotated_gate = module.gate_values(*rotated)
    torch.testing.assert_close(
        gate.mean(dim=-2),
        torch.ones_like(gate.mean(dim=-2)),
    )
    torch.testing.assert_close(rotated_gate, gate, rtol=2.0e-5, atol=2.0e-5)


def test_coherence_collapse_is_contiguous_and_differentiable() -> None:
    torch.manual_seed(223)
    module = PathCoherenceGateCollapse(5)
    with torch.no_grad():
        module.coherence_weight.normal_()
    real = torch.randn(2, 3, 2, 4, 5, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output = module.forward_packed(real, imag)
    assert output[0].shape == (2, 3, 2, 1, 5)
    assert output[1].shape == output[0].shape
    assert output[0].is_contiguous()
    assert output[1].is_contiguous()
    loss = output[0].square().mean() + output[1].square().mean()
    loss.backward()
    assert real.grad is not None
    assert imag.grad is not None
    assert module.coherence_weight.grad is not None
    assert bool(torch.count_nonzero(module.coherence_weight.grad))


def test_coherence_candidate_changes_all_seven_path_collapses() -> None:
    model = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(model)
    assert len(runner._path_collapses(model)) == 7
    assert all(
        type(collapse) is PathCoherenceGateCollapse
        for collapse in runner._path_collapses(model)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_955_364


def test_residual_coherence_starts_near_direct_with_mean_one_gate() -> None:
    torch.manual_seed(227)
    module = ResidualPathCoherenceGateCollapse(6)
    real = torch.randn(2, 3, 2, 4, 6)
    imag = torch.randn_like(real)
    gate = module.gate_values(real, imag)
    torch.testing.assert_close(
        gate.mean(dim=-2),
        torch.ones_like(gate.mean(dim=-2)),
    )
    assert float((gate - 1.0).abs().max().detach()) < 0.1


def test_residual_coherence_has_active_gate_gradients_at_direct_initialization() -> None:
    torch.manual_seed(229)
    module = ResidualPathCoherenceGateCollapse(5)
    real = torch.randn(2, 3, 2, 4, 5, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output = module.forward_packed(real, imag)
    loss = output[0].square().mean() + output[1].square().mean()
    loss.backward()
    assert module.coherence_weight.grad is not None
    assert bool(torch.count_nonzero(module.coherence_weight.grad))
    assert module.correction_scale.grad is not None
    assert bool(torch.count_nonzero(module.correction_scale.grad))
    assert all(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in module.correction.parameters()
    )


def test_residual_coherence_candidate_changes_all_seven_path_collapses() -> None:
    model = residual_runner._build(residual_runner.VARIANT, runtime.model_config())
    residual_runner._assert_model(model)
    assert len(residual_runner._path_collapses(model)) == 7
    assert all(
        type(collapse) is ResidualPathCoherenceGateCollapse
        for collapse in residual_runner._path_collapses(model)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_969_707
