from __future__ import annotations

# pyright: reportCallIssue=false
import pytest
import torch

from lnet.pac_phase_gated_transition import (
    DirectMagnitudePathCollapse,
    PathPhaseMagnitudeCollapse,
    PathSelfMagnitudeCollapse,
)


@pytest.mark.parametrize(
    "module",
    [
        PathPhaseMagnitudeCollapse(7, path_hidden=4),
        PathSelfMagnitudeCollapse(7, path_hidden=4),
        DirectMagnitudePathCollapse(7),
    ],
)
def test_mean_one_path_collapses_are_finite_and_differentiable(
    module: torch.nn.Module,
) -> None:
    real = torch.randn(2, 3, 2, 4, 7, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    output = module.forward_packed(real, imag)
    assert output[0].shape == (2, 3, 2, 1, 7)
    assert output[1].shape == output[0].shape
    assert output[0].is_contiguous()
    assert output[1].is_contiguous()
    loss = output[0].square().mean() + output[1].square().mean()
    loss.backward()
    assert real.grad is not None
    assert bool(torch.isfinite(real.grad).all())
    assert imag.grad is not None
    assert bool(torch.isfinite(imag.grad).all())
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    )


@pytest.mark.parametrize(
    "module",
    [
        PathPhaseMagnitudeCollapse(5, path_hidden=4),
        PathSelfMagnitudeCollapse(5, path_hidden=4),
        DirectMagnitudePathCollapse(5),
    ],
)
def test_mean_one_path_gate_cannot_change_average_hidden_gain(
    module: torch.nn.Module,
) -> None:
    real = torch.randn(2, 3, 2, 4, 5)
    imag = torch.randn_like(real)
    if isinstance(module, PathPhaseMagnitudeCollapse):
        joint = module.value_detector(real, imag)
        detector = (
            joint[0][..., module.path_hidden :, :],
            joint[1][..., module.path_hidden :, :],
        )
    elif isinstance(module, PathSelfMagnitudeCollapse):
        detector = module.value(real, imag)
    else:
        detector = real, imag
    gate = module.gate(
        detector[0].transpose(-2, -1),
        detector[1].transpose(-2, -1),
    )
    torch.testing.assert_close(gate.mean(dim=-1), torch.ones_like(gate.mean(dim=-1)))


def test_direct_magnitude_gate_is_invariant_to_global_complex_phase() -> None:
    module = DirectMagnitudePathCollapse(6)
    real = torch.randn(2, 3, 2, 4, 6)
    imag = torch.randn_like(real)
    angle = torch.tensor(0.71)
    cosine, sine = angle.cos(), angle.sin()
    rotated = cosine * real - sine * imag, sine * real + cosine * imag
    gate = module.gate(real.transpose(-2, -1), imag.transpose(-2, -1))
    rotated_gate = module.gate(
        rotated[0].transpose(-2, -1),
        rotated[1].transpose(-2, -1),
    )
    torch.testing.assert_close(rotated_gate, gate, rtol=1.0e-5, atol=1.0e-5)
