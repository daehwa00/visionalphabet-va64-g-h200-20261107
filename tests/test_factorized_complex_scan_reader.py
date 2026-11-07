from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import Tensor
from torch.nn import functional

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader


def _dense_reference(
    module: FactorizedComplexConv2dReader,
    real: Tensor,
    imag: Tensor,
) -> tuple[Tensor, Tensor]:
    if module.input_norm is not None:
        real, imag = module.input_norm(real, imag)
    kernel_real, kernel_imag = module.unit_energy_kernel()
    packed_kernel = torch.cat(
        (
            torch.cat((kernel_real, -kernel_imag), dim=1),
            torch.cat((kernel_imag, kernel_real), dim=1),
        ),
        dim=0,
    )
    packed_input = torch.cat(
        (real.movedim(-1, 1), imag.movedim(-1, 1)),
        dim=1,
    )
    output = functional.conv2d(
        packed_input,
        packed_kernel,
        padding=module.padding,
    )
    if module.match_input_rms:
        input_energy = packed_input.square().sum(dim=1, keepdim=True).div(module.input_modes)
        output_energy = (
            output.reshape(
                output.shape[0],
                2,
                module.output_modes,
                output.shape[2],
                output.shape[3],
            )
            .square()
            .sum(dim=(1, 2), keepdim=False)
            .unsqueeze(1)
            .div(module.output_modes)
        )
        output = output * torch.sqrt(
            (input_energy + module.variance_epsilon)
            / (output_energy + module.variance_epsilon)
        )
    return (
        output[:, : module.output_modes].movedim(1, -1).contiguous(),
        output[:, module.output_modes :].movedim(1, -1).contiguous(),
    )


@pytest.mark.parametrize("normalize_input", [False, True])
@pytest.mark.parametrize("match_input_rms", [False, True])
def test_factorized_reader_matches_synthesized_dense_kernel(
    normalize_input: bool,
    match_input_rms: bool,
) -> None:
    torch.manual_seed(19)
    candidate = FactorizedComplexConv2dReader(
        3,
        4,
        rank=2,
        kernel_size=3,
        normalize_input=normalize_input,
        match_input_rms=match_input_rms,
    ).float()
    reference = deepcopy(candidate)
    real = torch.randn(2, 5, 7, 3, dtype=torch.float, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    candidate_real = real.detach().clone().requires_grad_()
    candidate_imag = imag.detach().clone().requires_grad_()

    expected = _dense_reference(reference, real, imag)
    actual = candidate(candidate_real, candidate_imag)
    torch.testing.assert_close(actual[0], expected[0], rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=1.0e-5, atol=1.0e-6)

    expected_loss = expected[0].square().mean() + expected[1].square().mean()
    actual_loss = actual[0].square().mean() + actual[1].square().mean()
    expected_loss.backward()
    actual_loss.backward()
    assert real.grad is not None
    assert imag.grad is not None
    assert candidate_real.grad is not None
    assert candidate_imag.grad is not None
    torch.testing.assert_close(candidate_real.grad, real.grad, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(candidate_imag.grad, imag.grad, rtol=1.0e-5, atol=1.0e-6)
    candidate_parameters = dict(candidate.named_parameters())
    for name, parameter in reference.named_parameters():
        candidate_parameter = candidate_parameters[name]
        assert parameter.grad is not None
        assert candidate_parameter.grad is not None
        torch.testing.assert_close(
            candidate_parameter.grad,
            parameter.grad,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
