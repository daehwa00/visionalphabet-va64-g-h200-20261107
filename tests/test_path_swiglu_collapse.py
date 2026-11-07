from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from lnet.pac_phase_gated_transition import PathSwiGLUCollapse
from lnet.pac_triton_grouped_path_cffn import (
    d4_grouped_path_swiglu,
    d4_grouped_path_swiglu_reference,
)


def _explicit(
    module: PathSwiGLUCollapse,
    real: torch.Tensor,
    imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = module.value(real, imag)
    packed = torch.cat(
        (real.transpose(-2, -1), imag.transpose(-2, -1)),
        dim=-1,
    )
    gate = functional.silu(
        torch.einsum("...mi,mhi->...mh", packed, module.gate_weight)
    ).transpose(-2, -1)
    return module.output(value[0] * gate, value[1] * gate)


def test_path_swiglu_packed_reference_matches_declared_equation_and_gradients() -> None:
    torch.manual_seed(101)
    module = PathSwiGLUCollapse(7, path_hidden=4).double()
    real = torch.randn(2, 3, 2, 4, 7, dtype=torch.double, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    expected = _explicit(module, real, imag)
    actual = d4_grouped_path_swiglu_reference(
        real,
        imag,
        *module.packed_parameters(),
    )
    loss = actual[0].square().mean() + actual[1].square().mean()
    loss.backward()

    torch.testing.assert_close(actual, expected)
    assert real.grad is not None
    assert bool(torch.count_nonzero(real.grad))
    assert imag.grad is not None
    assert bool(torch.count_nonzero(imag.grad))
    assert all(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in module.parameters()
    )


def test_zero_path_swiglu_gate_is_exactly_zero_output() -> None:
    torch.manual_seed(103)
    module = PathSwiGLUCollapse(5, path_hidden=3)
    with torch.no_grad():
        module.gate_weight.zero_()
    real = torch.randn(2, 3, 2, 4, 5)
    imag = torch.randn_like(real)

    actual = module.forward_packed(real, imag)
    torch.testing.assert_close(actual, (torch.zeros_like(actual[0]), torch.zeros_like(actual[1])))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_path_swiglu_cuda_fusion_matches_reference_forward_and_backward() -> None:
    torch.manual_seed(107)
    reference_module = PathSwiGLUCollapse(8, path_hidden=4).cuda()
    fused_module = PathSwiGLUCollapse(8, path_hidden=4).cuda()
    fused_module.load_state_dict(reference_module.state_dict())
    reference_real = torch.randn(
        2,
        5,
        4,
        4,
        8,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_imag = torch.randn_like(reference_real, requires_grad=True)
    fused_real = reference_real.detach().clone().requires_grad_()
    fused_imag = reference_imag.detach().clone().requires_grad_()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = d4_grouped_path_swiglu_reference(
            reference_real,
            reference_imag,
            *reference_module.packed_parameters(),
        )
        actual = d4_grouped_path_swiglu(
            fused_real,
            fused_imag,
            *fused_module.packed_parameters(),
        )
        expected_loss = expected[0].square().mean() + expected[1].square().mean()
        actual_loss = actual[0].square().mean() + actual[1].square().mean()
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual, expected, rtol=3.0e-2, atol=3.0e-2)
    torch.testing.assert_close(fused_real.grad, reference_real.grad, rtol=8.0e-2, atol=8.0e-2)
    torch.testing.assert_close(fused_imag.grad, reference_imag.grad, rtol=8.0e-2, atol=8.0e-2)
    for fused_parameter, reference_parameter in zip(
        fused_module.parameters(),
        reference_module.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            fused_parameter.grad,
            reference_parameter.grad,
            rtol=1.2e-1,
            atol=1.2e-1,
        )
