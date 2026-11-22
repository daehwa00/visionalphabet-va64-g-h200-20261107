from __future__ import annotations

import copy
from contextlib import nullcontext

import pytest
import torch

from lnet.complex_layers import ComplexLinear


@pytest.mark.parametrize("mode", ["cpu", "cuda", "cuda_compile"])
def test_joint_projection_preserves_values_and_gradients(mode) -> None:
    device = "cpu" if mode == "cpu" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(43)
    reference = ComplexLinear(160, 128).to(device)
    optimized = copy.deepcopy(reference)
    optimized._joint_coordinate_projection = True
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    x = torch.randn(2, 2, 7, 7, 160, device=device, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    function = optimized.forward_planes
    if mode == "cuda_compile":
        function = torch.compile(function, fullgraph=True)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with context:
        expected, actual = reference.forward_planes(x), function(y)
    norm_atol = 1e-6
    if device == "cpu":
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        # Changing GEMM N changes cuBLAS BF16 tiling. Coordinate subtraction
        # amplifies elementwise relative errors near zero; bound vector error.
        delta = (actual.float() - expected.float()).norm()
        assert delta <= 0.005 * expected.float().norm() + norm_atol
    probe = torch.randn_like(actual)
    (expected * probe).float().mean().backward()
    (actual * probe).float().mean().backward()
    for a, b in zip((x, *reference.parameters()), (y, *optimized.parameters()), strict=True):
        assert a.grad is not None
        assert b.grad is not None
        left, right = a.grad.float(), b.grad.float()
        assert torch.isfinite(right).all()
        assert (left - right).norm() <= 0.03 * left.norm() + norm_atol
    assert reference.state_dict().keys() == optimized.state_dict().keys()
