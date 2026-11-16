"""Fused downsample endpoints versus the existing scan -> Path CFFN boundary."""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_product_scan_pipeline import (
    run_product_scan_coarse_path_collapse_packed,
    run_product_scan_pipeline_packed,
)
from lnet.pac_real2d_math import discrete_pole_real2d
from lnet.pac_triton_grouped_path_cffn import d4_grouped_path_collapse_packed


def _inputs(device: str):
    torch.manual_seed(831)
    modes = 7
    damping = torch.linspace(0.06, 0.3, modes, device=device).view(1, 1, 1, modes)
    phase = torch.linspace(0.05, 1.2, modes, device=device).view(1, 1, 1, modes)
    poles = tuple(
        tuple(
            value.detach().requires_grad_()
            for value in discrete_pole_real2d(damping, phase, 1.0)
        )
        for _ in range(2)
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    source = torch.randn(2, 8, 10, 2 * modes, device=device, dtype=dtype, requires_grad=True)
    module = PathOnlyCollapse(modes, path_hidden=4).to(device)
    parameters = tuple(value.detach().requires_grad_() for value in module.packed_parameters())
    return *poles, source, parameters


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)])
@pytest.mark.parametrize("gain", ["global", "pointwise"])
def test_coarse_path_fusion_forward_and_all_gradients(device: str, gain: str) -> None:
    px, py, source, parameters = _inputs(device)
    coarse, _ = run_product_scan_pipeline_packed(
        px, py, source, epilogue="coarse", gain_normalization=gain, emit_descriptor=False,
    )
    expected = d4_grouped_path_collapse_packed(coarse, *parameters)
    saved_shapes = []
    def save(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        actual = run_product_scan_coarse_path_collapse_packed(
            px, py, source, parameters, gain_normalization=gain,
        )
    assert actual.shape == (2, 4, 5, 14)
    assert actual.is_contiguous()
    assert (2, 4, 5, 4, 14) not in saved_shapes
    atol, rtol = (2e-2, 2e-2) if device == "cuda" else (2e-6, 2e-5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    probe = torch.randn_like(expected)
    inputs = (*px, *py, source, *parameters)
    reference_grad = torch.autograd.grad(expected, inputs, probe, retain_graph=True)
    candidate_grad = torch.autograd.grad(actual, inputs, probe)
    for candidate, reference in zip(candidate_grad, reference_grad, strict=True):
        torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coarse_path_fusion_compiled_forward_and_backward() -> None:
    px, py, source, parameters = _inputs("cuda")
    def forward(*values: Tensor):
        return run_product_scan_coarse_path_collapse_packed(
            tuple(values[:4]), tuple(values[4:8]), values[8], tuple(values[9:]),
            gain_normalization="global",
        )
    values = (*px, *py, source, *parameters)
    expected = forward(*values)
    actual = torch.compile(forward, fullgraph=True, dynamic=False)(*values)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    probe = torch.randn_like(expected)
    reference_grad = torch.autograd.grad(expected, values, probe, retain_graph=True)
    candidate_grad = torch.autograd.grad(actual, values, probe)
    for candidate, reference in zip(candidate_grad, reference_grad, strict=True):
        torch.testing.assert_close(candidate, reference, atol=2e-3, rtol=2e-3)
