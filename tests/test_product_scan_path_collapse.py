from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest
import torch
from torch import Tensor

from lnet.pac_phase_gated_transition import PathOnlyCollapse, PathSwiGLUCollapse
from lnet.pac_product_scan_pipeline import (
    ScanMemoryPolicy,
    run_product_scan_path_collapse_pipeline,
    run_product_scan_pipeline,
)
from lnet.pac_product_scan_reference import product_scan_full4_reference
from lnet.pac_real2d_math import discrete_pole_real2d
from lnet.pac_triton_grouped_path_cffn import d4_grouped_cell_path_swiglu_reference

Pole = tuple[Tensor, Tensor, Tensor, Tensor]
ComplexField = tuple[Tensor, Tensor]


def _pole(modes: int, phase_scale: float, device: torch.device) -> Pole:
    damping = torch.linspace(0.04, 0.35, modes, device=device).view(1, 1, 1, modes)
    phase = torch.linspace(0.0, phase_scale * torch.pi, modes, device=device).view(
        1,
        1,
        1,
        modes,
    )
    return cast(
        "Pole",
        tuple(
            value.detach().requires_grad_() for value in discrete_pole_real2d(damping, phase, 1.0)
        ),
    )


def _clone_pole(values: Pole) -> Pole:
    return cast("Pole", tuple(value.detach().clone().requires_grad_() for value in values))


def _clone_field(values: ComplexField) -> ComplexField:
    return cast(
        "ComplexField",
        tuple(value.detach().clone().requires_grad_() for value in values),
    )


def _reference(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
    collapse: PathOnlyCollapse,
) -> tuple[Tensor, Tensor]:
    full = run_product_scan_pipeline(
        pole_x,
        pole_y,
        source,
        epilogue="full16",
        gain_normalization="global",
        memory_policy="recompute",
    )
    if not isinstance(full, tuple):
        raise TypeError("full16 scan did not return directional cells")
    return collapse.forward_cells(full[0], full[1])


def _candidate(
    pole_x: Pole,
    pole_y: Pole,
    source: ComplexField,
    collapse: PathOnlyCollapse,
    *,
    memory_policy: ScanMemoryPolicy = "recompute",
) -> tuple[Tensor, Tensor]:
    return run_product_scan_path_collapse_pipeline(
        pole_x,
        pole_y,
        source,
        collapse.packed_parameters(),
        gain_normalization="global",
        memory_policy=memory_policy,
    )


def _case(
    device: torch.device,
    dtype: torch.dtype,
    *,
    height: int = 6,
    width: int = 8,
) -> tuple[PathOnlyCollapse, PathOnlyCollapse, Pole, Pole, ComplexField]:
    torch.manual_seed(67)
    modes = 8
    reference_collapse = PathOnlyCollapse(modes, path_hidden=8).to(device)
    candidate_collapse = deepcopy(reference_collapse)
    pole_x = _pole(modes, 0.7, device)
    pole_y = _pole(modes, 0.6, device)
    source = cast(
        "ComplexField",
        tuple(
            torch.randn(2, height, width, modes, device=device, dtype=dtype, requires_grad=True)
            for _ in range(2)
        ),
    )
    return reference_collapse, candidate_collapse, pole_x, pole_y, source


def test_scan_path_collapse_cpu_preserves_exact_outputs_and_gradients() -> None:
    reference_collapse, candidate_collapse, pole_x, pole_y, source = _case(
        torch.device("cpu"),
        torch.float32,
    )
    reference_pole_x = _clone_pole(pole_x)
    reference_pole_y = _clone_pole(pole_y)
    reference_source = _clone_field(source)
    candidate_pole_x = _clone_pole(pole_x)
    candidate_pole_y = _clone_pole(pole_y)
    candidate_source = _clone_field(source)
    expected = _reference(
        reference_pole_x,
        reference_pole_y,
        reference_source,
        reference_collapse,
    )
    actual = _candidate(
        candidate_pole_x,
        candidate_pole_y,
        candidate_source,
        candidate_collapse,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_odd_scan_path_collapse_cpu_matches_full_resolution_reference() -> None:
    reference_collapse, candidate_collapse, pole_x, pole_y, source = _case(
        torch.device("cpu"),
        torch.float32,
        height=7,
        width=7,
    )
    reference_pole_x = _clone_pole(pole_x)
    reference_pole_y = _clone_pole(pole_y)
    reference_source = _clone_field(source)
    candidate_pole_x = _clone_pole(pole_x)
    candidate_pole_y = _clone_pole(pole_y)
    candidate_source = _clone_field(source)
    full = product_scan_full4_reference(
        reference_pole_x,
        reference_pole_y,
        reference_source,
        gain_normalization="global",
    )
    expected = reference_collapse.forward_packed(*full)
    actual = _candidate(
        candidate_pole_x,
        candidate_pole_y,
        candidate_source,
        candidate_collapse,
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("memory_policy", ["recompute", "retain"])
@pytest.mark.parametrize(("height", "width"), [(6, 8), (7, 7)])
def test_scan_path_collapse_cuda_preserves_training_contract(
    height: int,
    width: int,
    memory_policy: ScanMemoryPolicy,
) -> None:
    reference_collapse, candidate_collapse, pole_x, pole_y, source = _case(
        torch.device("cuda"),
        torch.bfloat16,
        height=height,
        width=width,
    )
    warm_pole_x = _clone_pole(pole_x)
    warm_pole_y = _clone_pole(pole_y)
    warm_source = _clone_field(source)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if height % 2 == 0 and width % 2 == 0:
            warm_output = _reference(
                warm_pole_x,
                warm_pole_y,
                warm_source,
                reference_collapse,
            )
        else:
            warm_full = product_scan_full4_reference(
                warm_pole_x,
                warm_pole_y,
                warm_source,
                gain_normalization="global",
            )
            warm_output = reference_collapse.forward_packed(*warm_full)
    torch.autograd.grad(
        warm_output,
        (*warm_pole_x, *warm_pole_y, *warm_source, *reference_collapse.parameters()),
        tuple(torch.ones_like(value) for value in warm_output),
    )
    reference_pole_x = _clone_pole(pole_x)
    reference_pole_y = _clone_pole(pole_y)
    reference_source = _clone_field(source)
    candidate_pole_x = _clone_pole(pole_x)
    candidate_pole_y = _clone_pole(pole_y)
    candidate_source = _clone_field(source)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if height % 2 == 0 and width % 2 == 0:
            expected = _reference(
                reference_pole_x,
                reference_pole_y,
                reference_source,
                reference_collapse,
            )
        else:
            full = product_scan_full4_reference(
                reference_pole_x,
                reference_pole_y,
                reference_source,
                gain_normalization="global",
            )
            expected = reference_collapse.forward_packed(*full)
        actual = _candidate(
            candidate_pole_x,
            candidate_pole_y,
            candidate_source,
            candidate_collapse,
            memory_policy=memory_policy,
        )
    cotangent = tuple(torch.randn_like(value) for value in actual)
    reference_tensors = (
        *reference_pole_x,
        *reference_pole_y,
        *reference_source,
        *reference_collapse.parameters(),
    )
    candidate_tensors = (
        *candidate_pole_x,
        *candidate_pole_y,
        *candidate_source,
        *candidate_collapse.parameters(),
    )
    expected_gradients = torch.autograd.grad(expected, reference_tensors, cotangent)
    actual_gradients = torch.autograd.grad(actual, candidate_tensors, cotangent)
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=2.0e-2, atol=2.0e-2)
    gradient_names = (
        "pole_x_decay_real",
        "pole_x_decay_imag",
        "pole_x_gamma_real",
        "pole_x_gamma_imag",
        "pole_y_decay_real",
        "pole_y_decay_imag",
        "pole_y_gamma_real",
        "pole_y_gamma_imag",
        "source_real",
        "source_imag",
        *(name for name, _parameter in candidate_collapse.named_parameters()),
    )
    relative_l2 = {}
    for name, value, target in zip(
        gradient_names,
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        difference = value.float() - target.float()
        denominator = target.float().square().sum().sqrt().clamp_min(1.0e-12)
        relative_l2[name] = float(difference.square().sum().sqrt() / denominator)
    assert max(relative_l2.values()) <= 2.0e-2, relative_l2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scan_path_swiglu_cuda_preserves_outputs_and_gradients() -> None:
    torch.manual_seed(71)
    device = torch.device("cuda")
    modes = 8
    reference_collapse = PathSwiGLUCollapse(modes, path_hidden=2).to(device)
    candidate_collapse = deepcopy(reference_collapse)
    pole_x = _pole(modes, 0.7, device)
    pole_y = _pole(modes, 0.6, device)
    source = cast(
        "ComplexField",
        tuple(
            torch.randn(2, 6, 8, modes, device=device, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(2)
        ),
    )
    reference_pole_x = _clone_pole(pole_x)
    reference_pole_y = _clone_pole(pole_y)
    reference_source = _clone_field(source)
    candidate_pole_x = _clone_pole(pole_x)
    candidate_pole_y = _clone_pole(pole_y)
    candidate_source = _clone_field(source)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = run_product_scan_pipeline(
            reference_pole_x,
            reference_pole_y,
            reference_source,
            epilogue="full16",
            gain_normalization="global",
            memory_policy="recompute",
        )
        if not isinstance(full, tuple):
            raise TypeError("full16 scan did not return directional cells")
        expected = d4_grouped_cell_path_swiglu_reference(
            full[0],
            full[1],
            *reference_collapse.packed_parameters(),
        )
        actual = run_product_scan_path_collapse_pipeline(
            candidate_pole_x,
            candidate_pole_y,
            candidate_source,
            candidate_collapse.packed_parameters(),
            gain_normalization="global",
            memory_policy="recompute",
            path_swiglu=True,
        )

    cotangent = tuple(torch.randn_like(value) for value in actual)
    reference_tensors = (
        *reference_pole_x,
        *reference_pole_y,
        *reference_source,
        *reference_collapse.parameters(),
    )
    candidate_tensors = (
        *candidate_pole_x,
        *candidate_pole_y,
        *candidate_source,
        *candidate_collapse.parameters(),
    )
    expected_gradients = torch.autograd.grad(expected, reference_tensors, cotangent)
    actual_gradients = torch.autograd.grad(actual, candidate_tensors, cotangent)
    for value, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, target, rtol=2.0e-2, atol=2.0e-2)
    for value, target in zip(actual_gradients, expected_gradients, strict=True):
        difference = value.float() - target.float()
        denominator = target.float().square().sum().sqrt().clamp_min(1.0e-12)
        assert float(difference.square().sum().sqrt() / denominator) <= 2.0e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scan_path_collapse_is_compiled_training_compatible() -> None:
    collapse = PathOnlyCollapse(8, path_hidden=8).cuda().train()
    pole_x = _pole(8, 0.7, torch.device("cuda"))
    pole_y = _pole(8, 0.6, torch.device("cuda"))

    def forward(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
        return run_product_scan_path_collapse_pipeline(
            pole_x,
            pole_y,
            (real, imag),
            collapse.packed_parameters(),
            gain_normalization="global",
            memory_policy="recompute",
        )

    compiled = torch.compile(forward, mode="default", fullgraph=False, dynamic=False)
    real = torch.randn(2, 7, 7, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = compiled(real, imag)
    (output[0].float().square().mean() + output[1].float().square().mean()).backward()

    assert real.grad is not None
    assert imag.grad is not None
    assert all(parameter.grad is not None for parameter in collapse.parameters())
