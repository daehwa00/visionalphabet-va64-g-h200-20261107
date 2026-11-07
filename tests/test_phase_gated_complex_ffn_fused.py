# pyright: reportMissingImports=false, reportPrivateLocalImportUsage=false
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.nn import functional

import lnet.pac_phase_gated_cffn as phase_gated
import lnet.pac_triton_complex_rmsnorm as triton_rmsnorm
import lnet.pac_triton_phase_gate_linear as triton_phase_gate_linear
from lnet.pac_phase_gated_cffn import (
    PhaseGatedComplexFFN,
    phase_gated_complex_ffn_reference,
)
from lnet.pac_triton_complex_rmsnorm import (
    packed_complex_rms_norm,
    packed_complex_rms_norm_reference,
    supports_packed_complex_rms_norm,
)
from lnet.pac_triton_phase_gate import phase_gate_reference
from lnet.pac_triton_phase_gate_linear import (
    phase_gate_output_linear,
    supports_phase_gate_output_linear,
)
from lnet.pac_triton_phase_gate_linear_fused import supports_fused_phase_gate_output_linear
from lnet.pac_triton_rmsnorm_linear_fused import (
    fused_rmsnorm_input_linear,
    supports_fused_rmsnorm_input_linear,
)

_CUDA_LIBRARY_PATHS = [Path("/lib/x86_64-linux-gnu"), Path("/usr/lib/x86_64-linux-gnu")]
if configured_cuda_path := os.environ.get("TRITON_LIBCUDA_PATH"):
    _CUDA_LIBRARY_PATHS.append(Path(configured_cuda_path))
_LINKABLE_CUDA = any((directory / "libcuda.so").is_file() for directory in _CUDA_LIBRARY_PATHS)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _LINKABLE_CUDA,
    reason="requires a linkable CUDA driver",
)


def _loss(outputs: tuple[Tensor, Tensor]) -> Tensor:
    real, imag = outputs
    return real.float().square().mean() + 0.7 * imag.float().square().mean()


def _relative_l2(actual: Tensor, expected: Tensor) -> float:
    difference = (actual.float() - expected.float()).detach().norm()
    denominator = expected.float().detach().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return (difference / denominator).item()


def _run(
    module: PhaseGatedComplexFFN,
    real: Tensor,
    imag: Tensor,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, ...]]:
    parameters = tuple(module.parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = module(real, imag)
    gradients = torch.autograd.grad(_loss(outputs), (real, imag, *parameters))
    return outputs, gradients


def _run_reference(
    module: PhaseGatedComplexFFN,
    real: Tensor,
    imag: Tensor,
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, ...]]:
    parameters = tuple(module.parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = phase_gated_complex_ffn_reference(module, real, imag)
    gradients = torch.autograd.grad(_loss(outputs), (real, imag, *parameters))
    return outputs, gradients


@pytest.mark.parametrize(
    ("rows", "modes", "hidden", "self_gated"),
    [
        (513, 4, 8, False),
        (257, 7, 11, False),
        (129, 16, 24, True),
        (129, 96, 96, False),
        (65, 96, 144, False),
        (33, 96, 192, False),
    ],
)
def test_cuda_bf16_phase_gated_path_matches_reference(
    rows: int,
    modes: int,
    hidden: int,
    self_gated: bool,
) -> None:
    torch.manual_seed(53)
    fused = PhaseGatedComplexFFN(modes, hidden, self_gated=self_gated).cuda()
    reference = deepcopy(fused)
    fused_real = torch.randn(rows, modes, device="cuda", dtype=torch.bfloat16).requires_grad_()
    fused_imag = torch.randn(rows, modes, device="cuda", dtype=torch.bfloat16).requires_grad_()
    reference_real = fused_real.detach().clone().requires_grad_()
    reference_imag = fused_imag.detach().clone().requires_grad_()

    actual, actual_gradients = _run(fused, fused_real, fused_imag)
    expected, expected_gradients = _run_reference(reference, reference_real, reference_imag)

    torch.testing.assert_close(actual, expected, rtol=3.0e-2, atol=3.0e-2)
    for actual_gradient, expected_gradient in zip(
        actual_gradients[:2],
        expected_gradients[:2],
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=5.0e-2,
            atol=5.0e-2,
        )


def test_cuda_uses_optimized_path_without_caller_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"norm": 0, "gate_output": 0}
    original_norm = phase_gated.packed_complex_rms_norm
    original_gate_output = phase_gated.phase_gate_output_linear

    def tracked_norm(
        real: Tensor,
        imag: Tensor,
        norm_weight: Tensor,
        epsilon: float,
    ) -> Tensor:
        calls["norm"] += 1
        return original_norm(real, imag, norm_weight, epsilon)

    def tracked_gate_output(
        projected: Tensor,
        alpha: Tensor,
        output_weight: Tensor,
        *,
        redistribution: float,
        self_gated: bool,
    ) -> Tensor:
        calls["gate_output"] += 1
        return original_gate_output(
            projected,
            alpha,
            output_weight,
            redistribution=redistribution,
            self_gated=self_gated,
        )

    monkeypatch.setattr(phase_gated, "packed_complex_rms_norm", tracked_norm)
    monkeypatch.setattr(phase_gated, "phase_gate_output_linear", tracked_gate_output)
    module = PhaseGatedComplexFFN(16, 24).cuda()
    real = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
    imag = torch.randn_like(real)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        module(real, imag)

    assert calls == {"norm": 1, "gate_output": 1}


def test_cuda_phase_gated_path_reads_transposed_dense_input_without_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(57)
    fused = PhaseGatedComplexFFN(4, 8).cuda()
    reference = deepcopy(fused)
    fused_real = (
        torch.randn(3, 4, 95, device="cuda", dtype=torch.bfloat16)
        .transpose(-2, -1)
        .requires_grad_()
    )
    fused_imag = (
        torch.randn(3, 4, 95, device="cuda", dtype=torch.bfloat16)
        .transpose(-2, -1)
        .requires_grad_()
    )
    reference_real = fused_real.detach().clone().requires_grad_()
    reference_imag = fused_imag.detach().clone().requires_grad_()
    observed_layouts: list[tuple[bool, tuple[int, ...]]] = []
    original = phase_gated.fused_phase_gated_cffn

    def tracked(
        real: Tensor,
        imag: Tensor,
        norm_weight: Tensor,
        input_weight: Tensor,
        alpha: Tensor,
        output_weight: Tensor,
        gamma: Tensor,
        *,
        epsilon: float,
        redistribution: float,
        self_gated: bool,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        observed_layouts.append((real.is_contiguous(), real.stride()))
        return original(
            real,
            imag,
            norm_weight,
            input_weight,
            alpha,
            output_weight,
            gamma,
            epsilon=epsilon,
            redistribution=redistribution,
            self_gated=self_gated,
            collect_diagnostics=collect_diagnostics,
        )

    monkeypatch.setattr(phase_gated, "fused_phase_gated_cffn", tracked)
    actual, actual_gradients = _run(fused, fused_real, fused_imag)
    expected, expected_gradients = _run_reference(
        reference,
        reference_real,
        reference_imag,
    )

    assert observed_layouts == [(False, fused_real.stride())]
    torch.testing.assert_close(actual, expected, rtol=3.0e-2, atol=3.0e-2)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=8.0e-2,
            atol=5.0e-2,
        )


def test_cuda_narrow_phase_gated_path_fuses_complete_ffn_without_caller_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    diagnostic_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original = phase_gated.fused_phase_gated_cffn

    def tracked(
        real: Tensor,
        imag: Tensor,
        norm_weight: Tensor,
        input_weight: Tensor,
        alpha: Tensor,
        output_weight: Tensor,
        gamma: Tensor,
        *,
        epsilon: float,
        redistribution: float,
        self_gated: bool,
        collect_diagnostics: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        nonlocal calls
        calls += 1
        outputs = original(
            real,
            imag,
            norm_weight,
            input_weight,
            alpha,
            output_weight,
            gamma,
            epsilon=epsilon,
            redistribution=redistribution,
            self_gated=self_gated,
            collect_diagnostics=collect_diagnostics,
        )
        diagnostic_shapes.append((tuple(outputs[2].shape), tuple(outputs[3].shape)))
        return outputs

    monkeypatch.setattr(phase_gated, "fused_phase_gated_cffn", tracked)
    module = PhaseGatedComplexFFN(4, 8).cuda()
    real = torch.randn(12, 4, 96, device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    imag = torch.randn_like(real)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = module(real, imag)

    assert calls == 1
    assert diagnostic_shapes == [((1024, 32), (1024, 8))]
    assert output_real.stride() == real.stride()
    assert output_imag.stride() == imag.stride()


def test_cuda_uses_fused_backward_without_caller_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"norm": 0, "gate": 0}
    original_norm = triton_rmsnorm._backward_op  # pyright: ignore[reportPrivateUsage]
    original_gate = triton_phase_gate_linear._backward_op  # pyright: ignore[reportPrivateUsage]

    def tracked_norm(
        real: Tensor,
        imag: Tensor,
        norm_weight: Tensor,
        row_inverse_rms: Tensor,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        calls["norm"] += 1
        return original_norm(real, imag, norm_weight, row_inverse_rms, grad_output)

    def tracked_gate(
        projected: Tensor,
        alpha: Tensor,
        grad_hidden: Tensor,
        redistribution: float,
        self_gated: bool,
    ) -> tuple[Tensor, Tensor]:
        calls["gate"] += 1
        return original_gate(
            projected,
            alpha,
            grad_hidden,
            redistribution,
            self_gated,
        )

    monkeypatch.setattr(triton_rmsnorm, "_backward_op", tracked_norm)
    monkeypatch.setattr(triton_phase_gate_linear, "_backward_op", tracked_gate)
    module = PhaseGatedComplexFFN(16, 24).cuda()
    real = torch.randn(257, 16, device="cuda", requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = module(real, imag)
    _loss(outputs).backward()

    assert calls == {"norm": 1, "gate": 1}


def test_cuda_fp32_autocast_uses_packed_path_without_changing_residual_dtype() -> None:
    torch.manual_seed(61)
    fused = PhaseGatedComplexFFN(8, 12).cuda()
    reference = deepcopy(fused)
    real = torch.randn(37, 8, device="cuda", requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    reference_real = real.detach().clone().requires_grad_()
    reference_imag = imag.detach().clone().requires_grad_()
    actual, actual_gradients = _run(fused, real, imag)
    expected, expected_gradients = _run_reference(
        reference,
        reference_real,
        reference_imag,
    )

    assert actual[0].dtype is actual[1].dtype is torch.float32
    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=5.0e-2,
            atol=5.0e-2,
        )


def test_cuda_optimized_contract_does_not_silently_bypass_component_hooks() -> None:
    module = PhaseGatedComplexFFN(8, 12).cuda()
    handle = module.norm.register_forward_hook(lambda _module, _inputs, _output: None)
    real = torch.randn(17, 8, device="cuda", dtype=torch.bfloat16)
    imag = torch.randn_like(real)
    try:
        with pytest.raises(RuntimeError, match="hook-free"):
            module(real, imag)
    finally:
        handle.remove()


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_packed_rmsnorm_matches_internal_equation_with_arbitrary_cotangent(
    source_dtype: torch.dtype,
) -> None:
    torch.manual_seed(67)
    real = torch.randn(41, 19, device="cuda", dtype=source_dtype, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    norm_weight = (1.0 + 0.05 * torch.randn(19, device="cuda")).requires_grad_()
    expected_real = real.detach().clone().requires_grad_()
    expected_imag = imag.detach().clone().requires_grad_()
    expected_norm_weight = norm_weight.detach().clone().requires_grad_()
    cotangent = torch.randn(41, 38, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert supports_packed_complex_rms_norm(real, imag, norm_weight)
        actual = packed_complex_rms_norm(real, imag, norm_weight, 1.0e-6)
        expected = packed_complex_rms_norm_reference(
            expected_real,
            expected_imag,
            expected_norm_weight,
            1.0e-6,
        ).to(torch.bfloat16)
    actual_gradients = torch.autograd.grad(actual, (real, imag, norm_weight), cotangent)
    expected_gradients = torch.autograd.grad(
        expected,
        (expected_real, expected_imag, expected_norm_weight),
        cotangent,
    )

    torch.testing.assert_close(actual, expected, rtol=8.0e-3, atol=8.0e-3)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2.0e-2,
            atol=2.0e-2,
        )


def test_packed_rmsnorm_matches_transposed_dense_coordinates() -> None:
    torch.manual_seed(71)
    real = (
        torch.randn(7, 19, 41, device="cuda", dtype=torch.bfloat16)
        .transpose(-2, -1)
        .requires_grad_()
    )
    imag = (
        torch.randn(7, 19, 41, device="cuda", dtype=torch.bfloat16)
        .transpose(-2, -1)
        .requires_grad_()
    )
    norm_weight = (1.0 + 0.05 * torch.randn(19, device="cuda")).requires_grad_()
    expected_real = real.detach().clone().requires_grad_()
    expected_imag = imag.detach().clone().requires_grad_()
    expected_norm_weight = norm_weight.detach().clone().requires_grad_()
    cotangent = torch.randn(7, 41, 38, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert not real.is_contiguous()
        assert real.transpose(-2, -1).is_contiguous()
        assert supports_packed_complex_rms_norm(real, imag, norm_weight)
        actual = packed_complex_rms_norm(real, imag, norm_weight, 1.0e-6)
        expected = packed_complex_rms_norm_reference(
            expected_real,
            expected_imag,
            expected_norm_weight,
            1.0e-6,
        ).to(torch.bfloat16)
    actual_gradients = torch.autograd.grad(actual, (real, imag, norm_weight), cotangent)
    expected_gradients = torch.autograd.grad(
        expected,
        (expected_real, expected_imag, expected_norm_weight),
        cotangent,
    )

    torch.testing.assert_close(actual, expected, rtol=8.0e-3, atol=8.0e-3)
    for actual_gradient, expected_gradient in zip(
        actual_gradients[:2],
        expected_gradients[:2],
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2.0e-2,
            atol=2.0e-2,
        )
    assert _relative_l2(actual_gradients[2], expected_gradients[2]) < 1.0e-2


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_fused_rmsnorm_input_projection_matches_internal_equation(
    source_dtype: torch.dtype,
) -> None:
    torch.manual_seed(72)
    rows = 67
    modes = 4
    output_width = 32
    real = torch.randn(rows, modes, device="cuda", dtype=source_dtype, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    norm_weight = (1.0 + 0.05 * torch.randn(modes, device="cuda")).requires_grad_()
    input_weight = torch.randn(
        output_width,
        2 * modes,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    expected_real = real.detach().clone().requires_grad_()
    expected_imag = imag.detach().clone().requires_grad_()
    expected_norm_weight = norm_weight.detach().clone().requires_grad_()
    expected_weight = input_weight.detach().clone().requires_grad_()
    cotangent = torch.randn(rows, output_width, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert supports_fused_rmsnorm_input_linear(
            real,
            imag,
            norm_weight,
            input_weight,
            epsilon=1.0e-6,
        )
        actual = fused_rmsnorm_input_linear(
            real,
            imag,
            norm_weight,
            input_weight,
            epsilon=1.0e-6,
        )
        expected_normalized = packed_complex_rms_norm_reference(
            expected_real,
            expected_imag,
            expected_norm_weight,
            1.0e-6,
        ).to(torch.bfloat16)
        expected = functional.linear(expected_normalized, expected_weight)
    actual_gradients = torch.autograd.grad(
        actual,
        (real, imag, norm_weight, input_weight),
        cotangent,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (expected_real, expected_imag, expected_norm_weight, expected_weight),
        cotangent,
    )

    assert _relative_l2(actual, expected) < 2.0e-2
    gradient_errors = [
        _relative_l2(actual_gradient, expected_gradient)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        )
    ]
    assert all(error < 2.0e-2 for error in gradient_errors), gradient_errors


@pytest.mark.parametrize("self_gated", [False, True])
@pytest.mark.parametrize(("hidden", "output_width"), [(24, 38), (8, 8)])
def test_phase_gate_output_linear_matches_internal_equation_with_arbitrary_cotangent(
    hidden: int,
    output_width: int,
    self_gated: bool,
) -> None:
    torch.manual_seed(73)
    rows = 43
    projected_width = 2 * hidden if self_gated else 4 * hidden
    projected = torch.randn(
        rows,
        projected_width,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    alpha = (0.1 * torch.randn(hidden, device="cuda")).requires_grad_()
    output_weight = torch.randn(
        output_width,
        2 * hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    expected_projected = projected.detach().clone().requires_grad_()
    expected_alpha = alpha.detach().clone().requires_grad_()
    expected_weight = output_weight.detach().clone().requires_grad_()
    cotangent = torch.randn(
        rows,
        output_width,
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual = phase_gate_output_linear(
        projected,
        alpha,
        output_weight,
        redistribution=0.5,
        self_gated=self_gated,
    )
    expected_hidden = phase_gate_reference(
        expected_projected,
        expected_alpha,
        redistribution=0.5,
        self_gated=self_gated,
    )
    expected = functional.linear(expected_hidden, expected_weight)
    actual_gradients = torch.autograd.grad(
        actual,
        (projected, alpha, output_weight),
        cotangent,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (expected_projected, expected_alpha, expected_weight),
        cotangent,
    )

    assert supports_phase_gate_output_linear(
        projected,
        alpha,
        output_weight,
        redistribution=0.5,
        self_gated=self_gated,
    )
    assert supports_fused_phase_gate_output_linear(
        projected,
        alpha,
        output_weight,
        redistribution=0.5,
        self_gated=self_gated,
    ) is (hidden == 8)
    tolerance = 2.0e-2 if hidden == 8 else 5.0e-3
    assert _relative_l2(actual, expected) < tolerance
    gradient_errors = [
        _relative_l2(actual_gradient, expected_gradient)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        )
    ]
    assert all(error < tolerance for error in gradient_errors), gradient_errors


def test_phase_gate_output_linear_recomputes_hidden_in_backward() -> None:
    rows = 43
    hidden = 24
    projected = torch.randn(
        rows,
        4 * hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    alpha = torch.full((hidden,), 0.075, device="cuda", requires_grad=True)
    output_weight = torch.randn(
        38,
        2 * hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    saved_shapes: list[tuple[int, ...]] = []

    def record(value: Tensor) -> Tensor:
        saved_shapes.append(tuple(value.shape))
        return value

    with torch.autograd.graph.saved_tensors_hooks(record, lambda value: value):
        output = phase_gate_output_linear(
            projected,
            alpha,
            output_weight,
            redistribution=0.5,
            self_gated=False,
        )

    assert tuple(projected.shape) in saved_shapes
    assert tuple(alpha.shape) in saved_shapes
    assert tuple(output_weight.shape) in saved_shapes
    assert (rows, 2 * hidden) not in saved_shapes
    output.float().square().mean().backward()


def test_complete_narrow_fusion_does_not_save_full_projected_rows() -> None:
    rows = 2049
    modes = 4
    hidden = 8
    module = PhaseGatedComplexFFN(modes, hidden).cuda()
    real = torch.randn(
        rows,
        modes,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    imag = torch.randn_like(real, requires_grad=True)
    saved_shapes: list[tuple[int, ...]] = []

    def record(value: Tensor) -> Tensor:
        saved_shapes.append(tuple(value.shape))
        return value

    with (
        torch.autograd.graph.saved_tensors_hooks(record, lambda value: value),
        torch.autocast("cuda", dtype=torch.bfloat16),
    ):
        outputs = module(real, imag)

    assert (rows, 4 * hidden) not in saved_shapes
    assert (rows, 2 * modes) not in saved_shapes
    _loss(outputs).backward()


def test_large_rmsnorm_backward_reduction_matches_reference() -> None:
    torch.manual_seed(79)
    rows = 8193
    modes = 96
    real = torch.randn(rows, modes, device="cuda", requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)
    norm_weight = (1.0 + 0.05 * torch.randn(modes, device="cuda")).requires_grad_()
    expected_real = real.detach().clone().requires_grad_()
    expected_imag = imag.detach().clone().requires_grad_()
    expected_norm_weight = norm_weight.detach().clone().requires_grad_()
    norm_cotangent = torch.randn(
        rows,
        2 * modes,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        normalized = packed_complex_rms_norm(real, imag, norm_weight, 1.0e-6)
        expected_normalized = packed_complex_rms_norm_reference(
            expected_real,
            expected_imag,
            expected_norm_weight,
            1.0e-6,
        ).to(torch.bfloat16)
    gradients = torch.autograd.grad(
        normalized,
        (real, imag, norm_weight),
        norm_cotangent,
    )
    expected_gradients = torch.autograd.grad(
        expected_normalized,
        (expected_real, expected_imag, expected_norm_weight),
        norm_cotangent,
    )

    assert _relative_l2(normalized, expected_normalized) < 1.0e-4
    assert all(
        _relative_l2(actual, expected) < 1.0e-5
        for actual, expected in zip(gradients, expected_gradients, strict=True)
    )


def test_compiled_cuda_replays_do_not_reuse_stale_backward_reductions() -> None:
    torch.manual_seed(83)
    rows = 1025
    modes = 16
    hidden = 24
    module = PhaseGatedComplexFFN(modes, hidden).cuda()
    reference = deepcopy(module)
    compiled = torch.compile(
        module,
        fullgraph=True,
        dynamic=False,
        options={
            "triton.cudagraphs": True,
            "triton.cudagraph_trees": False,
        },
    )

    for step in range(3):
        torch.manual_seed(101 + step)
        real = torch.randn(rows, modes, device="cuda", requires_grad=True)
        imag = torch.randn_like(real, requires_grad=True)
        expected_real = real.detach().clone().requires_grad_()
        expected_imag = imag.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = compiled(real, imag)
            expected_outputs = phase_gated_complex_ffn_reference(
                reference,
                expected_real,
                expected_imag,
            )
        gradients = torch.autograd.grad(
            _loss(outputs),
            (real, imag, *tuple(module.parameters())),
        )
        expected_gradients = torch.autograd.grad(
            _loss(expected_outputs),
            (expected_real, expected_imag, *tuple(reference.parameters())),
        )

        assert all(
            _relative_l2(actual, expected) < 1.0e-3
            for actual, expected in zip(outputs, expected_outputs, strict=True)
        )
        assert all(
            _relative_l2(actual, expected) < 2.0e-2
            for actual, expected in zip(gradients, expected_gradients, strict=True)
        )


def test_custom_kernels_are_available_in_deterministic_mode() -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(mode=True)
        real = torch.randn(17, 8, device="cuda", dtype=torch.bfloat16)
        imag = torch.randn_like(real)
        norm_weight = torch.ones(8, device="cuda")
        assert supports_packed_complex_rms_norm(real, imag, norm_weight)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_large_cuda_workload_compiles_without_shape_cutoff() -> None:
    multiprocessors = torch.cuda.get_device_properties(0).multi_processor_count
    rows = 1024 * multiprocessors + 1
    module = PhaseGatedComplexFFN(1, 1).cuda()
    compiled = torch.compile(module, fullgraph=True)
    real = torch.randn(rows, 1, device="cuda", dtype=torch.bfloat16).requires_grad_()
    imag = torch.randn_like(real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = compiled(real, imag)
    _loss(outputs).backward()

    assert real.grad is not None
    assert imag.grad is not None


def test_compiled_cuda_bf16_forward_backward() -> None:
    torch.manual_seed(59)
    module = PhaseGatedComplexFFN(16, 24).cuda()
    compiled = torch.compile(module, fullgraph=True)
    real = torch.randn(257, 16, device="cuda", dtype=torch.bfloat16).requires_grad_()
    imag = torch.randn_like(real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = compiled(real, imag)
    _loss(outputs).backward()

    assert real.grad is not None
    assert torch.isfinite(real.grad).all()
    assert imag.grad is not None
    assert torch.isfinite(imag.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


@pytest.mark.parametrize(
    ("rows", "modes", "hidden"),
    [(513, 4, 8), (129, 96, 96)],
)
def test_compiled_cuda_generic_phase_gated_shapes(
    rows: int,
    modes: int,
    hidden: int,
) -> None:
    torch.manual_seed(97)
    module = PhaseGatedComplexFFN(modes, hidden).cuda()
    compiled = torch.compile(module, fullgraph=True)
    real = torch.randn(rows, modes, device="cuda", dtype=torch.bfloat16).requires_grad_()
    imag = torch.randn_like(real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = compiled(real, imag)
    _loss(outputs).backward()

    assert real.grad is not None
    assert torch.isfinite(real.grad).all()
    assert imag.grad is not None
    assert torch.isfinite(imag.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_compiled_cuda_transposed_path_layout_forward_backward() -> None:
    torch.manual_seed(99)
    module = PhaseGatedComplexFFN(4, 8).cuda()
    compiled = torch.compile(module, fullgraph=True)
    real = (
        torch.randn(3, 4, 96, device="cuda", dtype=torch.bfloat16)
        .transpose(-2, -1)
        .requires_grad_()
    )
    imag = (
        torch.randn(3, 4, 96, device="cuda", dtype=torch.bfloat16)
        .transpose(-2, -1)
        .requires_grad_()
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = compiled(real, imag)
    _loss(outputs).backward()

    assert real.grad is not None
    assert torch.isfinite(real.grad).all()
    assert imag.grad is not None
    assert torch.isfinite(imag.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_compiled_cuda_fp32_autocast_forward_backward_preserves_dtype() -> None:
    torch.manual_seed(73)
    module = PhaseGatedComplexFFN(16, 24).cuda()
    compiled = torch.compile(module, fullgraph=True)
    real = torch.randn(257, 16, device="cuda", requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = compiled(real, imag)
    _loss(outputs).backward()

    assert outputs[0].dtype is outputs[1].dtype is torch.float32
    assert real.grad is not None
    assert torch.isfinite(real.grad).all()
    assert imag.grad is not None
    assert torch.isfinite(imag.grad).all()
