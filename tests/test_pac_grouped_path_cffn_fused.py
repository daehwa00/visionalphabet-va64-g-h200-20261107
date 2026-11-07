# pyright: reportArgumentType=false, reportMissingImports=false
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
from __future__ import annotations

import pytest
import torch

from lnet.pac_kernel_launch_config import registered_candidates
from lnet.pac_triton_grouped_path_cffn import (
    BACKWARD_LAUNCH_NAME,
    CELL_BACKWARD_LAUNCH_NAME,
    CELL_FORWARD_LAUNCH_NAME,
    FORWARD_LAUNCH_NAME,
    _cell_forward_op,
    _forward_op,
    d4_grouped_cell_path_collapse,
    d4_grouped_cell_path_collapse_reference,
    d4_grouped_path_collapse,
    d4_grouped_path_collapse_reference,
    supports_d4_grouped_path_collapse,
)


def _inputs(
    *,
    modes: int,
    hidden: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, ...]:
    source_real = torch.randn(2, 3, 2, 4, modes, device=device, dtype=dtype)
    source_imag = torch.randn_like(source_real)
    input_weight = torch.randn(modes, 2 * hidden, 8, device=device) * 0.1
    input_bias = torch.randn(modes, 2 * hidden, device=device) * 0.1
    output_weight = torch.randn(modes, 2, 2 * hidden, device=device) * 0.1
    output_bias = torch.randn(modes, 2, device=device) * 0.1
    return (
        source_real,
        source_imag,
        input_weight,
        input_bias,
        output_weight,
        output_bias,
    )


def _differentiable(values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().clone().requires_grad_() for value in values)


def _cell_inputs(
    *,
    modes: int,
    hidden: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, ...]:
    values = _inputs(modes=modes, hidden=hidden, dtype=dtype, device=device)
    return (
        torch.randn(2, 3, 2, 4, 4, modes, device=device, dtype=dtype),
        torch.randn(2, 3, 2, 4, 4, modes, device=device, dtype=dtype),
        *values[2:],
    )


def test_reference_operator_preserves_outputs_and_all_gradients() -> None:
    torch.manual_seed(601)
    reference_inputs = _differentiable(_inputs(modes=5, hidden=3))
    active_inputs = _differentiable(reference_inputs)

    expected = d4_grouped_path_collapse_reference(*reference_inputs)
    actual = d4_grouped_path_collapse(*active_inputs)

    torch.testing.assert_close(actual, expected)
    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, active_inputs, cotangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_output_contract_is_one_path_for_arbitrary_leading_shape() -> None:
    values = _inputs(modes=7, hidden=5)

    output_real, output_imag = d4_grouped_path_collapse(*values)

    assert output_real.shape == (2, 3, 2, 1, 7)
    assert output_imag.shape == output_real.shape
    assert output_real.is_contiguous()
    assert output_imag.is_contiguous()


def test_fused_support_rejects_unsupported_storage() -> None:
    values = _inputs(modes=7, hidden=5)

    assert not supports_d4_grouped_path_collapse(*values)
    noncontiguous = (
        values[0].transpose(0, 1),
        values[1].transpose(0, 1),
        *values[2:],
    )
    assert not supports_d4_grouped_path_collapse(*noncontiguous)


def test_cpu_public_operator_uses_the_reference_before_private_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _inputs(modes=7, hidden=5)
    monkeypatch.setattr(
        "lnet.pac_triton_grouped_path_cffn._forward_op",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected fused dispatch")),
    )

    actual = d4_grouped_path_collapse(*values)
    expected = d4_grouped_path_collapse_reference(*values)

    torch.testing.assert_close(actual, expected)


def test_custom_operator_schema_fake_autograd_and_aot_contracts() -> None:
    values = _inputs(modes=3, hidden=2)

    result = torch.library.opcheck(_forward_op, values, raise_exception=True)

    assert set(result.values()) == {"SUCCESS"}


def test_cell_custom_operator_schema_fake_autograd_and_aot_contracts() -> None:
    values = _cell_inputs(modes=3, hidden=2)

    result = torch.library.opcheck(_cell_forward_op, values, raise_exception=True)

    assert set(result.values()) == {"SUCCESS"}


def test_compiled_cpu_reference_preserves_outputs_and_all_gradients() -> None:
    torch.manual_seed(603)
    reference_inputs = _differentiable(_inputs(modes=3, hidden=2))
    compiled_inputs = _differentiable(reference_inputs)
    compiled = torch.compile(d4_grouped_path_collapse, fullgraph=True, dynamic=False)

    expected = d4_grouped_path_collapse_reference(*reference_inputs)
    actual = compiled(*compiled_inputs)
    cotangent = tuple(torch.randn_like(value) for value in expected)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, compiled_inputs, cotangent)

    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_grouped_path_launches_have_a_bounded_autotune_space() -> None:
    for name in (
        FORWARD_LAUNCH_NAME,
        BACKWARD_LAUNCH_NAME,
        CELL_FORWARD_LAUNCH_NAME,
        CELL_BACKWARD_LAUNCH_NAME,
    ):
        candidates = registered_candidates(name)
        assert 1 < len(candidates) <= 8
        assert all(
            set(candidate.blocks) == {"BLOCK_MODES", "BLOCK_TOKENS"} for candidate in candidates
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32])
def test_cuda_fused_grouped_cell_collapse_matches_reference(
    source_dtype: torch.dtype,
) -> None:
    torch.manual_seed(605)
    reference_inputs = _differentiable(
        _cell_inputs(modes=7, hidden=8, dtype=source_dtype, device="cuda")
    )
    active_inputs = _differentiable(reference_inputs)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = d4_grouped_cell_path_collapse_reference(*reference_inputs)
        actual = d4_grouped_cell_path_collapse(*active_inputs)
    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, active_inputs, cotangent)

    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=2.0e-2, rtol=2.0e-2)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        difference = actual_gradient.float() - expected_gradient.float()
        denominator = expected_gradient.float().square().sum().sqrt().clamp_min(1.0e-12)
        assert float(difference.square().sum().sqrt() / denominator) <= 2.0e-2
        assert bool(torch.isfinite(actual_gradient).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("modes", "hidden"), [(7, 8), (13, 16)])
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32])
def test_cuda_fused_grouped_path_collapse_matches_reference(
    modes: int,
    hidden: int,
    source_dtype: torch.dtype,
) -> None:
    torch.manual_seed(607)
    reference_inputs = _differentiable(
        _inputs(modes=modes, hidden=hidden, dtype=source_dtype, device="cuda")
    )
    active_inputs = _differentiable(reference_inputs)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = d4_grouped_path_collapse_reference(*reference_inputs)
        actual = d4_grouped_path_collapse(*active_inputs)
    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_gradients = torch.autograd.grad(expected, reference_inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, active_inputs, cotangent)

    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=2.0e-2, rtol=2.0e-2)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        difference = actual_gradient.float() - expected_gradient.float()
        denominator = expected_gradient.float().square().sum().sqrt().clamp_min(1.0e-12)
        assert float(difference.square().sum().sqrt() / denominator) <= 2.0e-2
        assert bool(torch.isfinite(actual_gradient).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32])
def test_compiled_cuda_grouped_path_collapse_is_fullgraph_compatible(
    source_dtype: torch.dtype,
) -> None:
    torch.manual_seed(611)
    inputs = _differentiable(_inputs(modes=7, hidden=8, dtype=source_dtype, device="cuda"))
    compiled = torch.compile(
        d4_grouped_path_collapse,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = compiled(*inputs)
    (output[0].float().square().mean() + output[1].float().square().mean()).backward()

    assert output[0].shape == (*inputs[0].shape[:-2], 1, inputs[0].shape[-1])
    assert all(value.grad is not None for value in inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_cuda_grouped_cell_collapse_is_fullgraph_compatible() -> None:
    torch.manual_seed(613)
    inputs = _differentiable(
        _cell_inputs(modes=7, hidden=8, dtype=torch.bfloat16, device="cuda")
    )
    compiled = torch.compile(
        d4_grouped_cell_path_collapse,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = compiled(*inputs)
    (output[0].float().square().mean() + output[1].float().square().mean()).backward()

    assert output[0].shape == (2, 6, 4, 1, 7)
    assert all(value.grad is not None for value in inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_support_accepts_fp32_only_at_the_bf16_autocast_boundary() -> None:
    values = _inputs(modes=7, hidden=8, dtype=torch.bfloat16, device="cuda")

    assert supports_d4_grouped_path_collapse(*values)
    fp32_values = (values[0].float(), values[1].float(), *values[2:])
    assert not supports_d4_grouped_path_collapse(*fp32_values)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert supports_d4_grouped_path_collapse(*fp32_values)

    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(mode=True)
        assert not supports_d4_grouped_path_collapse(*values)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert not supports_d4_grouped_path_collapse(*fp32_values)
    finally:
        torch.use_deterministic_algorithms(mode=previous)
