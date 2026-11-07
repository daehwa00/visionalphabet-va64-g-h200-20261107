# pyright: reportMissingImports=false, reportUnknownLambdaType=false
from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional

from lnet.pac_d4_path_cffn import d4_grouped_path_collapse as fused_path_collapse
from lnet.pac_grouped_path_cffn import (
    GroupedWidelyLinear,
    grouped_cartesian_cffn,
)


def _reference_grouped(
    layer: GroupedWidelyLinear,
    real: torch.Tensor,
    imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    direct_real = torch.einsum("...ip,poi->...op", real, layer.weight_real)
    direct_imag = torch.einsum("...ip,poi->...op", imag, layer.weight_imag)
    conjugate_real = torch.einsum("...ip,poi->...op", real, layer.conjugate_real)
    conjugate_imag = torch.einsum("...ip,poi->...op", imag, layer.conjugate_imag)
    output_real = direct_real - direct_imag + conjugate_real + conjugate_imag
    output_imag = (
        torch.einsum("...ip,poi->...op", real, layer.weight_imag)
        + torch.einsum("...ip,poi->...op", imag, layer.weight_real)
        + torch.einsum("...ip,poi->...op", real, layer.conjugate_imag)
        - torch.einsum("...ip,poi->...op", imag, layer.conjugate_real)
    )
    if layer.bias_real is not None and layer.bias_imag is not None:
        output_real = output_real + layer.bias_real.T
        output_imag = output_imag + layer.bias_imag.T
    return output_real, output_imag


def test_grouped_widely_linear_matches_explicit_math_and_gradients() -> None:
    torch.manual_seed(621)
    layer = GroupedWidelyLinear(5, 4, 8).double()
    real = torch.randn(2, 3, 2, 4, 5, dtype=torch.double, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    actual = layer(real, imag)
    expected = _reference_grouped(layer, real, imag)

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    actual_loss = actual[0].square().sum() + actual[1].square().sum()
    expected_loss = expected[0].square().sum() + expected[1].square().sum()
    tensors = (real, imag, *layer.parameters())
    actual_gradients = torch.autograd.grad(actual_loss, tensors, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_loss, tensors)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=1.0e-11,
            atol=1.0e-11,
        )


def test_grouped_cartesian_cffn_matches_sequential_projections() -> None:
    torch.manual_seed(623)
    input_projection = GroupedWidelyLinear(5, 4, 8).double()
    output_projection = GroupedWidelyLinear(5, 8, 1).double()
    real = torch.randn(2, 3, 2, 4, 5, dtype=torch.double, requires_grad=True)
    imag = torch.randn_like(real, requires_grad=True)

    actual = grouped_cartesian_cffn(
        real,
        imag,
        input_projection=input_projection,
        output_projection=output_projection,
    )
    hidden = input_projection(real, imag)
    expected = output_projection(functional.silu(hidden[0]), functional.silu(hidden[1]))

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_cpu_projection_hooks_keep_the_observable_reference_path() -> None:
    input_projection = GroupedWidelyLinear(5, 4, 8)
    output_projection = GroupedWidelyLinear(5, 8, 1)
    real = torch.randn(2, 3, 2, 4, 5)
    imag = torch.randn_like(real)
    calls: list[str] = []
    handles = (
        input_projection.register_forward_hook(lambda *_args: calls.append("input")),
        output_projection.register_forward_hook(lambda *_args: calls.append("output")),
    )
    try:
        grouped_cartesian_cffn(
            real,
            imag,
            input_projection=input_projection,
            output_projection=output_projection,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert calls == ["input", "output"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32])
def test_cuda_grouped_cffn_dispatches_and_updates_original_parameters(
    monkeypatch: pytest.MonkeyPatch,
    source_dtype: torch.dtype,
) -> None:
    torch.manual_seed(627)
    input_projection = GroupedWidelyLinear(7, 4, 8).cuda().train()
    output_projection = GroupedWidelyLinear(7, 8, 1).cuda().train()
    real = torch.randn(
        2,
        3,
        2,
        4,
        7,
        device="cuda",
        dtype=source_dtype,
        requires_grad=True,
    )
    imag = torch.randn_like(real, requires_grad=True)
    calls: list[None] = []

    def observed(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(None)
        return fused_path_collapse(*args)

    monkeypatch.setattr(
        "lnet.pac_grouped_path_cffn.d4_grouped_path_collapse",
        observed,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = grouped_cartesian_cffn(
            real,
            imag,
            input_projection=input_projection,
            output_projection=output_projection,
        )
        loss = output_real.float().square().mean() + output_imag.float().square().mean()
    loss.backward()

    assert calls == [None]
    assert real.grad is not None
    assert imag.grad is not None
    assert all(parameter.grad is not None for parameter in input_projection.parameters())
    assert all(parameter.grad is not None for parameter in output_projection.parameters())


def test_cuda_grouped_cffn_rejects_an_unsupported_fused_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with FakeTensorMode(allow_non_fake_inputs=True):
        input_projection = GroupedWidelyLinear(7, 4, 8)
        output_projection = GroupedWidelyLinear(7, 8, 1)
        real = torch.empty(2, 3, 2, 4, 7, device="cuda", dtype=torch.bfloat16)
        imag = torch.empty_like(real)
        monkeypatch.setattr(
            "lnet.pac_grouped_path_cffn.supports_d4_grouped_path_collapse",
            lambda *_args: False,
        )

        with pytest.raises(RuntimeError, match="no alternate CUDA implementation"):
            grouped_cartesian_cffn(
                real,
                imag,
                input_projection=input_projection,
                output_projection=output_projection,
            )


def test_cuda_grouped_cffn_rejects_projection_hooks() -> None:
    with FakeTensorMode(allow_non_fake_inputs=True):
        input_projection = GroupedWidelyLinear(7, 4, 8)
        output_projection = GroupedWidelyLinear(7, 8, 1)
        handle = input_projection.register_forward_hook(lambda *_args: None)
        real = torch.empty(2, 3, 2, 4, 7, device="cuda", dtype=torch.bfloat16)
        imag = torch.empty_like(real)

        try:
            with pytest.raises(RuntimeError, match="requires transparent projections"):
                grouped_cartesian_cffn(
                    real,
                    imag,
                    input_projection=input_projection,
                    output_projection=output_projection,
                )
        finally:
            handle.remove()


def test_cuda_grouped_cffn_rejects_projection_subclasses() -> None:
    class DerivedProjection(GroupedWidelyLinear):
        pass

    with FakeTensorMode(allow_non_fake_inputs=True):
        input_projection = DerivedProjection(7, 4, 8)
        output_projection = GroupedWidelyLinear(7, 8, 1)
        real = torch.empty(2, 3, 2, 4, 7, device="cuda", dtype=torch.bfloat16)
        imag = torch.empty_like(real)

        with pytest.raises(RuntimeError, match="requires exact grouped projections"):
            grouped_cartesian_cffn(
                real,
                imag,
                input_projection=input_projection,
                output_projection=output_projection,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float32])
def test_compiled_grouped_cffn_supports_reduce_overhead_training(
    source_dtype: torch.dtype,
) -> None:
    torch.manual_seed(631)
    input_projection = GroupedWidelyLinear(7, 4, 8).cuda().train()
    output_projection = GroupedWidelyLinear(7, 8, 1).cuda().train()

    def forward(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return grouped_cartesian_cffn(
            real,
            imag,
            input_projection=input_projection,
            output_projection=output_projection,
        )

    runtime = torch.compile(
        forward,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    real = torch.randn(
        2,
        3,
        2,
        4,
        7,
        device="cuda",
        dtype=source_dtype,
        requires_grad=True,
    )
    imag = torch.randn_like(real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = runtime(real, imag)
        loss = output_real.float().square().mean() + output_imag.float().square().mean()
    loss.backward()

    assert output_real.shape == (2, 3, 2, 1, 7)
    assert output_real.dtype is output_imag.dtype is torch.bfloat16
    assert real.grad is not None
    assert all(parameter.grad is not None for parameter in input_projection.parameters())
