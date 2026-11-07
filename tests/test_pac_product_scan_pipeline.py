from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import Tensor

from lnet.pac_product_scan_pipeline import (
    ProductScanEpilogue,
    ScanMemoryPolicy,
    run_product_scan_pipeline,
)
from lnet.pac_real2d_math import discrete_pole_real2d

if TYPE_CHECKING:
    from lnet.pac_product_scan_contracts import ProductGainNormalization


def _inputs(
    device: torch.device,
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor],
]:
    modes = 16
    generator = torch.Generator(device=device).manual_seed(3901)
    damping = torch.logspace(
        math.log10(0.04),
        math.log10(0.35),
        modes,
        device=device,
    ).view(1, 1, 1, modes)
    poles = []
    for phase_scale in (0.75, 0.70):
        phase = torch.linspace(0.0, phase_scale * math.pi, modes, device=device).view(
            1, 1, 1, modes
        )
        poles.append(
            tuple(
                value.detach().requires_grad_()
                for value in discrete_pole_real2d(damping, phase, 1.0)
            )
        )
    source_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    source = tuple(
        torch.randn((2, 8, 8, modes), generator=generator, device=device)
        .to(dtype=source_dtype)
        .requires_grad_()
        for _ in range(2)
    )
    return (
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", poles[0]),
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", poles[1]),
        cast("tuple[Tensor, Tensor]", source),
    )


def _clone_inputs(
    inputs: tuple[
        tuple[Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor],
    ],
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor],
]:
    pole_x, pole_y, source = inputs
    cloned = tuple(value.detach().clone().requires_grad_() for value in (*pole_x, *pole_y, *source))
    return (
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", cloned[:4]),
        cast("tuple[Tensor, Tensor, Tensor, Tensor]", cloned[4:8]),
        cast("tuple[Tensor, Tensor]", cloned[8:10]),
    )


def test_product_scan_pipeline_rejects_unknown_epilogue() -> None:
    inputs = _inputs(torch.device("cpu"))
    invalid_epilogue = cast("ProductScanEpilogue", cast("object", "unknown"))

    with pytest.raises(ValueError, match="unsupported product scan epilogue"):
        run_product_scan_pipeline(
            *inputs,
            epilogue=invalid_epilogue,
            gain_normalization="pointwise",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("epilogue", ["coarse", "full16", "descriptor"])
def test_cuda_scan_only_recomputation_matches_retained_autograd(
    epilogue: ProductScanEpilogue,
) -> None:
    retained_inputs = _inputs(torch.device("cuda"))
    recomputed_inputs = _clone_inputs(retained_inputs)

    def evaluate(
        inputs: tuple[
            tuple[Tensor, Tensor, Tensor, Tensor],
            tuple[Tensor, Tensor, Tensor, Tensor],
            tuple[Tensor, Tensor],
        ],
        policy: ScanMemoryPolicy,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        output = run_product_scan_pipeline(
            *inputs,
            epilogue=epilogue,
            gain_normalization="global",
            memory_policy=policy,
        )
        outputs = output if isinstance(output, tuple) else (output,)
        leaves = (*inputs[0], *inputs[1], *inputs[2])
        torch.manual_seed(4100)
        grad_outputs = tuple(torch.randn_like(value) for value in outputs)
        return outputs, torch.autograd.grad(outputs, leaves, grad_outputs)

    retained_outputs, retained_gradients = evaluate(retained_inputs, "retain")
    recomputed_outputs, recomputed_gradients = evaluate(recomputed_inputs, "recompute")
    for actual, expected in zip(recomputed_outputs, retained_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-4, atol=3.0e-4)
    for actual, expected in zip(recomputed_gradients, retained_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3.0e-3, atol=3.0e-3)


@pytest.mark.parametrize("epilogue", ["coarse", "full16", "descriptor"])
@pytest.mark.parametrize("gain_normalization", ["pointwise", "global"])
def test_cpu_scan_recomputation_covers_every_public_normalization(
    epilogue: ProductScanEpilogue,
    gain_normalization: ProductGainNormalization,
) -> None:
    retained_inputs = _inputs(torch.device("cpu"))
    recomputed_inputs = _clone_inputs(retained_inputs)

    def evaluate(
        inputs: tuple[
            tuple[Tensor, Tensor, Tensor, Tensor],
            tuple[Tensor, Tensor, Tensor, Tensor],
            tuple[Tensor, Tensor],
        ],
        policy: ScanMemoryPolicy,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        output = run_product_scan_pipeline(
            *inputs,
            epilogue=epilogue,
            gain_normalization=gain_normalization,
            memory_policy=policy,
        )
        outputs = output if isinstance(output, tuple) else (output,)
        leaves = (*inputs[0], *inputs[1], *inputs[2])
        generator = torch.Generator().manual_seed(4200)
        grad_outputs = tuple(
            torch.randn(value.shape, generator=generator, dtype=value.dtype) for value in outputs
        )
        return outputs, torch.autograd.grad(outputs, leaves, grad_outputs)

    retained_outputs, retained_gradients = evaluate(retained_inputs, "retain")
    recomputed_outputs, recomputed_gradients = evaluate(recomputed_inputs, "recompute")
    for actual, expected in zip(recomputed_outputs, retained_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    for actual, expected in zip(recomputed_gradients, retained_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
