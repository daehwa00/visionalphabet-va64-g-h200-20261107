# pyright: reportMissingImports=false, reportOptionalMemberAccess=false
from __future__ import annotations

import pytest
import torch

from lnet.pac_path_cffn import FactorizedQuadrantPathModeCFFNCombiner


def test_factorized_path_cffn_preserves_the_four_path_contract() -> None:
    torch.manual_seed(501)
    combiner = FactorizedQuadrantPathModeCFFNCombiner(7, 11, 5)
    source_real = torch.randn(2, 3, 4, 7, requires_grad=True)
    source_imag = torch.randn_like(source_real, requires_grad=True)

    output_real, output_imag = combiner.forward_packed(source_real, source_imag)

    assert output_real.shape == source_real.shape
    assert output_imag.shape == source_imag.shape
    (output_real.square().mean() + output_imag.square().mean()).backward()
    assert source_real.grad is not None
    assert source_imag.grad is not None
    assert combiner.path_input.weight_real.grad is not None
    assert combiner.path_output.weight_real.grad is not None
    assert combiner.path_synthesis_real.grad is not None
    assert combiner.path_synthesis_imag.grad is not None


def test_identity_handoff_removes_the_path_cffn() -> None:
    combiner = FactorizedQuadrantPathModeCFFNCombiner(5, 9, 7)
    combiner.use_identity_path_handoff_()
    source_real = torch.randn(2, 4, 5)
    source_imag = torch.randn_like(source_real)

    output_real, output_imag = combiner.forward_packed(source_real, source_imag)

    assert output_real.shape == source_real.shape
    assert output_imag.shape == source_imag.shape


def test_cpu_projection_hooks_remain_observable() -> None:
    combiner = FactorizedQuadrantPathModeCFFNCombiner(5, 9, 7)
    source_real = torch.randn(2, 4, 5)
    source_imag = torch.randn_like(source_real)
    calls: list[None] = []
    handle = combiner.path_input.register_forward_hook(
        lambda *_args: calls.append(None),
    )
    try:
        combiner.forward_packed(source_real, source_imag)
    finally:
        handle.remove()

    assert calls == [None]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_factorized_combiner_supports_reduce_overhead_training() -> None:
    torch.manual_seed(505)
    combiner = FactorizedQuadrantPathModeCFFNCombiner(7, 11, 8).cuda().train()
    runtime = torch.compile(
        combiner.forward_packed,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )
    source_real = torch.randn(
        2,
        3,
        4,
        7,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    source_imag = torch.randn_like(source_real, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_real, output_imag = runtime(source_real, source_imag)
        loss = output_real.float().square().mean() + output_imag.float().square().mean()
    loss.backward()

    assert source_real.grad is not None
    assert combiner.path_input.weight_real.grad is not None
