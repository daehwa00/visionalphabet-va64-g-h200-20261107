"""Inspect the decomposed RMSNorm VJP and native SwiGLU with CPU Inductor."""

# ruff: noqa: B023, S101, SLF001, T201
import re

import torch
from torch._inductor.utils import run_and_get_code

from lnet.memory_backbone import SwiGLU
from lnet.model_w29c import _CompiledChannelRMSNorm


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(501)
    ffn = SwiGLU(64, 96).to(dtype=torch.bfloat16)
    x = torch.randn(8, 64, dtype=torch.bfloat16, requires_grad=True)
    skip = torch.randn_like(x, requires_grad=True)
    scale = torch.ones(64, dtype=x.dtype, requires_grad=True)
    for residual in (False, True):
        def model(x, skip, scale):  # noqa: ANN001, ANN202
            state = x + skip if residual else x
            # Exercise the same ATen forward and tensor VJP the CUDA arm selects.
            normalized = _CompiledChannelRMSNorm.apply(state, scale, 1e-6, False)  # noqa: FBT003
            return ffn(normalized)

        explained = torch._dynamo.explain(model)(x, skip, scale)
        assert explained.graph_count == 1
        assert explained.graph_break_count == 0
        aot = torch.compile(model, backend="aot_eager", fullgraph=True)
        expected = model(x, skip, scale)
        actual = aot(x, skip, scale)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        dy = torch.randn_like(expected)
        params = (x, skip, scale, *ffn.parameters())
        actual_grads = torch.autograd.grad(actual, params, dy, allow_unused=True)
        expected_grads = torch.autograd.grad(expected, params, dy, allow_unused=True)
        for got, want in zip(actual_grads, expected_grads, strict=True):
            if want is None:
                assert got is None
            else:
                error = (got.float() - want.float()).norm()
                assert error <= 2 * torch.finfo(want.dtype).eps * want.float().norm()
        compiled = torch.compile(model, fullgraph=True)

        def forward_backward() -> tuple[torch.Tensor | None, ...]:
            return torch.autograd.grad(compiled(x, skip, scale), params, dy, allow_unused=True)

        _, sources = run_and_get_code(forward_backward)
        print("residual", residual, "graphs", [1, 0])
        for index, source in enumerate(sources):
            names = sorted(set(re.findall(r"\bcpp_fused_\w+", source)))
            print("forward" if index == 0 else "backward", *names)


if __name__ == "__main__":
    main()
