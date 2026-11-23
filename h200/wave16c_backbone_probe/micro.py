"""Compare individual backbone operations with their PyTorch references."""

import json
import math
import statistics
from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor
from torch.nn import functional

from lnet.fused_backbone_ops import channel_rms_backward, packed_swiglu
from lnet.fused_stem_ops import fused_nhwc_layernorm_gelu

# ruff: noqa: T201


def eager_gate(packed: Tensor) -> Tensor:
    value, gate = packed.chunk(2, dim=-1)
    return value * functional.silu(gate)


def reference_rms(
    x: Tensor, grad: Tensor, weight: Tensor, inverse: Tensor,
) -> tuple[Tensor, Tensor]:
    norm = x * inverse
    scaled = grad * weight.float()
    projection = (norm * scaled).sum(-1, keepdim=True)
    dx = ((scaled - norm / x.shape[-1] * projection) * inverse).to(x.dtype)
    dw = (grad * norm).reshape(-1, x.shape[-1]).sum(0).to(weight.dtype)
    return dx, dw


def compare(a: Tensor, b: Tensor, atol: float, rtol: float) -> dict:
    delta = (a.float() - b.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "passed": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
    }


def gate_result(op: Callable, packed: Tensor, grad: Tensor) -> tuple[Tensor, Tensor]:
    out = op(packed)
    dx, = torch.autograd.grad(out, packed, grad)
    return out, dx


def timings(operations: dict[str, Callable]) -> dict:
    for operation in operations.values():
        for _ in range(3):
            operation()
    torch.cuda.synchronize()
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    samples = {arm: [] for arm in operations}
    for repeat in range(12):
        for arm in list(operations)[::1 if repeat % 2 == 0 else -1]:
            start.record()
            for _ in range(12):
                operations[arm]()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) / 12)
    return {
        arm: {"median_ms": statistics.median(values), "samples_ms": values}
        for arm, values in samples.items()
    }


def stem_step(
    x: Tensor, w: Tensor, bias: Tensor, conv_bias: Tensor, dy: Tensor,
) -> tuple[Tensor, ...]:
    out = fused_nhwc_layernorm_gelu(x, w, bias, 1e-5, nchw=False, conv_bias=conv_bias)
    return (out, *torch.autograd.grad(out, (x, w, bias, conv_bias), dy))


def stem_micro(shape: tuple[int, ...], dtype: torch.dtype) -> dict:
    x = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    w = torch.ones(shape[-1], device="cuda", requires_grad=True)
    bias = torch.zeros_like(w, requires_grad=True)
    conv_bias = torch.zeros_like(w, requires_grad=True)
    dy = torch.randn_like(x)
    return {
        "op": "stem",
        "shape": shape,
        "dtype": str(dtype),
        "timings": timings({"baseline": partial(stem_step, x, w, bias, conv_bias, dy)}),
    }


def main() -> None:
    torch.manual_seed(937)
    torch.set_float32_matmul_precision("highest")
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        rtol = 2 * torch.finfo(dtype).eps
        atol = 1e-5
        for shape in ((3, 7, 11, 31), (1, 9, 19, 160), (4, 13, 7, 513), (128, 56, 56, 128)):
            x = torch.randn(shape, dtype=dtype, device="cuda")
            grad = torch.randn_like(x)
            hidden = shape[-1] * 3 // 2 if shape[0] == 128 else shape[-1]
            packed = torch.randn((*shape[:-1], 2 * hidden), dtype=dtype, device="cuda",
                                 requires_grad=True)
            gate_grad = torch.randn((*shape[:-1], hidden), dtype=dtype, device="cuda")
            result = gate_result(packed_swiglu, packed, gate_grad)
            control = gate_result(eager_gate, packed, gate_grad)
            weight = torch.rand(x.shape[-1], dtype=dtype, device="cuda")
            inv = (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            norm = channel_rms_backward(x, grad, weight, inv)
            expected = reference_rms(x, grad, weight, inv)
            gate = [compare(a, b, atol, rtol) for a, b in zip(result, control, strict=True)]
            rms = [compare(a, b, atol, rtol) for a, b in zip(norm, expected, strict=True)]
            row = {"shape": shape, "dtype": str(dtype), "gate": gate, "rms": rms}
            if math.prod(shape) > 1000000:
                gate_base = torch.compile(eager_gate, fullgraph=True)
                rms_base = torch.compile(reference_rms, fullgraph=True)
                gate_candidate = torch.compile(packed_swiglu, fullgraph=True)
                rms_candidate = torch.compile(channel_rms_backward, fullgraph=True)
                row["gate_timing"] = timings({
                    "baseline": partial(gate_result, gate_base, packed, gate_grad),
                    "candidate": partial(gate_result, gate_candidate, packed, gate_grad),
                })
                row["rms_timing"] = timings({
                    "baseline": partial(rms_base, x, grad, weight, inv),
                    "candidate": partial(rms_candidate, x, grad, weight, inv),
                })
            print(json.dumps(row), flush=True)
        for stem_shape in ((128, 112, 112, 32), (128, 56, 56, 64)):
            print(json.dumps(stem_micro(stem_shape, dtype)), flush=True)


if __name__ == "__main__":
    main()
