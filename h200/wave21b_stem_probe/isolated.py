"""Measure exact-match affine-reduction candidates with CUDA Graphs."""

# ruff: noqa: T201
import json
import math
import os
import statistics
from collections.abc import Callable

import torch
from torch.nn import functional

from lnet.fused_stem_ops import _backward, _forward


def capture(call: Callable, stream: torch.cuda.Stream | None = None) -> torch.cuda.CUDAGraph:
    stream = stream or torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(4):
            call()
    return graph


def measure(graph: torch.cuda.CUDAGraph) -> float:
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / 80)
    return statistics.median(samples)


def run(shape: tuple[int, ...], dtype: torch.dtype, *, offset: int = 0) -> bool:
    size = math.prod(shape)
    x = torch.randn(size + offset, device="cuda", dtype=dtype)[offset:].view(shape)
    affine = tuple(torch.empty(shape[-1], device="cuda") for _ in range(3))
    w, b, cb = affine
    w.uniform_(0.7, 1.3)
    b.uniform_(-0.3, 0.3)
    cb.uniform_(-0.3, 0.3)
    dy = torch.randn_like(x)

    def forward() -> torch.Tensor:
        return _forward(x, w, b, cb, 1e-5, dtype, False)  # noqa: FBT003

    def backward() -> tuple[torch.Tensor, ...]:
        return _backward(x, w, b, cb, dy, 1e-5, False)  # noqa: FBT003

    results = {}
    os.environ["LNET_STEM_OPT"] = "off"
    control = (forward(), *backward())
    os.environ["LNET_STEM_OPT"] = "auto"
    candidate = (forward(), *backward())
    results["exact"] = all(torch.equal(a, b) for a, b in zip(control, candidate, strict=True))

    x32 = x.float().requires_grad_()
    for parameter in affine:
        parameter.requires_grad_()
    expected = functional.gelu(functional.layer_norm(x32 + cb, w.shape, w, b)).to(dtype)
    reference = (expected, *torch.autograd.grad(expected, (x32, *affine), dy))
    results["max_abs_fp32"] = [
        float((a.detach().float() - b.detach().float()).abs().max())
        for a, b in zip(control, reference, strict=True)
    ]
    for parameter in affine:
        parameter.requires_grad_(requires_grad=False)
    del reference, expected, x32, control, candidate
    if shape[0] == 128 and dtype != torch.float32:
        payload = x.numel() * x.element_size()
        for order in (("off", "auto"), ("auto", "off")):
            for mode in order:
                os.environ["LNET_STEM_OPT"] = mode
                fwd = measure(capture(forward))
                bwd = measure(capture(backward))
                results.setdefault(mode, []).append({
                    "forward_ms": fwd,
                    "backward_ms": bwd,
                    "total_ms": fwd + bwd,
                    "effective_forward_gbps": 2 * payload / fwd / 1e6,
                    "effective_backward_gbps": 3 * payload / bwd / 1e6,
                })
    record = {"shape": shape, "dtype": str(dtype), "storage_offset": offset, **results}
    print(json.dumps(record), flush=True)
    return results["exact"]


def main() -> None:
    torch.manual_seed(77)
    passed = True
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for shape in ((2, 37, 53, 53), (2, 37, 53, 64), (2, 37, 53, 96)):
            passed &= run(shape, dtype)
        passed &= run((2, 37, 53, 64), dtype, offset=1)
    for dtype in (torch.bfloat16, torch.float16):
        for shape in ((128, 112, 112, 32), (128, 56, 56, 64), (128, 48, 64, 64), (128, 37, 53, 53)):
            passed &= run(shape, dtype)
    if not passed:
        raise SystemExit("bitwise parity failed")


if __name__ == "__main__":
    main()
