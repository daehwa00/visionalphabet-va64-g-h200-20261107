"""Cycled CUDA Graph timings for Stage 0 forward, backward and combined."""

# ruff: noqa: T201
import json
import os
from collections.abc import Callable
from itertools import product

import torch
from torch import nn

from lnet.image_layers import LayerNorm2d, _fused_stem_block


def timed(graph: torch.cuda.CUDAGraph) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(12):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 12


def capture(fn: Callable[[], object], stream: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    return graph


def tensor_errors(
    current: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
) -> list[tuple[float, float]]:
    errors = []
    for got, expected in zip(current, reference, strict=True):
        delta = got.float() - expected.float()
        errors.append((delta.abs().max().item(), (delta.norm() / expected.float().norm()).item()))
    return errors


def benchmark(dtype: torch.dtype, hw: tuple[int, int], ci: int, co: int) -> None:
    torch.manual_seed(91)
    conv = nn.Conv2d(ci, co, 3, 2, 1, device="cuda").to(memory_format=torch.channels_last)
    norm = LayerNorm2d(co).cuda()
    x = torch.randn((128, ci, *hw), device="cuda").contiguous(memory_format=torch.channels_last)
    params = (*conv.parameters(), *norm.parameters())
    dy = torch.randn(
        (128, co, (hw[0] + 1) // 2, (hw[1] + 1) // 2), device="cuda", dtype=dtype
    ).contiguous(memory_format=torch.channels_last)
    graphs = {}
    retained = []
    reference = None
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    modes = ("off", "on", "tensor") if os.environ.get("WAVE25C_TENSOR") == "1" else ("off", "on")
    for mode in modes:
        os.environ["LNET_FUSED_STEM_CONV"] = "on" if mode == "tensor" else mode

        def forward(mode: str = mode) -> torch.Tensor:
            with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
                if mode == "tensor":
                    from tensor_core import run  # noqa: PLC0415

                    return run(
                        x,
                        conv.weight,
                        conv.bias,
                        norm.norm.weight,
                        norm.norm.bias,
                        norm.norm.eps,
                        dtype,
                    )
                return _fused_stem_block(x, conv, norm)

        with torch.cuda.stream(stream):
            y = forward()
            if "tensor" in modes and mode == "off":
                reference = (y, *torch.autograd.grad(y, params, dy, retain_graph=True))
            if mode == "tensor" and reference is not None:
                actual = (y, *torch.autograd.grad(y, params, dy, retain_graph=True))
                print("Tensor Core errors vs Wave 21C", dtype, tensor_errors(actual, reference))
        retained.append((y, dy))

        def backward(y: torch.Tensor = y, dy: torch.Tensor = dy) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(y, params, dy, retain_graph=True)

        def step(dy: torch.Tensor = dy) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(forward(), params, dy)

        # Each graph uses its own pool and saved tensors.
        graphs[mode] = [capture(fn, stream) for fn in (forward, backward, step)]
    timings = {mode: [] for mode in modes}
    for order in (modes, modes[::-1], modes):
        for mode in order:
            timings[mode].append([timed(graph) for graph in graphs[mode]])
    print(json.dumps({"dtype": str(dtype), "hw": hw, "ci": ci, "co": co, "timings": timings}))


if __name__ == "__main__":
    for kind, shape, widths in product(
        (torch.bfloat16, torch.float16), ((224, 224), (113, 127)), ((3, 32), (3, 64), (4, 32))
    ):
        if os.environ.get("WAVE25C_MAIN_ONLY") == "1" and (shape, widths) != ((224, 224), (3, 32)):
            continue
        benchmark(kind, shape, *widths)
