"""Measure isolated forward and backward CUDA Graph replay in milliseconds."""

# ruff: noqa: T201
import argparse
import itertools
import json
import os
import statistics

import torch
from torch import nn

from h200.wave25c_stem_probe.isolated import capture, timed
from lnet.image_layers import LayerNorm2d, _fused_stem_block


def benchmark(dtype: torch.dtype, stage: int, batch: int, mode: str) -> None:
    ci, co, hw = (32, 64, 112) if stage else (3, 32, 224)
    os.environ["LNET_STEM_W27C"] = mode
    torch.manual_seed(96)
    conv = nn.Conv2d(ci, co, 3, 2, 1, device="cuda").to(memory_format=torch.channels_last)
    norm = LayerNorm2d(co).cuda()
    # Stage 0 includes the autocast input conversion. Stage 1 receives an AMP activation.
    x = torch.randn(
        (batch, ci, hw, hw),
        device="cuda",
        dtype=dtype if stage else torch.float32,
    ).contiguous(memory_format=torch.channels_last)
    x.requires_grad_(bool(stage))
    params = (*conv.parameters(), *norm.parameters(), *((x,) if stage else ()))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def forward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            return _fused_stem_block(x, conv, norm, nhwc_output=bool(stage))

    with torch.cuda.stream(stream):
        y = forward()
        dy = torch.randn_like(y)
    torch.cuda.current_stream().wait_stream(stream)

    def backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(y, params, dy, retain_graph=True)

    fwd, bwd = (capture(fn, stream) for fn in (forward, backward))
    measurements = [[timed(graph) for graph in (fwd, bwd)] for _ in range(5)]
    print(
        json.dumps(
            {
                "dtype": str(dtype),
                "stage": stage,
                "batch": batch,
                "mode": mode,
                "forward_ms": statistics.median(t[0] for t in measurements),
                "backward_ms": statistics.median(t[1] for t in measurements),
                "samples": measurements,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--modes", nargs="+", default=["off", "stage0_opt", "stage0_v2", "stage01"])
    args = parser.parse_args()
    os.environ["LNET_FUSED_STEM_CONV"] = "on"
    for dtype, stage, mode in itertools.product(
        (torch.bfloat16, torch.float16),
        (0, 1),
        args.modes,
    ):
        if stage and mode in ("stage0_opt", "stage0_v2"):
            continue
        benchmark(dtype, stage, args.batch, mode)


if __name__ == "__main__":
    main()
