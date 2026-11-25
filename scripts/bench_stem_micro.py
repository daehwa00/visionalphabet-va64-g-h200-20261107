"""CUDA-graph Stage 0 timing with L2 eviction outside CUDA events."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from statistics import median

import torch
from torch import nn

from lnet.image_layers import LayerNorm2d, _fused_stem_block

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def arguments(description: str, modes: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--modes", default=modes)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--flush-l2", action="store_true")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser


def cold_graph_ms(call: Callable, iterations: int, flush: torch.Tensor | None) -> float:
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        call()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if flush is not None:
            flush.add_(1)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    # Output storage must survive replay.
    del output
    return median(samples)


def relative_errors(actual: tuple, expected: tuple) -> list[float]:
    errors = []
    for got, ref in zip(actual, expected, strict=True):
        got_fp32, ref_fp32 = got.float(), ref.float()
        errors.append(float((got_fp32 - ref_fp32).norm() / ref_fp32.norm().clamp_min(1e-8)))
    return errors


def result_json(out: Path | None, result: dict) -> int:
    payload = json.dumps(result, indent=2) + "\n"
    if out is None:
        print(payload)
    else:
        out.write_text(payload)
    return int(result["status"] != "ok")


def stem_case(batch: int, h: int, w: int, dtype: torch.dtype) -> tuple:
    x = torch.randn(batch, 3, h, w, device="cuda").contiguous(memory_format=torch.channels_last)
    conv = nn.Conv2d(3, 32, 3, 2, 1, device="cuda")
    norm = LayerNorm2d(32).cuda()

    def forward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype):
            return _fused_stem_block(x, conv, norm)

    params = (*conv.parameters(), *norm.parameters())
    return forward, params


def main() -> int:
    parser = arguments(__doc__, "off,cuda_tiled")
    args = parser.parse_args()
    modes = args.modes.split(",")
    if set(modes) - {"off", "cuda_tiled", "auto"} or min(args.batch, args.iters) <= 0:
        parser.error("expected valid modes, positive batch and iterations")
    os.environ["LNET_FUSED_STEM_CONV"] = "on"
    torch.manual_seed(176)
    measurements, parity = {}, {}
    flush = torch.zeros(128 * 1024**2, device="cuda", dtype=torch.uint8) if args.flush_l2 else None
    if args.parity:
        for name, dtype in DTYPES.items():
            forward, params = stem_case(2, 33, 35, dtype)
            rows = {}
            for mode in dict.fromkeys(("off", *modes)):
                os.environ["LNET_STEM_STAGE0_MODE"] = mode
                y = forward()
                if not rows:
                    dy = torch.randn_like(y)
                rows[mode] = (y, *torch.autograd.grad(y, params, dy))
            parity[name] = {mode: relative_errors(rows[mode], rows["off"]) for mode in modes}
    forward, params = stem_case(args.batch, 224, 224, torch.bfloat16)
    for mode in modes:
        os.environ["LNET_STEM_STAGE0_MODE"] = mode
        y = forward()
        dy = torch.ones_like(y)

        backward = partial(torch.autograd.grad, y, params, dy, retain_graph=True)

        measurements[mode] = {
            "forward_ms": cold_graph_ms(forward, args.iters, flush),
            "backward_ms": cold_graph_ms(backward, args.iters, flush),
        }
        del y, dy, backward
    passed = all(
        error <= {"bf16": 0.035, "fp16": 0.006}[dtype]
        for dtype, rows in parity.items() for errors in rows.values() for error in errors
    )
    return result_json(args.out, {
        "status": "ok" if passed else "error", "batch": args.batch,
        "flush_l2": args.flush_l2, "bf16_ms": measurements, "parity_rel_l2": parity,
    })


if __name__ == "__main__":
    raise SystemExit(main())
