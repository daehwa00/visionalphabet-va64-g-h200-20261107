"""CUDA-graph Stage 0 timing with L2 eviction outside CUDA events."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from statistics import median

import torch
from torch import nn

from lnet.image_layers import LayerNorm2d, _fused_conv_stem_supported, _fused_stem_block

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def arguments(description: str, modes: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--modes", default=modes)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--flush-l2", action="store_true")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--out", "--json-out", type=Path)
    return parser


def cold_graph_ms(
    call: Callable,
    iterations: int,
    flush: torch.Tensor | None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> float:
    if stream is None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
        stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    samples = []
    with torch.cuda.stream(stream):
        for _ in range(iterations):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if flush is not None:
                flush.zero_()
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
        got_fp32, ref_fp32 = got.detach().float(), ref.detach().float()
        errors.append(float((got_fp32 - ref_fp32).norm() / ref_fp32.norm().clamp_min(1e-8)))
    return errors


def result_json(out: Path | None, result: dict) -> int:
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if out is None:
        print(payload, flush=True)
    else:
        out.write_text(payload)
    return int(result["status"] != "ok")


def stem_case(batch: int, h: int, w: int, dtype: torch.dtype, *, input_grad: bool = False) -> tuple:
    x = torch.randn(batch, 3, h, w, device="cuda").contiguous(memory_format=torch.channels_last)
    x.requires_grad_(input_grad)
    conv = nn.Conv2d(3, 32, 3, 2, 1, device="cuda")
    norm = LayerNorm2d(32).cuda()
    projection = nn.Conv2d(32, 64, 3, padding=1, device="cuda")
    projection_norm = LayerNorm2d(64).cuda()

    def forward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            hidden = _fused_stem_block(x, conv, norm)
            return _fused_stem_block(hidden, projection, projection_norm, nhwc_output=True).movedim(
                -1, 1
            )

    params = (
        *conv.parameters(),
        *norm.parameters(),
        *projection.parameters(),
        *projection_norm.parameters(),
    )
    with torch.autocast("cuda", dtype=dtype):
        eligible = _fused_conv_stem_supported(x, conv, tiled=True)
    return forward, ((x, *params) if input_grad else params), eligible


def stem_parity(modes: list[str]) -> dict:
    parity = {}
    for name, dtype in DTYPES.items():
        for h, w in ((224, 224), (31, 33)):
            for input_grad in (False, True):
                forward, params, eligible = stem_case(4, h, w, dtype, input_grad=input_grad)
                rows = {}
                for mode in dict.fromkeys(("off", *modes)):
                    os.environ["LNET_STEM_STAGE0_MODE"] = mode
                    y = forward()
                    if not rows:
                        dy = torch.randn_like(y)
                    rows[mode] = (y.detach(), *torch.autograd.grad(y, params, dy))
                parity.setdefault(name, {})[f"{h}x{w}_{'dx' if input_grad else 'tiled'}"] = {
                    "cuda_tiled_eligible": eligible,
                    "errors": {mode: relative_errors(rows[mode], rows["off"]) for mode in modes},
                }
    return parity


def measure(
    forward: Callable, params: tuple, flush: torch.Tensor | None, args: argparse.Namespace
) -> dict:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.no_grad():
            dy = torch.randn_like(forward())

        def full_step() -> tuple:
            y = forward()
            return y, torch.autograd.grad(y, params, dy)

        for _ in range(max(2, args.warmup)):
            _, grads = full_step()
            if not all(
                torch.isfinite(grad).all().item() and grad.count_nonzero().item() for grad in grads
            ):
                raise RuntimeError("invalid warmup gradient")
        del grads, _
    stream.synchronize()
    fwd = cold_graph_ms(forward, args.iters, flush, stream=stream)
    total = cold_graph_ms(full_step, args.iters, flush, stream=stream)
    return {"forward_ms": fwd, "backward_ms": max(0.0, total - fwd), "total_ms": total}


def main() -> int:
    parser = arguments(__doc__, "off,cuda_tiled")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    modes = args.modes.split(",")
    if set(modes) - {"off", "cuda_tiled", "auto"} or min(args.batch, args.iters) <= 0:
        parser.error("expected valid modes, positive batch and iterations")
    os.environ["LNET_FUSED_STEM_CONV"] = "on"
    torch.manual_seed(176)
    measurements = {}
    flush = torch.zeros(128 * 1024**2, device="cuda", dtype=torch.uint8) if args.flush_l2 else None
    parity = stem_parity(modes) if args.parity else {}
    forward, params, eligible = stem_case(args.batch, 224, 224, torch.bfloat16)
    for mode in modes:
        os.environ["LNET_STEM_STAGE0_MODE"] = mode
        print(f"timing stem {mode}", flush=True)
        measurements[mode] = measure(forward, params, flush, args)
    if "off" in measurements:
        for row in measurements.values():
            row["delta_vs_off_ms"] = row["total_ms"] - measurements["off"]["total_ms"]
    passed = all(
        error <= {"bf16": 0.035, "fp16": 0.006}[dtype]
        for dtype, shapes in parity.items()
        for row in shapes.values()
        for errors in row["errors"].values()
        for error in errors
    )
    return result_json(
        args.out,
        {
            "status": "ok" if passed else "error",
            "batch": args.batch,
            "output_shape": [args.batch, 64, 112, 112],
            "auto_resolved_mode": "cuda_tiled" if eligible else "off",
            "flush_l2": args.flush_l2,
            "bf16_ms": measurements,
            "parity_rel_l2": parity,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
