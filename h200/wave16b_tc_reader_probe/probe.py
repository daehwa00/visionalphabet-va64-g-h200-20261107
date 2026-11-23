"""Verify real-reader parity and record isolated CUDA Graph timing samples."""

# ruff: noqa: ANN001, ANN003, ANN201, ANN202, SLF001, T201
import argparse
import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

from lnet import factorized_reader_kernels as reader

SHAPES = (
    (128, 56, 56, 160, 4),
    (128, 28, 28, 160, 4),
    (128, 14, 14, 160, 4),
    (128, 7, 7, 128, 4),
    (4, 128, 192, 160, 4),
    (4, 512, 768, 48, 4),
    (4, 96, 128, 80, 2),
)
EDGES = ((1, 5, 7, 1, 2), (3, 13, 15, 17, 4), (3, 17, 19, 48, 8))


def time_graph(calls, repetitions, samples):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            for call in calls:
                call()
    for _ in range(3):
        graph.replay()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in starts]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) / repetitions for start, end in zip(starts, ends, strict=True)]


def run(inputs, policy, repetitions, samples):
    x, real, imag, dy = inputs
    os.environ["VA_READER_MMA"] = policy
    launches = {}
    launch = reader._launch_reader_cuda

    def record(module, operation, grid, tensors, sizes, **kwargs):
        launches[operation] = partial(launch, module, operation, grid, tensors, sizes, **kwargs)
        launch(module, operation, grid, tensors, sizes, **kwargs)

    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    requested_before = torch.cuda.memory_stats()["requested_bytes.all.current"]
    result = {}
    with patch.object(reader, "_launch_reader_cuda", record):
        y = reader._packed_spatial(x, real, imag, strict_complex=False)
        dx, dw = reader._packed_spatial_backward(dy, x, real, imag, strict_complex=False)
    result["peak_extra_bytes"] = torch.cuda.max_memory_allocated() - before
    result["peak_requested_bytes"] = (
        torch.cuda.memory_stats()["requested_bytes.all.peak"] - requested_before
    )
    result["launches"] = list(launches)
    # Resolve the stream at replay/capture time. The original launch captured
    # the default stream, whereas CUDA Graph capture uses a side stream.
    for call in launches.values():
        call.keywords.pop("stream")

    counts = {
        "forward": x.numel() + y.numel(),
        "dinput": dy.numel() + dx.numel(),
        "dweight": x.numel() + dy.numel(),
    }
    for operation, call in launches.items():
        key = operation.removeprefix("mma_").removeprefix("shift_")

        def invoke(call=call):
            call(stream=reader._reader_stream(call.args[0], x.device))

        if samples:
            timings = time_graph([invoke], repetitions, samples)
            result[key] = {"ms": timings}
            if key in counts:
                result[key]["effective_GBps"] = [
                    counts[key] * x.element_size() / ms / 1e6 for ms in timings
                ]
    return (y, dx, dw), result


def check(actual, expected, dtype):
    stats = []
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        tolerance = (
            {"rtol": 0.008, "atol": 2e-4}
            if dtype == torch.bfloat16
            else {"rtol": 0.001, "atol": 2e-4}
        )
        if index == 2:
            tolerance = {"rtol": 3e-5, "atol": 0.002}
        max_abs = 0.0
        for left, right in zip(
            value.reshape(-1).split(4_000_000), reference.reshape(-1).split(4_000_000), strict=True
        ):
            torch.testing.assert_close(left, right, **tolerance)
            max_abs = max(max_abs, float((left - right).abs().max()))
        stats.append(max_abs)
    return stats


def verify_reference(inputs, outputs, dtype):
    x, real, imag, dy = inputs
    y, dx, dw = outputs
    dw_ref = torch.zeros_like(dw)
    # Bound the reference transient for the widest stage and oversize layouts.
    chunk = max(1, 4_000_000 // x[0].numel())
    for start in range(0, len(x), chunk):
        sl = slice(start, start + chunk)
        y_ref = reader._spatial_reference(x[sl], real, imag, strict_complex=False)
        dx_ref, partial = reader._spatial_reference_backward(
            dy[sl],
            x[sl],
            real,
            imag,
            strict_complex=False,
        )
        check((y[sl], dx[sl]), (y_ref, dx_ref), dtype)
        dw_ref.add_(partial)
    torch.testing.assert_close(dw, dw_ref, rtol=3e-5, atol=0.002)
    return float((dw - dw_ref).abs().max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--hopper-large", action="store_true")
    parser.add_argument(
        "--shift", action="store_true", help="Compare four shifted outputs per warp"
    )
    args = parser.parse_args()
    torch.manual_seed(61)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    data = {"device": props.name, "sm": [props.major, props.minor], "cases": []}
    shapes = (*SHAPES, *EDGES)
    if args.hopper_large:
        shapes += ((1024, 56, 56, 160, 4), (1024, 28, 28, 160, 4))
    for dtype in (torch.bfloat16, torch.float16):
        for shape in shapes:
            b, h, w, m, r = shape
            kwargs = {"device": "cuda", "dtype": dtype}
            # Scale the adjoint to expose FP32 accumulation errors without
            # overwhelming absolute tolerances with large reductions.
            inputs = (
                torch.randn(b, h, w, r * m, **kwargs),
                torch.randn(m, r, 3, 3, **kwargs) * 0.15,
                torch.randn(m, r, 3, 3, **kwargs) * 0.15,
                torch.randn(b, h, w, 2 * m, **kwargs) * 0.125,
            )
            print(dtype, shape, flush=True)
            baseline, base_stats = run(inputs, "off", args.repetitions, args.samples)
            forced, force_stats = run(inputs, "force", args.repetitions, args.samples)
            auto, auto_stats = run(inputs, "auto", args.repetitions, args.samples)
            case = {
                "shape": shape,
                "dtype": str(dtype),
                "baseline": base_stats,
                "forced": force_stats,
                "auto": auto_stats,
                "forced_baseline_max_abs": check(forced, baseline, dtype),
                "auto_baseline_max_abs": check(auto, baseline, dtype),
            }
            if any(
                f"mma_{op}" not in force_stats["launches"]
                for op in ("forward", "dinput", "dweight")
            ):
                raise AssertionError(f"Forced policy did not launch all MMA operations: {shape}")
            if force_stats["peak_requested_bytes"] > base_stats["peak_requested_bytes"]:
                raise AssertionError(f"MMA increased isolated peak allocation: {shape}")
            if args.shift:
                shifted, shifted_stats = run(inputs, "shift", args.repetitions, args.samples)
                case["shifted"] = shifted_stats
                case["shifted_baseline_max_abs"] = check(shifted, baseline, dtype)
                del shifted
            if args.reference:
                case["forced_reference_dw_max_abs"] = verify_reference(inputs, forced, dtype)
            data["cases"].append(case)
            args.out.write_text(json.dumps(data, indent=2) + "\n")
            del inputs, baseline, forced, auto


if __name__ == "__main__":
    main()
