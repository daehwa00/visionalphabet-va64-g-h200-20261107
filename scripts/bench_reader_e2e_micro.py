#!/usr/bin/env python3
"""End-to-end Reader training timing and parity on one CUDA stream."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import torch
from torch._functorch import config as aot_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphabet.factorized_complex_scan_reader import FactorizedComplexConv2dReader
from alphabet.memory_backbone import ChannelRMSNorm, FactorizedMemoryReader, reader_roofline_costs


@dataclass(frozen=True)
class Shape:
    batch: int
    height: int
    width: int
    channels: int
    modes: int
    rank: int
    kernel: int
    calls: int = 0


SHAPES = (
    Shape(1024, 56, 56, 64, 160, 4, 3, 2),
    Shape(1024, 28, 28, 96, 160, 4, 3, 2),
    Shape(1024, 14, 14, 128, 160, 4, 3, 6),
    Shape(1024, 14, 14, 128, 128, 4, 3, 1),
    Shape(1024, 7, 7, 256, 128, 4, 3, 2),
    Shape(8, 17, 19, 37, 29, 4, 3),
    Shape(8, 13, 11, 64, 48, 4, 5),
    Shape(8, 16, 16, 512, 160, 2, 7),
)
MODES = {
    "off": ("off", "off", False),
    "baseline_kernel": ("off", "baseline_kernel", False),
    "dense_nhwc": ("dense_nhwc", "off", False),
    "dense_nhwc_bench": ("dense_nhwc", "off", True),
    "dense_nhwc_bmm_bench": ("dense_nhwc_bmm", "off", True),
    "auto_roofline_bench": ("auto_roofline", "off", True),
}
DEFAULT_MODES = tuple(MODES)
MODES.update({
    "dense_nhwc_inductor": ("dense_nhwc_inductor", "off", False),
    "dense_nhwc_inductor_bench": ("dense_nhwc_inductor", "off", True),
    "off_bench": ("off", "off", True),
})
COMPILED_MODES = ("off_bench", "dense_nhwc_inductor_bench", "dense_nhwc_bench")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
SIGNALS = ("output", "grad_source", "grad_point_weight", "grad_spatial_weight", "grad_input_norm")
MIB = 1024**2


def make_reader(
    shape: Shape, device: torch.device, *, normalize: bool = True, match: bool = False
) -> FactorizedMemoryReader:
    legacy = FactorizedComplexConv2dReader(
        (shape.channels + 1) // 2, shape.modes, rank=(shape.rank + 1) // 2,
        kernel_size=shape.kernel, normalize_input=False, match_input_rms=match,
        execution_mode="factorized_triton",
    )
    reader = FactorizedMemoryReader(legacy)
    if reader.input_channels != shape.channels or reader.rank != shape.rank:
        # Complex-to-real conversion starts with even widths and ranks.
        reader.input_channels = shape.channels
        reader.rank = shape.rank
        reader.point_weight = torch.nn.Parameter(
            reader.point_weight[:, :shape.rank, :shape.channels].detach().contiguous()
        )
        reader.spatial_weight = torch.nn.Parameter(
            reader.spatial_weight[:, :shape.rank].detach().contiguous()
        )
    if normalize:
        reader.input_norm = ChannelRMSNorm(shape.channels, epsilon=1e-6 / 2)
        with torch.no_grad():
            reader.input_norm.weight.fill_(2**-0.5)
    return reader.to(device)


def set_mode(mode: str) -> None:
    execution, optimization, benchmark = MODES[mode]
    os.environ["ALPHABET_READER_MODE"] = execution
    os.environ["ALPHABET_READER_KERNEL_POLICY"] = optimization
    torch.backends.cudnn.benchmark = benchmark


def make_inputs(
    shape: Shape, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(169)
    dims = shape.batch, shape.height, shape.width
    x = torch.randn((*dims, shape.channels), dtype=dtype, device=device, generator=generator)
    dy = torch.randn((*dims, 2 * shape.modes), dtype=dtype, device=device, generator=generator)
    return x.requires_grad_(), dy


def signals(
    reader: FactorizedMemoryReader,
    x: torch.Tensor,
    dy: torch.Tensor,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, ...]:
    if reader.input_norm is None:
        raise ValueError("parity requires an input norm")
    reader.zero_grad(set_to_none=True)
    x.grad = None
    with torch.autocast(x.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        y = reader(x)
    (y * dy).sum(dtype=torch.float32).backward()
    values = y.detach(), x.grad, reader.point_weight.grad, reader.spatial_weight.grad, reader.input_norm.weight.grad
    if any(value is None for value in values):
        raise RuntimeError("Reader backward did not produce every requested gradient")
    reader.zero_grad(set_to_none=True)
    x.grad = None
    return cast("tuple[torch.Tensor, ...]", values)


def metrics(actual: torch.Tensor, reference: torch.Tensor, dtype: torch.dtype) -> dict:
    limit = 0.04 if dtype == torch.bfloat16 else 0.008
    chunks = []
    # Bound temporary FP32 storage even for the full 56x56 output.
    for actual_part, ref_part in zip(
        actual.reshape(-1).split(MIB), reference.reshape(-1).split(MIB), strict=True,
    ):
        left, right = actual_part.float(), ref_part.float()
        diff = left - right
        chunks.append(torch.stack((
            diff.abs().max(), right.abs().max(), diff.square().sum(), right.square().sum(),
            (left.isnan() | right.isnan()).sum(), (left.isinf() | right.isinf()).sum(),
        )).double())
    stats = torch.stack(chunks)
    maximum, peak = stats[:, :2].amax(0).tolist()
    error_square, ref_square, nan_count, inf_count = stats[:, 2:].sum(0).tolist()
    finite = nan_count == inf_count == 0
    rel_l2 = math.sqrt(error_square / max(ref_square, 1e-30)) if finite else None
    return {
        "max_abs": maximum if finite else None,
        "reference_max_abs": peak if finite else None,
        "rel_l2": rel_l2, "all_finite": finite, "nan": int(nan_count), "inf": int(inf_count),
        "passed": finite and rel_l2 <= limit and maximum <= limit * max(peak, 1e-6),
    }


def roofline(shape: Shape, dtype: torch.dtype, properties: tuple[int, int, int]) -> dict:
    dense, factorized = reader_roofline_costs(
        shape.batch, shape.height, shape.width, shape.channels, shape.modes, shape.rank,
        shape.kernel, dtype, *properties,
    )
    return {
        "mode": "dense_nhwc" if dense < factorized else "factorized_triton",
        "dense_cost": dense if math.isfinite(dense) else None,
        "factorized_cost": factorized if math.isfinite(factorized) else None,
    }


def parity(shape: Shape, dtype: torch.dtype, device: torch.device, modes: list[str]) -> dict:
    reader = make_reader(shape, device)
    x, dy = make_inputs(shape, dtype, device)
    set_mode("dense_nhwc")
    reference = signals(reader, x.detach().float().requires_grad_(), dy.float())
    set_mode("off")
    baseline = signals(reader, x, dy, autocast_dtype=dtype)
    results = {}
    for mode in modes:
        set_mode(mode)
        actual = baseline if mode == "off" else signals(reader, x, dy, autocast_dtype=dtype)
        results[mode] = {
            name: {
                signal: metrics(value, target, dtype)
                for signal, value, target in zip(SIGNALS, actual, expected, strict=True)
            }
            for name, expected in (("fp32_dense", reference), ("off", baseline))
        }
        del actual
    return results


def measure(
    shape: Shape, dtype: torch.dtype, device: torch.device, mode: str,
    args: argparse.Namespace, flush: torch.Tensor,
) -> dict[str, float | int]:
    set_mode(mode)
    reader = make_reader(shape, device)
    x, dy = make_inputs(shape, dtype, device)
    run = reader
    if args.compiled:
        torch._dynamo.reset()
        run = torch.compile(reader, backend="inductor", fullgraph=True, options={"triton.cudagraphs": False})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def forward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=dtype):
            return run(x)

    def backward(y: torch.Tensor) -> None:
        (y * dy).sum(dtype=torch.float32).backward()

    def reset() -> None:
        reader.zero_grad(set_to_none=True)
        x.grad = None

    autocast = aot_config.patch("backward_pass_autocast", "off") if args.compiled else nullcontext()
    with torch.cuda.stream(stream), autocast:
        for _ in range(args.warmup):
            reset()
            backward(forward())
        stream.synchronize()
        timings = {}
        for cold_backward in (True, False):
            reset()
            starting_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

            if args.mode == "graph":
                train_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(train_graph, stream=stream):
                    captured_output = forward()
                    if not cold_backward:
                        backward(captured_output)
                if cold_backward:
                    # Replay the pair in capture order so the saved activations
                    # stay live across the cache flush on this same stream.
                    bwd_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(bwd_graph, stream=stream, pool=train_graph.pool()):
                        backward(captured_output)

            # Initialize events before recording the timed samples.
            events = [
                tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))
                for _ in range(args.iterations)
            ]
            for iteration in events:
                for event in iteration:
                    event.record(stream)
            for start, after_forward, before_backward, end in events:
                if args.mode == "eager":
                    reset()
                flush.add_(1)
                start.record(stream)
                if args.mode == "graph":
                    train_graph.replay()
                else:
                    output = forward()
                if cold_backward:
                    after_forward.record(stream)
                    flush.add_(1)
                    before_backward.record(stream)
                if args.mode == "graph":
                    if cold_backward:
                        bwd_graph.replay()
                else:
                    backward(output)
                    del output
                end.record(stream)
            stream.synchronize()
            if cold_backward:
                timings["fwd_ms"] = statistics.median(
                    start.elapsed_time(mid) for start, mid, _, _ in events
                )
                timings["bwd_ms"] = statistics.median(
                    before.elapsed_time(end) for _, _, before, end in events
                )
            else:
                # Include only the end-to-end iteration and its graph pool.
                peak = torch.cuda.max_memory_allocated(device)
                timings["peak_allocated_bytes"] = peak
                timings["peak_increment_bytes"] = peak - starting_allocated
                timings["total_ms"] = statistics.median(
                    start.elapsed_time(end) for start, _, _, end in events
                )
            if args.mode == "graph":
                reset()
                del captured_output, train_graph
                if cold_backward:
                    del bwd_graph

    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled", action="store_true", help="Time Inductor-compiled forward/backward")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--modes", help="Comma-separated mode names")
    parser.add_argument("--dtypes", default="bf16,fp16")
    parser.add_argument("--timing-dtype", choices=DTYPES, help="Time only this dtype; check parity for all requested dtypes")
    parser.add_argument("--model-shapes-only", action="store_true", help="Skip timing for zero-weight edge shapes")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    defaults = COMPILED_MODES if args.compiled else DEFAULT_MODES
    modes = args.modes.split(",") if args.modes is not None else list(defaults)
    dtypes = args.dtypes.split(",")
    if any(mode not in MODES for mode in modes) or any(dtype not in DTYPES for dtype in dtypes):
        parser.error("--modes or --dtypes contains an unknown value")
    if min(args.warmup, args.iterations) < 1:
        parser.error("--warmup and --iterations must be positive")
    if not args.cpu_smoke and not torch.cuda.is_available():
        parser.error("CUDA is required unless --cpu-smoke is supplied")

    device = torch.device("cpu" if args.cpu_smoke else "cuda")
    torch.manual_seed(169)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.cpu_smoke:
        torch.set_num_threads(2)
        properties = (9, 0, 132)
        shapes = (Shape(2, 5, 7, 37, 29, 4, 3),)
    else:
        os.environ["ALPHABET_FUSED_FACTORIZED_READER"] = "1"
        os.environ["ALPHABET_READER_VECTOR_WIDTH"] = "auto"
        os.environ["ALPHABET_READER_GRADIENT_KERNEL"] = "off"
        props = torch.cuda.get_device_properties(device)
        properties = props.major, props.minor, props.multi_processor_count
        shapes = SHAPES
    flush = None if args.cpu_smoke else torch.zeros(128 * MIB // 4, dtype=torch.float32, device=device)
    report = {
        "device": str(device), "sm": properties[:2], "multiprocessors": properties[2],
        "compiled": args.compiled and not args.cpu_smoke,
        "mode": "cpu-smoke" if args.cpu_smoke else args.mode,
        "cold_cache_bytes": 0 if args.cpu_smoke else 128 * MIB,
        "dtypes": {},
    }
    failures = 0
    for name in dtypes:
        dtype = DTYPES[name]
        rows = []
        for shape in shapes:
            batches = {"small": replace(shape, batch=min(shape.batch, 2))}
            if not args.cpu_smoke:
                batches["full"] = shape
            checks = {key: parity(case, dtype, device, modes) for key, case in batches.items()}
            failures += sum(
                not metric["passed"]
                for group in checks.values()
                for candidate in group.values()
                for comparison in candidate.values()
                for metric in comparison.values()
            )
            timings = {}
            if (
                not args.cpu_smoke and (args.timing_dtype in (None, name))
                and (shape.calls or not args.model_shapes_only)
            ):
                for mode in modes:
                    torch.cuda.empty_cache()
                    timings[mode] = measure(shape, dtype, device, mode, args, flush)
            rows.append({
                "shape": asdict(shape), "roofline": roofline(shape, dtype, properties),
                "parity_roofline": {
                    key: roofline(case, dtype, properties) for key, case in batches.items()
                },
                "parity": checks, "timings": timings,
            })

        totals = {}
        if not args.cpu_smoke and args.timing_dtype in (None, name):
            for mode in modes:
                totals[mode] = {
                    key: sum(row["shape"]["calls"] * row["timings"][mode][key] for row in rows if row["timings"])
                    for key in ("fwd_ms", "bwd_ms", "total_ms")
                }
            for values in totals.values():
                for baseline in ("off", "off_bench", "baseline_kernel"):
                    if baseline in totals:
                        values[f"delta_vs_{baseline}_ms"] = {
                            key: values[key] - totals[baseline][key]
                            for key in ("fwd_ms", "bwd_ms", "total_ms")
                        }
        report["dtypes"][name] = {"shapes": rows, "weighted_calls": sum(s.calls for s in shapes), "weighted": totals}
    report["cudnn_benchmark_by_mode"] = {mode: MODES[mode][2] for mode in modes}
    report["parity_summary"] = {"passed": failures == 0, "failed_checks": failures}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.json_out:
        args.json_out.write_text(payload)
    else:
        print(payload, end="")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
