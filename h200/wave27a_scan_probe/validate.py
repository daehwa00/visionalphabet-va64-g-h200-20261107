"""Check scan gradients and capture isolated CUDA Graphs on the auxiliary GPU."""

import argparse
import json
import os
import statistics
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from test_general_cuda_scans import _inputs, _pipeline, _reference_vscan

from lnet import general_cuda_scans as scans
from lnet.product_scan_reference import bidirectional_product_scan_reference

# ruff: noqa: SLF001

PRESETS = ("off", "auto", "wave_plan", "bwd_chunk", "bwd_batch2", "hscan_unroll")
CASES = (
    *((s, s, 160, coarse) for s in (56, 28, 14) for coarse in (False, True)),
    (7, 7, 128, False), (37, 53, 160, False), (38, 54, 160, True),
)
DTYPES = (torch.bfloat16, torch.float16)
SCAN_PLANS = scans._scan_plans


def parity() -> Iterator[dict]:
    for h, w, m, coarse in CASES:
        for dtype in DTYPES:
            for interleaved in (False, True):
                yield from parity_case((h, w, m, coarse), dtype, interleaved=interleaved)


def parity_case(case: tuple, dtype: torch.dtype, *, interleaved: bool) -> Iterator[dict]:
    h, w, m, coarse = case
    torch.manual_seed(719)
    px, py, real, imag, aux, path = _inputs(h, w, m, 8, dtype, "cuda", batch=2)
    if interleaved:
        real, imag = torch.cat((real, imag), -1).detach().requires_grad_().chunk(2, -1)
    state32 = torch.stack(bidirectional_product_scan_reference(
        tuple(px), (real.float(), imag.float()),
    ))
    expected = (state32, *_reference_vscan(py, state32, aux, path, coarse))
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    if interleaved:
        vertical_grad = torch.randn((2, oh, ow, 2 * m), device="cuda", dtype=dtype).chunk(2, -1)
    else:
        vertical_grad = tuple(torch.randn_like(x, dtype=dtype) for x in expected[1:])
    gradients = (torch.randn_like(state32, dtype=dtype), *vertical_grad)
    targets = [real, imag, *px, *py, *path]
    fp32 = torch.autograd.grad(expected, targets, gradients, retain_graph=True)
    baseline = None
    norm_limit = 0.02 if dtype == torch.bfloat16 else 0.005
    for preset in PRESETS:
        os.environ["LNET_SCAN_W27A"] = preset
        actual = _pipeline(px, py, real, imag, aux, path, coarse, "cuda")
        observed = torch.autograd.grad(actual, targets, gradients, retain_graph=True)
        if baseline is None:
            baseline = observed
        errors = []
        for value, old, ref in zip(observed, baseline, fp32, strict=True):
            torch.testing.assert_close(
                value, old, rtol=2e-5, atol=3e-5,
                msg=lambda message, arm=preset: f"{case} {dtype} {interleaved=} {arm}: {message}",
            )
            error = ((value.float() - ref.float()).norm()
                     / ref.float().norm().clamp_min(0.02)).item()
            if not error <= norm_limit:
                raise ValueError((case, dtype, interleaved, preset, error))
            errors.append(error)
        yield {
            "case": case, "dtype": str(dtype), "interleaved": interleaved, "preset": preset,
            "exact": [torch.equal(x, y) for x, y in zip(observed, baseline, strict=True)],
            "fp32_relative_l2": errors,
        }


def force_geometry(*args: object, **kwargs: bool | int) -> list[scans.ScanPlan]:
    kwargs["cache_trajectories"] = True
    plans = SCAN_PLANS(*args, **kwargs)
    if kwargs["vertical"]:
        return [
            p for p in plans if p.warps == 2 and p.cache_weights and p.cache_source
            and p.chunk == 8 and not p.cache_line and not p.shared_acc
        ]
    warps = 4 if kwargs["backward"] else 3
    return [p for p in plans if p.warps == warps and p.cache_line]


def force_vector_weights(
    device: int, launch: scans.Launch | None, *, modes: int, hidden: int, wave_limit: int,
) -> scans.Launch | None:
    if launch is None or not launch.key.vertical:
        return launch
    plan, key, _, _ = launch
    smem = plan.smem + 11 * hidden * 32 * (4 - key.dtype.itemsize)
    vector = replace(key, vector_weights=True, precise=False)
    if not scans._resident(device, vector, smem):
        raise ValueError((device, smem, modes, wave_limit))
    return launch._replace(plan=plan._replace(smem=smem), key=vector)


def measure(operation: Callable, preset: str) -> tuple[tuple[torch.Tensor, ...], dict]:
    os.environ["LNET_SCAN_W27A"] = preset
    launches = []
    dispatch = scans._dispatch

    def track(source: torch.Tensor, launch: scans.Launch, args: tuple) -> None:
        launches.append(launch)
        dispatch(source, launch, args)

    with patch.object(scans, "_dispatch", track):
        operation()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation()
        peak = torch.cuda.max_memory_allocated() - before
    samples = []
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(7):
        begin.record()
        for _ in range(5):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / 5)
    plan, key, workers, partials = launches[-1]
    return tuple(output) if isinstance(output, list) else (output,), {
        "preset": preset, "ms": statistics.median(samples), "samples": samples, "peak_bytes": peak,
        "plan": plan._asdict(), "key": {**vars(key), "dtype": str(key.dtype)},
        "workers": workers, "partials": partials,
    }


def graph_parity(actual: torch.Tensor, expected: torch.Tensor, context: str) -> dict:
    exact = torch.equal(actual, expected)
    if exact:
        return {"exact": True, "max_abs": 0.0, "relative_l2": 0.0}
    reference = expected.float()
    diff = actual.float() - reference
    max_abs = diff.abs().max().item()
    relative_l2 = (diff.norm() / reference.norm().clamp_min(1e-6)).item()
    # A fused arithmetic difference can cross a storage cast boundary.
    max_relative = 1e-4 if actual.dtype == torch.float32 else 2 * torch.finfo(actual.dtype).eps
    max_error = max_relative * reference.abs().max().item() + 3e-5
    if not (max_abs <= max_error and relative_l2 <= 1e-4):
        raise ValueError((context, max_abs, relative_l2, max_error))
    return {"exact": False, "max_abs": max_abs, "relative_l2": relative_l2}


def graphs() -> Iterator[dict]:
    for dtype in DTYPES:
        for coarse in (False, True):
            yield from graphs_case(dtype, coarse=coarse)


def graphs_case(dtype: torch.dtype, *, coarse: bool) -> Iterator[dict]:
    torch.manual_seed(719)
    b = 24
    os.environ["LNET_SCAN_W27A"] = "off"
    px, py, real, imag, aux, path = _inputs(56, 56, 160, 8, dtype, "cuda", batch=b)
    states = scans.horizontal_forward(px, real, imag).detach()
    out = scans.vertical_forward(py, states, aux, path, coarse=coarse)
    dy = tuple(torch.randn_like(part) for part in out)
    horizontal_width = 28 if coarse else 56
    horizontal_grad = torch.randn((4, b, 56, horizontal_width, 160), device="cuda", dtype=dtype)
    operations = {
        "vb": lambda: scans.vertical_backward(py, states, aux, path, *dy, coarse=coarse,
                                              compact=coarse),
        "hb": lambda: scans.horizontal_backward(px, [real, imag], states, horizontal_grad, coarse),
        "hf": lambda: scans.horizontal_forward(px, real, imag),
    }
    scans._launch_for_shape.cache_clear()
    try:
        with (
            patch.object(scans, "_scan_plans", force_geometry),
            patch.object(scans, "_with_vector_path_weights", force_vector_weights),
        ):
            for kind, operation in operations.items():
                baseline = None
                for preset in PRESETS:
                    output, metrics = measure(operation, preset)
                    if baseline is None:
                        baseline = output
                    context = f"{kind} {dtype} {coarse=} {preset}"
                    yield {
                        "kernel": kind, "dtype": str(dtype), "coarse": coarse,
                        "parity": [
                            graph_parity(x, y, context)
                            for x, y in zip(output, baseline, strict=True)
                        ],
                        **metrics,
                    }
                    del output
    finally:
        scans._launch_for_shape.cache_clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("parity", "graphs"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("w") as stream:
        iterator = parity() if args.kind == "parity" else graphs()
        for row in iterator:
            print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    main()
