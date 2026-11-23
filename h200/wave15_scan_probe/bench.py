# ruff: noqa: SLF001, FBT003, ANN001, ANN201, ANN202, T201, E501
"""Measure candidate layouts in a single graph with Wave 14A as the control."""

import argparse
import inspect
import json
import statistics
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from lnet import general_cuda_scans as scans
from lnet.product_scan_reference import bidirectional_product_scan_reference

TRIAL_ENABLED = "state_layout" in inspect.signature(scans._hscan).parameters

CASES = (
    (56, 56, 160, False), (56, 56, 160, True),
    (28, 28, 160, False), (28, 28, 160, True),
    (14, 14, 160, False), (7, 7, 160, False),
    (128, 192, 160, False), (512, 768, 48, False),
)
ARMS = {
    "baseline": (0, False, False, False),
    "scratch": (0, True, False, False),
    "row": (1, False, False, False),
    "column": (2, False, False, False),
    "column-shared2": (2, False, False, False),
    "column-shared4": (2, False, False, False),
    "column-2warps": (2, False, False, False),
    "column-4warps": (2, False, False, False),
    "single": (2, True, True, False),
    "drop-source": (2, True, True, True),
}


def sample(shape, dtype, scale):
    return torch.randn(shape, device="cuda", dtype=dtype).mul_(scale)


def signature(outputs):
    return [
        value.flatten()[::max(1, value.numel() // 16384)].float()
        for value in outputs
    ]


def compare(outputs, expected):
    return [
        {
            "exact": torch.equal(value, reference),
            "max_abs": (value - reference).abs().max().item(),
            "reference_max_abs": reference.abs().max().item(),
            "finite": bool(value.isfinite().all()),
        }
        for value, reference in zip(outputs, expected, strict=True)
    ]


@contextmanager
def horizontal_warps(name):
    if name not in ("column-2warps", "column-4warps"):
        yield
        return

    warps = int(name.split("-")[1][0])
    original = scans._scan_plans

    def plans(*args: int, **kwargs: bool | int):
        result = original(*args, **kwargs)
        if kwargs["vertical"]:
            return result
        return [plan for plan in result if plan.warps == warps] or result

    scans._launch_for_shape.cache_clear()
    try:
        with patch.object(scans, "_scan_plans", plans):
            yield
    finally:
        scans._launch_for_shape.cache_clear()


def torch_reference(source, px, py, aux, path, dy, coarse, dtype, arms):
    # Check a complete spatial field with one batch item and four independent
    # modes. This keeps reference autograd bounded on the large rectangles.
    source = [value[:1, ..., :4].contiguous().requires_grad_() for value in source]
    px, py, path = (
        [value[:4].contiguous().requires_grad_() for value in group] for group in (px, py, path)
    )
    aux = [value[..., :4].contiguous() for value in aux]
    dy = [value[:1, ..., :4].contiguous() for value in dy]
    reference_states = torch.stack(bidirectional_product_scan_reference(tuple(px), tuple(source)))
    reference = scans._reference_vertical(py, reference_states, aux, path, coarse, 1e-6)
    grad = torch.autograd.grad(reference, (*source, *px, *py, *path), dy)
    forward_tol, backward_tol = {
        torch.float32: ((4e-5, 3e-6), (4e-4, 2e-4)),
        torch.float16: ((8e-3, 1e-3), (5e-2, 2e-2)),
        torch.bfloat16: ((4e-2, 5e-3), (1.2e-1, 8e-2)),
    }[dtype]
    results = {}
    for name in arms:
        layout, scratch, single, drop_source = ARMS[name]
        hopts = {"state_layout": layout} if TRIAL_ENABLED else {}
        vopts = {
            **hopts, "local_scratch": scratch, "single_store": single,
            "drop_source": drop_source,
            "shared_hidden": int(name[-1]) if name.startswith("column-shared") else 0,
        } if TRIAL_ENABLED else {}
        with torch.no_grad(), horizontal_warps(name):
            states, _ = scans._hscan(px, *source, **hopts)
            output = scans._vscan(py, states, aux, path, coarse, 1e-6, True, **vopts)
            dv = scans._vscan(py, states, aux, path, coarse, 1e-6, False, tuple(dy), compact=coarse, **vopts)
            dx, pole = scans._hscan(px, *source, states, dv[4], compact=coarse, **hopts)
            actual = (*dx.chunk(2, dim=-1), *pole, *dv[:4], *dv[5:])
        for values, target, (rtol, atol) in (
            (output, reference, forward_tol), (actual, grad, backward_tol),
        ):
            torch.testing.assert_close(values, target, rtol=rtol, atol=atol)
        results[name] = {
            "forward": compare(signature(output), signature(reference)),
            "backward": compare(signature(actual), signature(grad)),
        }
    return results


def probe(case, args):
    h, w, modes, coarse = case
    b = args.batch if h <= 56 else 4
    oh, ow = h // (1 + coarse), w // (1 + coarse)
    dtype = getattr(torch, args.dtype)
    source = [sample((b, h, w, modes), dtype, 0.4) for _ in range(2)]
    px, py = [[sample((modes,), torch.float32, 0.12) for _ in range(4)] for _ in range(2)]
    aux = [torch.ones((2, length, modes), device="cuda") for length in (w, h)]
    path = [sample(size, torch.float32, 0.17) for size in (
        (modes, 8, 8), (modes, 8), (modes, 2, 8), (modes, 2),
    )]
    dy = [sample((b, oh, ow, modes), dtype, 0.1) for _ in range(2)]
    expected = None
    results = {}
    for name in args.arms:
        with horizontal_warps(name):
            results[name], expected = measure_arm(
                name, h, modes, coarse, dtype, source, px, py, aux, path, dy,
                expected, args.repeats,
            )
    row = {"shape": [b, h, w, modes, coarse], "dtype": args.dtype, "results": results}
    if args.reference:
        row["torch_reference"] = torch_reference(source, px, py, aux, path, dy, coarse, dtype, args.arms)
    return row


def measure_arm(name, h, modes, coarse, dtype, source, px, py, aux, path, dy, expected, repeats):
    oh = h // (1 + coarse)
    layout, scratch, single, drop_source = ARMS[name]
    options = {
        "state_layout": layout, "local_scratch": scratch,
        "single_store": single, "drop_source": drop_source,
        "shared_hidden": int(name[-1]) if name.startswith("column-shared") else 0,
    }

    horizontal_options = {"state_layout": layout} if TRIAL_ENABLED else {}
    options = options if TRIAL_ENABLED else {}

    def operation():
        states, _ = scans._hscan(px, *source, **horizontal_options)
        output = scans._vscan(py, states, aux, path, coarse, 1e-6, True, **options)
        vertical = scans._vscan(
            py, states, aux, path, coarse, 1e-6, False, tuple(dy),
            compact=coarse, **options,
        )
        dx, pole = scans._hscan(
            px, *source, states, vertical[4], compact=coarse, **horizontal_options,
        )
        return [*output, dx, *pole, *vertical[:4], *vertical[5:]]

    warm = operation()
    torch.cuda.synchronize()
    values = signature(warm)
    expected = values if expected is None else expected
    parity = compare(values, expected)
    del warm, values
    plans = {}
    events = {}

    def track(source, launch, arguments):
        key = launch.key
        phase = ("vscan" if key.vertical else "hscan") + ("_bwd" if key.backward else "_fwd")
        start, stop = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
        plans[phase] = {
            **launch.plan._asdict(), "workers": launch.workers,
            "kernel": {k: v for k, v in asdict(launch.key).items() if k != "dtype"},
            "estimated_scratch_bytes": (
                0 if not key.vertical or launch.plan.cache_line else
                launch.workers * modes * dtype.itemsize * (8 * h if getattr(key, "single_store", False) else 4 * oh)
            ),
        }
        events[phase] = start, stop
        start.record()
        dispatch(source, launch, arguments)
        stop.record()

    graph = torch.cuda.CUDAGraph()
    dispatch = scans._dispatch
    torch.cuda.reset_peak_memory_stats()
    with patch.object(scans, "_dispatch", track), torch.cuda.graph(graph):
        outputs = operation()
    peak = torch.cuda.max_memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {phase: [] for phase in (*events, "combined")}
    for _ in range(repeats):
        start.record()
        for _ in range(5):
            graph.replay()
        stop.record()
        stop.synchronize()
        samples["combined"].append(start.elapsed_time(stop) / 5)
        for phase, (before, after) in events.items():
            samples[phase].append(before.elapsed_time(after))
    result = {
        "samples_ms": samples,
        "median_ms": {phase: statistics.median(times) for phase, times in samples.items()},
        "plan": plans, "sampled_parity": parity, "peak_allocated_bytes": peak,
    }
    del outputs, graph
    return result, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--cases", type=int, nargs="+", default=list(range(len(CASES))))
    args = parser.parse_args()
    torch.manual_seed(731)
    print(torch.cuda.get_device_name(), torch.__version__, scans._l2_size(0), flush=True)
    with args.out.open("w") as out:
        for index in args.cases:
            row = probe(CASES[index], args)
            print(json.dumps(row), file=out, flush=True)
            print(row["shape"], {k: v["median_ms"]["combined"] for k, v in row["results"].items()}, flush=True)


if __name__ == "__main__":
    main()
