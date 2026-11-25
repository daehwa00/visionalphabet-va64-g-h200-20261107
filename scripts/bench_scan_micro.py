"""Screen eager VScan backward policies with CUDA events and exact parity."""

from __future__ import annotations

# Shape descriptors carry a positional coarse flag.
# ruff: noqa: FBT001
import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lnet import general_cuda_scans as scans

if TYPE_CHECKING:
    from torch import Tensor

MODEL_SHAPES = (
    ("56d", 56, 56, 160, False, 1), ("56c", 56, 56, 160, True, 1),
    ("28d", 28, 28, 160, False, 1), ("28c", 28, 28, 160, True, 1),
    ("14d", 14, 14, 160, False, 5), ("14c", 14, 14, 160, True, 1),
    ("7d", 7, 7, 128, False, 1),
)
EDGE_SHAPES = (
    ("13x19x63c", 13, 19, 63, True, 1), ("9x11x34c", 9, 11, 34, True, 1),
    ("7x7x128c", 7, 7, 128, True, 1), ("1x1x1", 1, 1, 1, False, 1),
)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class Inputs(NamedTuple):
    pole: list[Tensor]
    states: Tensor
    gains: list[Tensor]
    path: list[Tensor]
    gr: Tensor
    gi: Tensor


def make_inputs(batch: int, height: int, width: int, modes: int, coarse: bool, dtype: torch.dtype) -> Inputs:
    pole = [torch.full((modes,), value, device="cuda") for value in (0.73, 0.13, 0.21, 0.09)]
    states = torch.randn(4, batch, height, width, modes, device="cuda", dtype=dtype).mul_(0.1)
    gains = [1 + torch.rand(2, size, modes, device="cuda") for size in (width, height)]
    path = [
        torch.randn(shape, device="cuda", dtype=dtype).mul_(0.125)
        for shape in ((modes, 8, 8), (modes, 8), (modes, 2, 8), (modes, 2))
    ]
    step = 1 + coarse
    grad = torch.randn(batch, height // step, width // step, 2 * modes, device="cuda", dtype=dtype)
    gr, gi = grad.mul_(0.05).split(modes, dim=-1)
    return Inputs(pole, states, gains, path, gr, gi)


def best_mode(totals: dict[str, float], parity: dict[str, bool], threshold: float = 0.15) -> str:
    off = totals["off"]
    eligible = (mode for mode, ms in totals.items() if parity.get(mode, False) and off - ms >= threshold)
    return min(eligible, key=totals.__getitem__, default="off")


def _time(inputs: Inputs, coarse: bool, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        scans.vertical_backward(*inputs, coarse, compact=coarse)
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, stop in events:
        start.record()
        scans.vertical_backward(*inputs, coarse, compact=coarse)
        stop.record()
    events[-1][1].synchronize()
    return statistics.median(start.elapsed_time(stop) for start, stop in events)


def _plan(inputs: Inputs, coarse: bool) -> scans.Launch:
    launch = scans._plan_launch(
        inputs.states, tuple(inputs.states.shape[1:]), backward=True, vertical=True,
        coarse=coarse, hidden=8, precise=False, compact=coarse, interleaved=(True, True),
    )
    if launch is None:
        raise RuntimeError("No fused vertical backward plan")
    return launch


def _describe(launch: scans.Launch, device: int) -> dict[str, int | bool]:
    plan, key, _, _ = launch
    return {
        "warps": key.warps, "chunk": key.chunk, "csrc": key.cache_source,
        "vecw": key.vector_weights, "sgs_rows": plan.sgs_rows, "smem": plan.smem,
        "occupancy": key.warps * scans._resident(device, key, plan.smem),
    }


def _equal(actual: list[Tensor], expected: list[Tensor]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(actual, expected, strict=True))


def _parity(
    modes: tuple[str, ...], timing_dtype: str, mismatches: dict[str, list[str]],
) -> None:
    for dtype_name, dtype in DTYPES.items():
        shapes = EDGE_SHAPES if dtype_name == timing_dtype else (*MODEL_SHAPES, *EDGE_SHAPES)
        for label, h, w, m, coarse, _ in shapes:
            inputs = make_inputs(2, h, w, m, coarse, dtype)
            os.environ["LNET_SCAN_SGS"] = "off"
            expected = scans.vertical_backward(*inputs, coarse, compact=coarse)
            for mode in modes[1:]:
                os.environ["LNET_SCAN_SGS"] = mode
                actual = scans.vertical_backward(*inputs, coarse, compact=coarse)
                if not _equal(actual, expected):
                    mismatches[mode].append(f"{dtype_name}/{label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default=",".join(scans.SGS_MODES))
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--parity", action="store_true", help="Require exact BF16 and FP16 parity to qualify.")
    parser.add_argument("--out", type=Path, help="Write JSON to this path instead of stdout.")
    args = parser.parse_args()
    requested = tuple(token.strip() for token in args.modes.split(","))
    if set(requested).difference(scans.SGS_MODES):
        parser.error("--modes must contain valid LNET_SCAN_SGS tokens")
    if args.batch < 1 or args.iters < 1 or args.warmup < 0:
        parser.error("--batch and --iters must be positive and --warmup cannot be negative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    modes = tuple(dict.fromkeys(("off", *requested)))
    os.environ["LNET_VSCAN_DECOUPLED"] = "off"
    started = perf_counter()
    torch.manual_seed(141)
    results = {mode: {} for mode in modes}
    totals = dict.fromkeys(modes, 0.0)
    mismatches = {mode: [] for mode in modes} if args.parity else {}
    with torch.inference_mode():
        for label, h, w, m, coarse, calls in MODEL_SHAPES:
            inputs = make_inputs(args.batch, h, w, m, coarse, DTYPES[args.dtype])
            expected = None
            samples = {}
            for mode in modes:
                os.environ["LNET_SCAN_SGS"] = mode
                launch = _plan(inputs, coarse)
                if launch not in samples:
                    ms = _time(inputs, coarse, args.warmup, args.iters)
                    passed = True
                    if args.parity:
                        actual = scans.vertical_backward(*inputs, coarse, compact=coarse)
                        if expected is None:
                            expected = actual
                        else:
                            passed = _equal(actual, expected)
                        del actual
                    samples[launch] = ms, passed
                ms, passed = samples[launch]
                if not passed:
                    mismatches[mode].append(f"{args.dtype}/{label}")
                totals[mode] += calls * ms
                results[mode][label] = {
                    "median_ms": ms, "calls": calls,
                    "plan": _describe(launch, inputs.states.device.index or 0),
                }
            del inputs, expected
        if args.parity:
            _parity(modes, args.dtype, mismatches)
    parity = {mode: not failures for mode, failures in mismatches.items()}
    report = {
        "batch": args.batch, "dtype": args.dtype, "warmup": args.warmup, "iters": args.iters,
        "best_mode": best_mode(totals, parity), "weighted_ms": totals,
        "elapsed_seconds": perf_counter() - started,
        "delta_vs_off_ms": {mode: ms - totals["off"] for mode, ms in totals.items()},
        "parity_pass": parity, "parity_mismatches": mismatches, "shapes": results,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.out is None:
        print(output, end="")
    else:
        args.out.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
