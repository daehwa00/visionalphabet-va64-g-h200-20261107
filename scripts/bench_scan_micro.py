"""Screen eager CUDA scan policies with CUDA events and per-tensor parity."""

from __future__ import annotations

# Shape descriptors carry a positional coarse flag.
# ruff: noqa: FBT001
import argparse
import json
import math
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
    from collections.abc import Callable

    from torch import Tensor

MODEL_SHAPES = (
    ("56d", 56, 56, 160, False, 1),
    ("56c", 56, 56, 160, True, 1),
    ("28d", 28, 28, 160, False, 1),
    ("28c", 28, 28, 160, True, 1),
    ("14d", 14, 14, 160, False, 5),
    ("14c", 14, 14, 160, True, 1),
    ("7d", 7, 7, 128, False, 1),
)
EDGE_SHAPES = (
    ("13x19x63c", 13, 19, 63, True, 1),
    ("9x11x34c", 9, 11, 34, True, 1),
    ("7x7x128c", 7, 7, 128, True, 1),
    ("1x1x1", 1, 1, 1, False, 1),
    ("19x23x65d", 19, 23, 65, False, 1),
    ("37x17x97d", 37, 17, 97, False, 1),
    ("64x21x95d", 64, 21, 95, False, 1),
)
EDGE_LABELS = {shape[0] for shape in EDGE_SHAPES}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
OPS = {
    "vscan_bwd": ("LNET_SCAN_SGS", scans.SGS_MODES),
    "vscan_fwd": ("LNET_SCAN_VFWD", scans.VFWD_MODES),
    "hscan_fwd": ("LNET_SCAN_HSCAN", scans.HSCAN_MODES),
    "hscan_bwd": ("LNET_SCAN_HSCAN", scans.HSCAN_MODES),
}
W43_MODES = {
    "vscan_bwd": "strict_ro",
    "vscan_fwd": "batch4",
    "hscan_fwd": "bwd_opt",
    "hscan_bwd": "bwd_opt",
}
POLE_NAMES = ("decay_real", "decay_imag", "gain_real", "gain_imag")
TENSOR_NAMES = {
    "vscan_bwd": (*POLE_NAMES, "states", "path_wi", "path_bi", "path_wo", "path_bo"),
    "vscan_fwd": ("output",),
    "hscan_fwd": ("states",),
    "hscan_bwd": (*POLE_NAMES, "source"),
}


class Inputs(NamedTuple):
    pole: list[Tensor]
    states: Tensor
    gains: list[Tensor]
    path: list[Tensor]
    gr: Tensor
    gi: Tensor


def make_inputs(
    batch: int, height: int, width: int, modes: int, coarse: bool, dtype: torch.dtype
) -> Inputs:
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
    eligible = (
        mode for mode, ms in totals.items() if parity.get(mode, False) and off - ms >= threshold
    )
    return min(eligible, key=totals.__getitem__, default="off")


def _time(
    run: Callable,
    warmup: int,
    iters: int,
    flush_buffer: Tensor | None = None,
    *,
    graph: bool = False,
) -> float:
    for _ in range(warmup):
        run()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            output = run()
        run = capture.replay
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    # Initialize lazy event handles before the first measured call.
    for start, stop in events:
        start.record()
        stop.record()
    torch.cuda.current_stream().synchronize()
    for start, stop in events:
        if flush_buffer is not None:
            flush_buffer.zero_()
        start.record()
        run()
        stop.record()
    events[-1][1].synchronize()
    if graph:
        del output
    return statistics.median(start.elapsed_time(stop) for start, stop in events)


def _case(op: str, shape: tuple, batch: int, dtype: torch.dtype, *, compact: bool) -> tuple:
    _, h, w, m, coarse, _ = shape
    backward = op.endswith("_bwd")
    vertical = op.startswith("vscan")
    metadata = {"interleaved": (True, True)}
    if vertical:
        inputs = make_inputs(batch, h, w, m, coarse, dtype)
        source = inputs.states
    if op == "vscan_bwd":

        def run() -> list[Tensor]:
            return scans.vertical_backward(*inputs, coarse, compact=compact)
    elif op == "vscan_fwd":

        def run() -> list[Tensor]:
            packed = scans.vertical_forward_packed(
                *inputs[:4],
                coarse,
                1e-6,
                planar=False,
            )
            return [packed]
    else:
        # Real and imaginary channels share storage as in the packed model.
        pair = torch.randn(batch, h, w, 2 * m, device="cuda", dtype=dtype).mul_(0.1)
        real, imag = pair.split(m, dim=-1)
        source = real
        metadata = scans._horizontal_layout(real, imag)
        pole = [torch.full((m,), value, device="cuda") for value in (0.73, 0.13, 0.21, 0.09)]
        if backward:
            # The forward output is needed only for backward and is generated once.
            states = scans.horizontal_forward(pole, real, imag)
            adjoints = torch.randn(
                4,
                batch,
                h,
                w // (1 + compact),
                m,
                device="cuda",
                dtype=dtype,
            ).mul_(0.05)

            def run() -> list[Tensor]:
                return scans.horizontal_backward(
                    pole, [real, imag], states, adjoints, compact=compact
                )
        else:

            def run() -> list[Tensor]:
                return [scans.horizontal_forward(pole, real, imag)]

    def plan() -> scans.Launch:
        launch = scans._plan_launch(
            source,
            (batch, h, w, m),
            backward=backward,
            vertical=vertical,
            coarse=vertical and coarse,
            hidden=8 if vertical else 0,
            precise=op == "vscan_fwd",
            compact=backward and compact,
            **metadata,
        )
        if launch is None:
            raise RuntimeError(f"No fused {op} plan")
        return launch

    return run, plan


def _describe(launch: scans.Launch) -> dict[str, int | bool]:
    plan, key, workers, partials = launch
    return {
        "warps": key.warps,
        "chunk": key.horizontal_span or key.chunk,
        "cache_source": key.cache_source,
        "readonly": key.backward_readonly or key.forward_readonly or key.horizontal_readonly,
        "cache_line": key.cache_line,
        "cache_weights": key.cache_weights,
        "vector_weights": key.vector_weights,
        "register_path": key.register_path,
        "batch_path": key.batch_path,
        "register_batch2": key.register_batch2,
        "unroll": key.horizontal_unroll or (8 if key.horizontal_unroll8 else 0),
        "sgs_rows": plan.sgs_rows,
        "smem": plan.smem,
        "workers": workers,
        "partials": partials,
        "occupancy": key.warps * scans._resident(torch.cuda.current_device(), key, plan.smem),
    }


def parity_stats(actual: Tensor, expected: Tensor) -> dict[str, int | float | bool | None]:
    if not actual.numel():
        return {
            "max_abs": 0.0,
            "max_rel": 0.0,
            "mismatch_count": 0,
            "exact_match": True,
            "tol_pass": True,
        }
    actual32, expected32 = actual.float(), expected.float()
    error = (actual32 - expected32).abs()
    relative = error / expected32.abs().clamp_min(1e-12)
    if actual.dtype == torch.float32:
        close = error <= 1e-5 + 1e-4 * expected32.abs()
    else:
        close = (error <= 1e-3 + 1e-2 * expected32.abs()) | (
            actual == torch.nextafter(expected, actual)
        )
    max_abs, max_rel, mismatch_count, passed = torch.stack(
        (
            error.amax().double(),
            relative.amax().double(),
            (actual != expected).sum().double(),
            (close & actual.isfinite() & expected.isfinite()).all().double(),
        )
    ).tolist()
    return {
        "max_abs": max_abs if math.isfinite(max_abs) else None,
        "max_rel": max_rel if math.isfinite(max_rel) else None,
        "mismatch_count": int(mismatch_count),
        "exact_match": mismatch_count == 0,
        "tol_pass": bool(passed),
    }


def _parity(
    op: str,
    modes: tuple[str, ...],
    args: argparse.Namespace,
    timed_keys: dict[str, dict[str, scans.Kernel]],
) -> dict:
    env = OPS[op][0]
    report = {}
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for dtype_name, dtype in DTYPES.items():
        for shape in (*MODEL_SHAPES, *EDGE_SHAPES):
            label, h, w, m, coarse, _ = shape
            # Bound input volume, allowing short scans to exercise production kernel choices.
            parity_batch = 64 * 56 * 56 // (h * w)
            batch = 2 if label in EDGE_LABELS else min(args.batch, parity_batch)
            if label == "56d" and op == "vscan_bwd":
                # Exercise the two-wave grouping even when timing a small batch.
                batch = max(batch, 8, scans._ceil(16 * sms, w * scans._ceil(m, 32)))
            for compact in _layouts(op, label, coarse, w):
                os.environ[env] = "off"
                run, plan = _case(op, shape, batch, dtype, compact=compact)
                expected = run()
                samples = {}
                record = report[f"{dtype_name}/{label}/compact={compact}"] = {
                    "batch": batch,
                    "modes": {},
                }
                for mode in modes:
                    os.environ[env] = mode
                    launch = plan()
                    if launch not in samples:
                        actual = expected if mode == "off" else run()
                        samples[launch] = {
                            "plan": _describe(launch),
                            "tensors": {
                                name: parity_stats(value, reference)
                                for name, value, reference in zip(
                                    TENSOR_NAMES[op],
                                    actual,
                                    expected,
                                    strict=True,
                                )
                            },
                        }
                    timed_key = (
                        timed_keys.get(label, {}).get(mode) if dtype_name == args.dtype else None
                    )
                    record["modes"][mode] = {
                        **samples[launch],
                        "timing_kernel_match": None
                        if timed_key is None
                        else launch.key == timed_key,
                    }
    return report


def _layouts(op: str, label: str, coarse: bool, width: int) -> tuple[bool, ...]:
    if not coarse or not op.endswith("_bwd"):
        return (False,)
    if op == "hscan_bwd" and width % 2:
        return (False,)
    return (False, True) if label in EDGE_LABELS else (True,)


def _screen(
    op: str,
    args: argparse.Namespace,
    modes: tuple[str, ...],
    flush_buffer: Tensor | None,
) -> dict:
    env = OPS[op][0]
    results = {mode: {} for mode in modes}
    totals = dict.fromkeys(modes, 0.0)
    timed_keys = {}
    for shape in MODEL_SHAPES:
        label, _, width, _, coarse, calls = shape
        if label == "7d" and op.startswith("hscan"):
            calls = 3
        os.environ[env] = "off"
        (compact,) = _layouts(op, label, coarse, width)
        run, plan = _case(op, shape, args.batch, DTYPES[args.dtype], compact=compact)
        samples = {}
        timed_keys[label] = {}
        for mode in modes:
            os.environ[env] = mode
            launch = plan()
            timed_keys[label][mode] = launch.key
            if launch not in samples:
                ms = _time(run, args.warmup, args.iters, flush_buffer)
                samples[launch] = ms, _describe(launch)
            ms, details = samples[launch]
            totals[mode] += calls * ms
            results[mode][label] = {"median_ms": ms, "calls": calls, "plan": details}
        del run, plan
    winner = W43_MODES[op]
    for shapes in results.values():
        for label, sample in shapes.items():
            sample["delta_vs_off_ms"] = sample["median_ms"] - results["off"][label]["median_ms"]
            sample["delta_vs_w43_ms"] = sample["median_ms"] - results[winner][label]["median_ms"]
    diagnostics = _parity(op, modes, args, timed_keys) if args.parity else {}
    passes = {
        field: {
            mode: all(
                stats[field]
                for case in diagnostics.values()
                for stats in case["modes"][mode]["tensors"].values()
            )
            for mode in modes
        }
        if diagnostics
        else {}
        for field in ("exact_match", "tol_pass")
    }
    covered = (
        {
            mode: all(
                case["modes"][mode]["timing_kernel_match"] is not False
                for case in diagnostics.values()
            )
            for mode in modes
        }
        if diagnostics
        else {}
    )
    exact = {
        mode: passed and covered["off"] and covered[mode]
        for mode, passed in passes["exact_match"].items()
    }
    tolerated = {
        mode: passed and covered["off"] and covered[mode]
        for mode, passed in passes["tol_pass"].items()
    }
    return {
        "best_mode": best_mode(totals, exact),
        "best_tolerated_mode": best_mode(
            totals, tolerated if op in ("vscan_bwd", "hscan_bwd") else exact
        ),
        "timing_kernel_covered": covered,
        "weighted_ms": totals,
        "delta_vs_off_ms": {mode: ms - totals["off"] for mode, ms in totals.items()},
        "delta_vs_w43_ms": {mode: ms - totals[winner] for mode, ms in totals.items()},
        "parity_pass": passes["exact_match"],
        "parity_mismatches": {
            mode: [
                label
                for label, case in diagnostics.items()
                if not all(
                    stats["exact_match"] for stats in case["modes"][mode]["tensors"].values()
                )
            ]
            for mode in passes["exact_match"]
        },
        "tolerance_pass": passes["tol_pass"],
        "parity": diagnostics,
        "shapes": results,
    }


def _combined_projection(report: dict) -> dict[str, float]:
    winners = W43_MODES
    totals = {op: report[op]["weighted_ms"] for op in winners}
    projected = sum(totals[op][mode] for op, mode in winners.items())
    off = sum(times["off"] for times in totals.values())
    strict = off - totals["vscan_bwd"]["off"] + totals["vscan_bwd"]["strict"]
    return {
        "weighted_ms": projected,
        "delta_vs_off_ms": projected - off,
        "delta_vs_strict_ms": projected - strict,
    }


def _e2e_case(data: Inputs, coarse: bool, mode: str) -> tuple[Callable, Callable]:
    policy = scans._vscan_saved_policy(data.states, data.path, coarse, mode)

    def forward() -> tuple[Tensor, Tensor, Tensor | None]:
        if policy == "off":
            r, i = scans.vertical_forward(
                data.pole,
                data.states,
                data.gains,
                data.path,
                coarse,
            )
            return r, i, None
        return scans._vertical_forward_saved(
            data.pole,
            data.states,
            data.gains,
            data.path,
            coarse,
            1e-6,
            planar=False,
            precise=True,
            separate=True,
            trajectory_policy=policy,
        )

    def backward(saved: Tensor | None) -> list[Tensor]:
        return scans.vertical_backward(
            data.pole,
            data.states,
            data.gains,
            data.path,
            data.gr,
            data.gi,
            coarse=coarse,
            compact=coarse,
            saved_features=saved,
        )

    return forward, backward


def _e2e_sample(
    data: Inputs,
    coarse: bool,
    mode: str,
    args: argparse.Namespace,
    flush: Tensor | None,
) -> dict:
    forward, backward = _e2e_case(data, coarse, mode)

    def full_step() -> list[Tensor]:
        _, _, saved = forward()
        return backward(saved)

    torch.cuda.reset_peak_memory_stats()
    total_ms = _time(full_step, args.warmup, args.iters, flush, graph=True)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    _, _, trajectory = forward()
    return {
        "fwd_ms": _time(forward, args.warmup, args.iters, flush, graph=True),
        "bwd_ms": _time(lambda: backward(trajectory), args.warmup, args.iters, flush, graph=True),
        "total_ms": total_ms,
        "peak_vram_gib": peak,
        "selected_saved_path": trajectory is not None,
        "saved_traj_gib": trajectory.nbytes / 1024**3 if trajectory is not None else 0.0,
        "cuda_graph_ok": True,
    }


def _e2e_parity(shapes: tuple, modes: tuple[str, ...], batch: int) -> dict:
    report = {}
    for name, dtype in DTYPES.items():
        cases = {}
        for label, h, w, m, coarse, _ in shapes:
            inputs = make_inputs(batch, h, w, m, coarse, dtype)
            outputs = {}
            expected = None
            for mode in modes:
                forward, backward = _e2e_case(inputs, coarse, mode)
                real, imag, trajectory = forward()
                values = [real, imag, *backward(trajectory)]
                if expected is None:
                    expected = values
                names = ("real", "imag", *TENSOR_NAMES["vscan_bwd"])
                stats = {}
                for key, got, want in zip(names, values, expected, strict=True):
                    metrics = parity_stats(got, want)
                    delta = torch.linalg.vector_norm(got.float() - want.float())
                    rel_l2 = (
                        delta / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)
                    ).item()
                    metrics["rel_l2"] = rel_l2 if math.isfinite(rel_l2) else None
                    metrics["tol_pass"] &= rel_l2 < (1e-2 if name == "bf16" else 1e-3)
                    stats[key] = metrics
                outputs[mode] = stats
            cases[label] = outputs
        report[name] = cases
    return report


def _screen_e2e(args: argparse.Namespace, modes: tuple[str, ...], flush: Tensor | None) -> dict:
    check_auto = "auto_vram" in modes
    modes = tuple(mode for mode in modes if mode != "auto_vram")
    auto = {}
    saved_weighted = dict.fromkeys(modes, 0.0)
    weighted = dict.fromkeys(modes, 0.0)
    fwd_weighted = dict.fromkeys(modes, 0.0)
    bwd_weighted = dict.fromkeys(modes, 0.0)
    results = {}
    for label, h, w, m, coarse, weight in MODEL_SHAPES:
        if check_auto:
            device = torch.device("cuda", torch.cuda.current_device())
            # Policy only inspects tensor metadata. Expanded placeholders avoid allocating full inputs.
            placeholder = torch.empty((), device=device, dtype=DTYPES[args.dtype])
            states = placeholder.expand(4, args.batch, h, w, m)
            path = [placeholder.expand(m, 8, 8)]
            free, total = torch.cuda.mem_get_info(device)
            auto[label] = {
                "resolved_mode": scans._vscan_saved_policy(states, path, coarse, "auto_vram"),
                "free_gib": free / 1024**3,
                "total_gib": total / 1024**3,
            }
        data = make_inputs(args.batch, h, w, m, coarse, DTYPES[args.dtype])
        cases = {}
        for mode in modes:
            print(f"timing vscan {label} {mode}", flush=True)
            result = _e2e_sample(data, coarse, mode, args, flush)
            cases[mode] = result
            saved_weighted[mode] += weight * result["saved_traj_gib"]
            weighted[mode] += weight * result["total_ms"]
            fwd_weighted[mode] += weight * result["fwd_ms"]
            bwd_weighted[mode] += weight * result["bwd_ms"]
            result["delta_vs_off_ms"] = result["total_ms"] - cases["off"]["total_ms"]
        results[label] = cases
        del data
    report = {
        "shapes": results,
        "weighted_ms": weighted,
        "weighted_total_ms": weighted,
        "auto_vram_decisions": auto,
        "total_saved_traj_gib": saved_weighted,
        "call_weights": {label: weight for label, *_, weight in MODEL_SHAPES},
        "weighted_fwd_ms": fwd_weighted,
        "weighted_bwd_ms": bwd_weighted,
        "weighted_delta_vs_off_ms": {mode: ms - weighted["off"] for mode, ms in weighted.items()},
    }
    if args.parity:
        edge = (("17x19x160d", 17, 19, 160, False, 1), ("31x23x96c", 31, 23, 96, True, 1))
        report["parity"] = _e2e_parity((*MODEL_SHAPES, *edge), modes, args.parity_batch)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", help="Comma-separated policies for vscan_bwd or vscan_e2e.")
    parser.add_argument(
        "--ops",
        default="vscan_bwd",
        help="Comma-separated operations, including vscan_e2e.",
    )
    parser.add_argument("--all-ops", action="store_true", help="Screen all four scan operations.")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--flush-l2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear the L2 working set before each timed iteration (default: enabled).",
    )
    parser.add_argument(
        "--parity", action="store_true", help="Collect BF16 and FP16 parity diagnostics."
    )
    parser.add_argument("--parity-batch", type=int, default=4)
    parser.add_argument(
        "--out", "--json-out", type=Path, help="Write JSON to this path instead of stdout."
    )
    args = parser.parse_args()
    ops = (
        tuple(OPS)
        if args.all_ops
        else tuple(dict.fromkeys(token.strip() for token in args.ops.split(",")))
    )
    available = (*OPS, "vscan_e2e")
    if set(ops).difference(available):
        parser.error(f"--ops must contain names from {', '.join(available)}")
    requested = (
        tuple(token.strip() for token in args.modes.split(",")) if args.modes else scans.SGS_MODES
    )
    e2e_modes = ("off", "saved_fused", "saved_split", "auto_vram")
    if args.modes and "vscan_e2e" in ops:
        if "vscan_bwd" in ops:
            parser.error("--modes applies to one VScan operation at a time")
        if set(requested).difference(e2e_modes):
            parser.error("--modes must contain valid LNET_SCAN_VSCAN_SAVED tokens")
        e2e_modes = tuple(dict.fromkeys(("off", *requested)))
    elif set(requested).difference(scans.SGS_MODES):
        parser.error("--modes must contain valid LNET_SCAN_SGS tokens")
    if min(args.batch, args.parity_batch) < 1 or args.iters < 1 or args.warmup < 0:
        parser.error("--batch and --iters must be positive and --warmup cannot be negative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    os.environ["LNET_VSCAN_DECOUPLED"] = "off"
    os.environ["LNET_SCAN_VSCAN_SAVED"] = "off"
    started = perf_counter()
    torch.manual_seed(141)
    report = {"batch": args.batch, "dtype": args.dtype, "warmup": args.warmup, "iters": args.iters}
    flush_bytes = 0
    if args.flush_l2:
        flush_bytes = max(
            (128 if "vscan_e2e" in ops else 64) * 1024**2,
            2 * scans._l2_size(torch.cuda.current_device()),
        )
    report.update(flush_l2=args.flush_l2, flush_bytes=flush_bytes)
    with torch.inference_mode():
        flush_buffer = (
            torch.empty(flush_bytes, device="cuda", dtype=torch.uint8) if flush_bytes else None
        )
        for op in ops:
            if op == "vscan_e2e":
                report["scan_flags"] = {env: os.environ.get(env, "off") for env, _ in OPS.values()}
                report[op] = _screen_e2e(args, e2e_modes, flush_buffer)
                continue
            for env, _ in OPS.values():
                os.environ[env] = "off"
            modes = OPS[op][1]
            if op == "vscan_bwd" and args.modes:
                modes = tuple(dict.fromkeys(("off", W43_MODES[op], *requested)))
            report[op] = _screen(op, args, modes, flush_buffer)
    if ops == ("vscan_bwd",):
        report.update(report["vscan_bwd"])
    if all(
        op in report and set(modes) <= report[op]["weighted_ms"].keys()
        for op, (_, modes) in OPS.items()
    ):
        report["combined_projection"] = _combined_projection(report)
    report["elapsed_seconds"] = perf_counter() - started
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is None:
        print(output, end="")
    else:
        args.out.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
