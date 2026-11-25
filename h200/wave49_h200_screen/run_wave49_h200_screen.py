"""Run Wave 49 Reader and vertical scan screens on a whole H200."""

from __future__ import annotations

# The isolated runner intentionally measures private launch metadata and imports CUDA lazily.
# ruff: noqa: ANN001, ANN201, ANN202, BLE001, C901, FBT003, PLC0415, PLR0915, S603, SLF001, T201
import argparse
import ctypes
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import traceback
import typing
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
SCAN_ENV = {
    "LNET_SCAN_SGS": "strict_ro",
    "LNET_SCAN_VFWD": "batch4",
    "LNET_SCAN_HSCAN": "bwd_opt",
}
ARMS = {
    "arm_off": ("off", "off", False),
    "arm_proven_floor": ("off", "proven_floor", False),
    "arm_dense_nhwc_bench": ("dense_nhwc", "proven_floor", True),
}
GIB = 1024**3
READER_BOOTSTRAP = (
    "import runpy, sys, typing; "
    "from typing_extensions import override; "
    "typing.override = getattr(typing, 'override', override); "
    "sys.argv.pop(0); runpy.run_path(sys.argv[0], run_name='__main__')"
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n")


def attempt(run: Callable) -> dict:
    start = perf_counter()
    try:
        result = run()
    except Exception:
        result = {"error": traceback.format_exc()}
    return {**result, "wall_seconds": perf_counter() - start}


def reader(output: Path, mode: str, timeout: float) -> dict:
    graph = mode == "graph"
    modes = ["off", "proven_floor", "dense_nhwc", "dense_nhwc_bench"]
    if graph:
        modes.append("dense_nhwc_bmm_bench")
    modes.append("auto_roofline_bench")
    target = output / f"reader_e2e_{mode}.json"
    command = [
        sys.executable,
        "-c",
        READER_BOOTSTRAP,
        str(ROOT / "scripts/bench_reader_e2e_micro.py"),
        "--mode",
        mode,
        "--modes",
        ",".join(modes),
        "--dtypes",
        "bf16,fp16" if graph else "bf16",
        "--warmup",
        "3",
        "--iterations",
        "10" if graph else "8",
        "--json-out",
        str(target),
    ]
    start = perf_counter()
    try:
        proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        run = {"error": f"{mode} Reader exceeded its {timeout:.2f}s time budget"}
        save(target, run)
        return run
    run = {"returncode": proc.returncode, "wall_seconds": perf_counter() - start}
    if target.is_file():
        payload = json.loads(target.read_text())
        run["parity_summary"] = payload["parity_summary"]
        save(target, payload)
    else:
        run["error"] = (proc.stdout + proc.stderr)[-8000:]
        save(target, run)
    return run


def kernel_bucket(name: str, context: str) -> str:
    kernel = name.lower()
    scope = context.lower()
    for scan in ("vscan_bwd", "vscan_fwd", "hscan_bwd", "hscan_fwd"):
        if scan in scope:
            return scan
    if "vscan_sgs" in kernel or "reduce_vscan_partials" in kernel:
        return "vscan_bwd"
    if "reduce_hscan_partials" in kernel:
        return "hscan_bwd"
    hints = (
        ("optimizer", ("adamw", "fused_adam", "foreach", "clip_grad", "optimizer")),
        (
            "cudnn_conv",
            ("cudnn", "scudnn", "xmma", "implicit_convolve", "aten::convolution_backward"),
        ),
        ("cublas_gemm", ("cublas", "gemm", "aten::mm", "aten::bmm", "aten::addmm")),
        (
            "factorized_reader",
            ("factorized_reader", "factorized_spatial", "reader_spatial", "reader_"),
        ),
        ("fused_stem", ("fused_stem", "stem_conv")),
        ("swiglu", ("swiglu",)),
        ("rmsnorm", ("rmsnorm", "rms_norm", "rms_match")),
        ("nr4", ("nr4", "non_redundant")),
        ("triton_other", ("triton",)),
    )
    for bucket, names in hints:
        if any(hint in kernel or hint in scope for hint in names):
            return bucket
    return "other"


@contextmanager
def scan_scopes(torch, scans) -> Iterator[None]:
    original = scans._launch_kernel

    def launch(source, key, grid, smem, args) -> None:
        axis = "vscan" if key.vertical or key.projection or key.projection_reduction else "hscan"
        direction = "bwd" if key.backward or key.reduction or key.projection_reduction else "fwd"
        with torch.profiler.record_function(f"wave49::{axis}_{direction}"):
            original(source, key, grid, smem, args)

    scans._launch_kernel = launch
    try:
        yield
    finally:
        scans._launch_kernel = original


def summarize_trace(events: list[dict]) -> dict:
    scopes = defaultdict(list)
    launches = {}
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X":
            continue
        if category in {"cpu_op", "user_annotation"}:
            scopes[event["pid"], event["tid"]].append(event)
        elif category in {"cuda_runtime", "cuda_driver"}:
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = event
    grouped = defaultdict(lambda: {"count": 0, "total_ms": 0.0, "buckets_ms": {}})
    buckets = dict.fromkeys(
        (
            "vscan_bwd",
            "vscan_fwd",
            "hscan_bwd",
            "hscan_fwd",
            "factorized_reader",
            "cudnn_conv",
            "cublas_gemm",
            "fused_stem",
            "swiglu",
            "rmsnorm",
            "nr4",
            "triton_other",
            "optimizer",
            "other",
        ),
        0.0,
    )
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        name = event["name"]
        launch = launches.get(event.get("args", {}).get("correlation"))
        context = ""
        if launch is not None:
            ts = launch["ts"]
            context = " ".join(
                item["name"]
                for item in scopes[launch["pid"], launch["tid"]]
                if item["ts"] <= ts <= item["ts"] + item["dur"]
            )
        bucket = kernel_bucket(name, context)
        ms = event["dur"] / 1_000
        buckets[bucket] += ms
        item = grouped[name]
        item["count"] += 1
        item["total_ms"] += ms
        subtotals = item["buckets_ms"]
        subtotals[bucket] = subtotals.get(bucket, 0.0) + ms
    if not grouped:
        raise RuntimeError("Kineto trace contains no CUDA kernels")
    kernels = sorted(
        ({"name": name, **item} for name, item in grouped.items()),
        key=lambda item: item["total_ms"],
        reverse=True,
    )
    return {
        "kernels": [item for item in kernels if item["total_ms"] >= 0.05],
        "unique_kernel_names": len(kernels),
        "kernel_launches": sum(item["count"] for item in kernels),
        "buckets_ms": buckets,
        "total_kernel_ms": sum(buckets.values()),
        "below_0p05_ms_total": sum(item["total_ms"] for item in kernels if item["total_ms"] < 0.05),
    }


def profile_step(torch, scans, step: Callable) -> dict:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with (
        scan_scopes(torch, scans),
        torch.profiler.profile(activities=activities, record_shapes=False) as prof,
    ):
        step()
        torch.cuda.synchronize()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        prof.export_chrome_trace(str(path))
        return summarize_trace(json.loads(path.read_text())["traceEvents"])


def fullmodel(torch, output: Path) -> dict:
    from va_model_builder import build_dw_terminal_rmsnorm

    from lnet import general_cuda_scans as scans

    os.environ.update(SCAN_ENV, LNET_FUSED_FACTORIZED_READER="1", LNET_READER_VEC="auto")
    generator = torch.Generator(device="cuda").manual_seed(169)
    inputs = torch.randn((1024, 3, 224, 224), device="cuda", generator=generator).contiguous(
        memory_format=torch.channels_last,
    )
    targets = torch.randint(100, (1024,), device="cuda", generator=generator)
    reference = {}
    report = {"batch": 1024, "dtype": "bf16", "arms": {}, "scan_env": SCAN_ENV}

    def arm(name: str) -> dict:
        execution, optimization, benchmark = ARMS[name]
        os.environ.update(LNET_READER_EXEC_MODE=execution, VA_READER_OPT_MODE=optimization)
        torch.backends.cudnn.benchmark = benchmark
        torch.manual_seed(169)
        model = (
            build_dw_terminal_rmsnorm(factorized_reader=True)
            .to(
                device="cuda",
                memory_format=torch.channels_last,
            )
            .train()
        )
        params = dict(model.named_parameters())
        optimizer = torch.optim.AdamW(params.values(), fused=True)

        def backward():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            return loss

        def step() -> None:
            backward()
            torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Check gradients before any optimizer updates and reuse this as warmup 1.
        loss = backward().item()
        grads = {
            key: param.grad.detach().float().cpu()
            for key, param in params.items()
            if param.grad is not None
        }
        finite = sum(bool(grad.isfinite().all()) for grad in grads.values())
        sanity = {
            "loss": loss if math.isfinite(loss) else None,
            "grad_count": len(grads),
            "finite_grads": finite,
        }
        if name == "arm_off":
            reference.update(loss=loss, grads=grads)
        elif "grads" in reference and finite == len(grads):
            baseline = reference["grads"]
            if grads.keys() == baseline.keys():
                pairs = [(grad, baseline[key]) for key, grad in grads.items()]
                difference = sum(float((grad - ref).square().sum()) for grad, ref in pairs)
                scale = sum(float(ref.square().sum()) for _, ref in pairs)
                rel_l2 = math.sqrt(difference / max(1e-30, scale))
                loss_error = abs(loss - reference["loss"])
                if math.isfinite(rel_l2) and math.isfinite(loss_error):
                    sanity.update(
                        loss_abs_error=loss_error,
                        gradient_rel_l2=rel_l2,
                        gradient_max_abs=max(
                            float((grad - ref).abs().max()) for grad, ref in pairs
                        ),
                        parity_pass=finite == 193 and loss_error <= 0.02 and rel_l2 <= 0.04,
                    )
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(4):
            start = perf_counter()
            step()
            torch.cuda.synchronize()
            times.append((perf_counter() - start) * 1_000)
        peak = torch.cuda.max_memory_allocated() / GIB
        profile = attempt(lambda: profile_step(torch, scans, step))
        if name == "arm_proven_floor":
            profile.pop("kernels", None)
        return {
            "step_ms": statistics.median(times),
            "samples_ms": times,
            "peak_allocated_gib": peak,
            "sanity": sanity,
            "profile": profile,
        }

    for name in ARMS:
        report["arms"][name] = attempt(partial(arm, name))
        torch.cuda.empty_cache()
        save(output / "fullmodel_unfiltered.json", report)
    return report


def scan_inputs(torch, batch: int, height: int, width: int):
    shape = (batch, height, width, 160)
    dtype = torch.bfloat16
    pole = [torch.full((160,), value, device="cuda") for value in (0.73, 0.13, 0.21, 0.09)]
    states = torch.randn((4, *shape), dtype=dtype, device="cuda").mul_(0.1)
    gains = [1 + torch.rand(2, size, 160, device="cuda") for size in (width, height)]
    path = [
        torch.randn(size, device="cuda", dtype=dtype).mul_(0.125)
        for size in (
            (160, 8, 8),
            (160, 8),
            (160, 2, 8),
            (160, 2),
        )
    ]
    return shape, pole, states, gains, path


def elapsed(torch, run: Callable, flush) -> dict:
    for _ in range(3):
        run()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(8)
    ]
    for start, end in events:
        start.record()
        end.record()
    torch.cuda.synchronize()
    for start, end in events:
        flush.add_(1)
        start.record()
        run()
        end.record()
    events[-1][1].synchronize()
    samples = [start.elapsed_time(end) for start, end in events]
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def describe_launch(torch, scans, launch) -> dict:
    device = torch.cuda.current_device()
    driver, function = scans._function(device, launch.key)
    registers = ctypes.c_int()
    status = driver.cuFuncGetAttribute(ctypes.byref(registers), 4, function)
    key = asdict(launch.key)
    key["dtype"] = str(key["dtype"])
    return {
        "plan": launch.plan._asdict(),
        "key": key,
        "workers": launch.workers,
        "partials": launch.partials,
        "registers_per_thread": registers.value if status == 0 else None,
        "active_warps_per_sm": scans._resident(device, launch.key, launch.plan.smem)
        * launch.key.warps,
    }


def scan_case(torch, scans, batch: int, height: int, width: int, flush) -> dict:
    shape, pole, states, gains, path = scan_inputs(torch, batch, height, width)
    result = {
        "shape": shape,
        "columns": batch * width,
        "stride_h_bytes": states.stride(2) * states.element_size(),
    }

    def measure(op: str) -> dict:
        pure, backward = op.startswith("pure"), op.endswith("bwd")
        args = (pole, states, gains, path)
        if pure and backward:
            features = torch.randn(
                (8, batch * height * width, 160), dtype=states.dtype, device=states.device
            )

            def run():
                return scans._vscan(*args, False, 1e-6, False, features=features, pure=True)
        elif pure:

            def run():
                return scans._vscan(*args, False, 1e-6, True, pure=True)
        elif backward:
            grad = torch.randn((2, *shape), dtype=states.dtype, device=states.device)
            real, imag = grad.unbind(0)

            def run():
                return scans.vertical_backward(*args, real, imag, False)
        else:

            def run():
                return scans.vertical_forward_packed(*args, False, 1e-6, planar=True)

        original = scans._dispatch
        launches = []

        def capture(source, launch, args) -> None:
            launches.append(launch)
            original(source, launch, args)

        scans._dispatch = capture
        try:
            run()
        finally:
            scans._dispatch = original
        timing = elapsed(torch, run, flush)
        return {
            **timing,
            "launches": [describe_launch(torch, scans, launch) for launch in launches],
        }

    for op in ("vscan_fwd", "pure_vscan_fwd", "vscan_bwd", "pure_vscan_bwd"):
        result[op] = attempt(partial(measure, op))
    return result


def command_result(command: list[str], timeout: int = 20) -> dict:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return {
            "returncode": None,
            "timeout_seconds": timeout,
            "stdout": (error.stdout or b"").decode(errors="replace"),
            "stderr": (error.stderr or b"").decode(errors="replace"),
        }
    else:
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def ncu_probe() -> dict:
    result = {"which": command_result(["which", "ncu"])}
    ncu = shutil.which("ncu")
    if ncu:
        metrics = (
            "sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,"
            "dram__bytes_write.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
        )
        result["version"] = command_result([ncu, "--version"])
        result["smoke"] = command_result(
            [
                ncu,
                "--kernel-name-base",
                "demangled",
                "--kernel-name",
                "regex:^(vscan_sgs|general_vscan|scan)$",
                "--launch-skip",
                "0",
                "--launch-count",
                "1",
                "--metrics",
                metrics,
                sys.executable,
                str(Path(__file__).resolve()),
                "--ncu-target",
            ]
        )
    return result


def stride(torch, output: Path) -> dict:
    from lnet import general_cuda_scans as scans

    os.environ.update(SCAN_ENV, LNET_VSCAN_DECOUPLED="off")
    torch.manual_seed(169)
    flush = torch.zeros(128 * 1024**2 // 4, dtype=torch.float32, device="cuda")
    report = {"scan_env": SCAN_ENV, "path_hidden": 8, "cases": {}, "transposes": {}}
    for height in (56, 28, 14):
        widths = (height, 1, 2) if height == 56 else (height, 1)
        for width in widths:
            name = f"{height}d_w{width}"
            report["cases"][name] = attempt(
                partial(
                    scan_case,
                    torch,
                    scans,
                    1024 * height // width,
                    height,
                    width,
                    flush,
                )
            )
            save(output / "vscan_stride_probe.json", report)
        for planes in (4, 2):

            def transpose(height: int, planes: int) -> dict:
                source = torch.randn(
                    (planes, 1024, height, height, 160), device="cuda", dtype=torch.bfloat16
                )
                return elapsed(torch, lambda: source.permute(0, 1, 3, 2, 4).contiguous(), flush)

            report["transposes"][f"{height}d_{planes}planes"] = attempt(
                partial(transpose, height, planes)
            )
    report["ncu"] = attempt(ncu_probe)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("wave49_results"))
    parser.add_argument("--ncu-target", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / part) for part in ("src", "scripts")]

    import torch
    from typing_extensions import override

    if not hasattr(typing, "override"):
        typing.override = override

    if args.ncu_target:
        from lnet import general_cuda_scans as scans

        os.environ.update(SCAN_ENV)
        shape, pole, states, gains, path = scan_inputs(torch, 8, 56, 56)
        real, imag = torch.randn((2, *shape), device="cuda", dtype=states.dtype).unbind(0)
        scans.vertical_backward(pole, states, gains, path, real, imag, False)
        torch.cuda.synchronize()
        return

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    if "H200" not in props.name or props.total_memory < 120 * GIB:
        raise RuntimeError(f"Expected a whole H200, found {props}")
    meta = {
        "started_utc": utc(),
        "torch": torch.__version__,
        "python": sys.version,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "stages": {},
    }
    stages = meta["stages"]
    reader_started = perf_counter()
    for mode in ("graph", "eager"):
        print(f"Reader {mode}", flush=True)
        timeout = 100 if mode == "graph" else max(1, 120 - (perf_counter() - reader_started))
        stages[f"reader_{mode}"] = attempt(partial(reader, output, mode, timeout))
    for name, function in (("fullmodel_unfiltered", fullmodel), ("vscan_stride_probe", stride)):
        print(name, flush=True)
        result = attempt(partial(function, torch, output))
        path = output / f"{name}.json"
        if "error" in result and path.is_file():
            result = {**json.loads(path.read_text()), **result}
        save(path, result)
        stages[name] = {key: result[key] for key in ("wall_seconds", "error") if key in result}
    meta["finished_utc"] = utc()
    save(output / "meta.json", meta)


if __name__ == "__main__":
    main()
