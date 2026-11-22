"""Run isolated synthetic ImageNet-100 benchmarks on the provisioned GPU."""
# ruff: noqa: PLC0415, SLF001

from __future__ import annotations

import argparse
import ctypes
import json
import re
import statistics
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

GIB = 1024**3
VARIANTS = ("17-live-complex-rms-s03", "17-e2-multiscale-s3s4-live-rms-v2")
COUNTS = ("passed", "skipped", "failed", "errors", "xfailed", "xpassed", "deselected")


def write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def wrap_pytest(args: argparse.Namespace) -> dict:
    log = args.log.read_text(encoding="utf-8", errors="replace")
    lines = log.splitlines()
    summary = next((line for line in reversed(lines) if re.search(r"\bin [\d.]+s\b", line)), "")
    counts = {
        name: sum(int(n) for n in re.findall(rf"\b(\d+) {name.rstrip('s')}s?\b", summary))
        for name in COUNTS
    }
    duration = re.search(r"\bin ([\d.]+)s\b", summary)
    return {
        "status": "ok" if args.exit_code == 0 else "error",
        "exit_code": args.exit_code,
        "counts": counts if duration and any(counts.values()) else None,
        "duration_seconds": float(duration[1]) if duration else args.elapsed,
        "summary": summary,
        "tail": lines[-80:],
        "dynamo_tests": [line for line in lines if "test_dynamo_single_graph_on_cuda[" in line],
    }


def probe(_args: argparse.Namespace) -> dict:
    import torch
    import triton

    from lnet.cuda_nvrtc import _check_nvrtc, _include_dirs, _load_cuda_driver, _nvrtc
    from lnet.general_cuda_scans import _horizontal_cache_budget, device_limits

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    l2 = ctypes.c_int()
    status = _load_cuda_driver().cuDeviceGetAttribute(ctypes.byref(l2), 38, 0)
    if status:
        raise RuntimeError(f"CUDA L2 attribute returned {status}")
    nvrtc_major, nvrtc_minor = ctypes.c_int(), ctypes.c_int()
    _check_nvrtc(_nvrtc().nvrtcVersion(ctypes.byref(nvrtc_major), ctypes.byref(nvrtc_minor)))
    return {
        "nvrtc": [nvrtc_major.value, nvrtc_minor.value],
        "nvrtc_include_dirs": [str(path) for path in _include_dirs()],
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "capability": torch.cuda.get_device_capability(0),
        "vram_gib": props.total_memory / GIB,
        "l2_bytes": l2.value,
        "device_limits": device_limits(0),
        "horizontal_cache_budget": _horizontal_cache_budget(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
    }


def setup(batch: int) -> torch.Tensor:
    import torch

    torch.manual_seed(501)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    return torch.randn(batch, 3, 224, 224, device="cuda").contiguous(
        memory_format=torch.channels_last
    )


def build_model(variant: str) -> torch.nn.Module:
    import torch
    import va_model_builder as builder

    model = builder.build_dw_terminal_rmsnorm(
        builder.model_config(output_dim=100), factorized_reader=True
    )
    if variant == VARIANTS[1]:
        model = builder.add_multiscale_readout(model)
    return model.to(device="cuda", memory_format=torch.channels_last)


def measure(step: Callable, batch: int, *, warmup: int, count: int) -> tuple[dict, list]:
    import torch

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings, outputs = [], []
    for _ in range(count):
        start = time.perf_counter()
        outputs.append(step())
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    median = statistics.median(timings)
    p25, _, p75 = statistics.quantiles(timings, n=4, method="inclusive")
    return {
        "median_ms": median,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "p25_ms": p25,
        "p75_ms": p75,
        "images_per_second": batch * 1000 / median,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }, outputs


def kernel_bucket(name: str) -> str:
    if name == "scan":
        return "general_scan_shared"
    if name == "reduce_hscan_partials":
        return "general_hscan"
    if name == "reduce_vscan_partials":
        return "general_vscan"
    for prefix, bucket in (
        ("reader_", "factorized_reader"),
        ("stem_", "fused_stem"),
        ("triton", "triton"),
    ):
        if name.startswith(prefix):
            return bucket
    if re.search("gemm|cublas|cutlass", name, re.IGNORECASE):
        return "cublas_gemm"
    return "other"


def profile_step(step: Callable) -> dict:
    import torch

    from lnet import general_cuda_scans as scans

    # Both NVRTC scan entry points are called scan. CUDA event pairs around each
    # launch let us estimate the split without modifying the kernel source.
    launches = []
    original = scans._launch_kernel

    def track(source: torch.Tensor, key: scans.Kernel, grid: int, smem: int, args: tuple) -> None:
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        original(source, key, grid, smem, args)
        end.record()
        if not key.reduction:
            launches.append((key.vertical, start, end))

    scans._launch_kernel = track
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
    finally:
        scans._launch_kernel = original

    buckets = dict.fromkeys(
        (
            "general_hscan",
            "general_vscan",
            "general_scan_shared",
            "factorized_reader",
            "fused_stem",
            "cublas_gemm",
            "triton",
            "other",
        ),
        0.0,
    )
    kernels = []
    dispatches = dict.fromkeys(buckets, 0)
    for event in prof.key_averages():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        elapsed = event.self_device_time_total / 1000
        bucket = kernel_bucket(event.key)
        buckets[bucket] += elapsed
        dispatches[bucket] += event.count
        kernels.append({"name": event.key, "count": event.count, "self_cuda_ms": elapsed})

    event_ms = {
        axis: sum(start.elapsed_time(end) for v, start, end in launches if v == vertical)
        for axis, vertical in (("general_hscan", False), ("general_vscan", True))
    }
    event_total = sum(event_ms.values())
    if event_total:
        shared = buckets.pop("general_scan_shared")
        for axis, elapsed in event_ms.items():
            buckets[axis] += shared * elapsed / event_total
    return {
        "self_cuda_ms_by_bucket": buckets,
        "dispatches_by_profiler_bucket": dispatches,
        "general_cuda_scans": len(launches),
        "general_cuda_scans_verified": len(launches) > 0 and dispatches["general_scan_shared"] > 0,
        "scan_axis_cuda_event_ms": event_ms,
        "scan_axis_attribution": "Profiler shared scan time apportioned by CUDA event durations",
        "top_kernels": sorted(kernels, key=lambda row: -row["self_cuda_ms"])[:25],
    }


def train_arm(args: argparse.Namespace) -> dict:
    import torch
    import va_model_builder as builder

    inputs = setup(args.batch)
    labels = torch.arange(args.batch, device="cuda") % 100
    model = build_model(args.variant)
    compiled = torch.compile(model, mode="default")
    optimizer = builder.build_optimizer(
        model,
        {
            "learning_rate": 3e-3,
            "weight_decay": 0.05,
            "modal_learning_rate_multiplier": 1.0,
            "pole_geometry_learning_rate_multiplier": 1.0,
            "fused_optimizer": True,
        },
    )

    def backward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(compiled(inputs)[0], labels)
        loss.backward()
        return loss.detach()

    def update() -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def step() -> torch.Tensor:
        loss = backward()
        update()
        return loss

    result, losses = measure(step, args.batch, warmup=3, count=8)
    rms = model.classifier.affine.standardizer
    loss_values = torch.stack(losses)
    result.update(
        finite_losses=bool(loss_values.isfinite().all()),
        losses={"first": float(loss_values[0]), "last": float(loss_values[-1])},
        ema_weight_after=float(rms.ema_weight),
        running_ms_finite=bool(rms.running_ms.isfinite().all()),
    )
    backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    terminal = getattr(model.backbone, "terminal_s3", None)
    result.update(
        finite_gradients=bool(gradients) and all(bool(g.isfinite().all()) for g in gradients),
        grad_tensor_count=len(gradients),
        terminal_s3_nonzero_grads=(
            sum(p.grad is not None and bool(p.grad.count_nonzero()) for p in terminal.parameters())
            if terminal is not None
            else None
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    if args.profile:
        write_json(args.out, {**result, "status": "profiling"})
        result["profiler"] = profile_step(step)
    return result


def dynamo_check() -> dict:
    import torch

    from lnet.image_layers import StandardizedAffineModalHead

    results = {}
    for dim in (1024, 1280, 1536):
        head = StandardizedAffineModalHead(dim, 100).cuda()
        inputs = torch.randn(4, dim, device="cuda")
        results[dim] = {}
        for mode in ("train", "eval", "folded"):
            head.train(mode == "train")
            if mode == "folded":
                head.fold_standardizer_into_linear()
            graph = torch._dynamo.explain(head)(inputs)
            results[dim][mode] = [graph.graph_count, graph.graph_break_count]
    return results


def infer_fold(args: argparse.Namespace) -> dict:
    import torch

    from lnet.image_layers import fold_terminal_rms_for_inference

    inputs = setup(args.batch)
    model = build_model(args.variant)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Initialize evaluation scales from a clean synthetic batch.
        model(inputs)
    model.eval()

    def benchmark(module: torch.nn.Module) -> tuple[dict, torch.Tensor]:
        compiled = torch.compile(module, mode="default")

        def step() -> torch.Tensor:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return compiled(inputs)[0]

        result, outputs = measure(step, args.batch, warmup=3, count=8)
        return result, outputs[-1].float()

    with torch.no_grad():
        unfolded, expected = benchmark(model)
        head = model.classifier.affine
        descriptor = torch.randn(128, head.input_dim, device="cuda")
        expected_fp32 = head(descriptor)
        fold_terminal_rms_for_inference(model)
        folded, actual = benchmark(model)
        fp32_diff = float((expected_fp32 - head(descriptor)).abs().max())
        reloaded = fold_terminal_rms_for_inference(build_model(args.variant).eval())
        reloaded.load_state_dict(model.state_dict(), strict=True)
        reloaded_compiled = torch.compile(reloaded, mode="default")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reloaded_output = reloaded_compiled(inputs)[0]
        reload_diff = float((actual - reloaded_output.float()).abs().max())
        try:
            reloaded.classifier.affine.train()(descriptor)
        except RuntimeError as exc:
            guard = "cannot run folded StandardizedAffineModalHead" in str(exc)
        else:
            guard = False
    diff = float((expected - actual).abs().max())
    return {
        "unfolded": unfolded,
        "folded": folded,
        "max_abs_logit_diff": diff,
        "bf16_parity_passed": diff < 1e-5,
        "fp32_head_max_abs_logit_diff": fp32_diff,
        "strict_reload_max_abs_logit_diff": reload_diff,
        "strict_reload_passed": reload_diff == 0,
        "folded_train_guard": guard,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "train-arm", "infer-fold", "wrap-pytest", "dynamo"):
        sub = subs.add_parser(name)
        sub.add_argument("--out", required=True, type=Path)
        if name in {"train-arm", "infer-fold"}:
            sub.add_argument("--variant", required=True, choices=VARIANTS)
            sub.add_argument("--batch", required=True, type=int)
        if name == "train-arm":
            sub.add_argument("--profile", action="store_true")
        if name == "wrap-pytest":
            sub.add_argument("--log", required=True, type=Path)
            sub.add_argument("--exit-code", required=True, type=int)
            sub.add_argument("--elapsed", required=True, type=int)
    args = parser.parse_args()
    commands = {
        "probe": probe,
        "train-arm": train_arm,
        "infer-fold": infer_fold,
        "wrap-pytest": wrap_pytest,
        "dynamo": lambda _: {"graphs": dynamo_check()},
    }
    metadata = {key: getattr(args, key) for key in ("variant", "batch") if hasattr(args, key)}
    try:
        result = {"status": "ok", **commands[args.command](args)}
    except Exception as exc:  # noqa: BLE001 - preserve per-arm diagnostics
        partial = json.loads(args.out.read_text()) if args.out.exists() else {}
        oom = type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()
        result = {**partial, "status": "oom" if oom else "error", "error": traceback.format_exc()}
    write_json(args.out, {**metadata, **result})


if __name__ == "__main__":
    main()
