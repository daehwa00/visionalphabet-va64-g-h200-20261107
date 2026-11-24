"""Instrument the matched Task 85 benchmark without modifying any model checkout."""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, PLC0415, SLF001

import functools
import json
import os
import runpy
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def swiglu_breakdown(profile):
    """Assign kernels to train phases by the CPU launch correlation ID."""
    trace_path = Path(os.environ["VA_T109_TRACE"])
    profile.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())["traceEvents"]
    trace_path.unlink()
    backward = next(event for event in trace if event.get("name") == "va_t101_backward")
    start = backward["ts"]
    end = start + backward["dur"]
    launches = {
        event["args"]["correlation"]: event["ts"]
        for event in trace
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "correlation" in event.get("args", {})
    }
    groups = {}
    for event in trace:
        if event.get("cat") != "kernel":
            continue
        name = event["name"]
        if "_swiglu_vec2d" in name:
            kind = "vector_swiglu"
        elif "input_norm" in name:
            kind = "input_norm"
        elif "_rmsnorm_" in name:
            kind = "rmsnorm"
        elif "triton_w28c_packed_swiglu" in name:
            kind = "packed"
        elif "mul_silu_split" in name or "mul_sigmoid_silu_split" in name:
            kind = "inductor_silu"
        else:
            continue
        launched = launches.get(event.get("args", {}).get("correlation"))
        phase = (
            "unknown" if launched is None or launched >= end
            else "bwd" if launched >= start else "fwd"
        )
        row = groups.setdefault(f"{kind}_{phase}", {"count": 0, "ms": 0.0})
        row["count"] += 1
        row["ms"] += event["dur"] / 1000
    return groups


def instrument(step, original, **options):  # noqa: C901, PLR0915
    import torch

    from lnet import factorized_reader_kernels as reader
    from lnet import fused_stem_ops as stem
    from lnet import general_cuda_scans as scans

    scan_dispatch = []
    reader_dispatch = []
    modules = {}
    profiles = []
    stem_launches = []
    launch_stem = stem._launch_tensor_core_stem
    backward = torch.Tensor.backward

    def timed_stem(x, *args, **kwargs):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = launch_stem(x, *args, **kwargs)
        end.record()
        stem_launches.append((int(x.shape[1]), start, end))
        return result

    profile = torch.profiler.profile
    launch_reader = reader._launch_reader_cuda
    dispatch_scan = scans._dispatch

    def profiling(**kwargs):
        kwargs["activities"] = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        active = profile(**kwargs)
        profiles.append(active)
        return active

    def scan(source, launch, args):
        dispatch_scan(source, launch, args)
        if launch is not None:
            key = launch.key
            scan_dispatch.append(
                (
                    *source.shape[-3:],
                    key.vertical,
                    key.backward,
                    key.coarse,
                    *(
                        getattr(key, field, False)
                        for field in (
                            "compact",
                            "vector_weights",
                            "register_path",
                            "register_batch2",
                            "horizontal_unroll8",
                            "mma_path",
                            "backward_batch2",
                            "horizontal_forward_unroll",
                            "backward_batch_tail",
                        )
                    ),
                )
            )

    def reader_plan(name, selector):
        @functools.wraps(selector)
        def select(projected, real, *args, **kwargs):
            picked = selector(projected, real, *args, **kwargs)
            if picked is not None:
                if name in {"_wave27b_weight", "_hopper_reader_weight"}:
                    vec, geometry, module, workers = picked
                    plan = (vec, *geometry, workers)
                else:
                    plan, module = picked
                modules[id(module)] = {
                    "selector": name,
                    "shape": list(projected.shape),
                    "plan": plan._asdict() if hasattr(plan, "_asdict") else list(plan),
                }
            return picked

        return select

    def reader_launch(module, operation, grid, tensors, sizes, **kwargs):
        launch_reader(module, operation, grid, tensors, sizes, **kwargs)
        partial_mib = tensors[-1].nbytes / 1024**2 if operation == "dweight" else None
        reader_dispatch.append(
            {
                "operation": operation,
                "grid": grid,
                "threads": kwargs.get("threads"),
                "partial_mib": partial_mib,
                **modules.get(id(module), {}),
            }
        )

    def marked_backward(tensor, *args, **kwargs):
        with torch.profiler.record_function("va_t101_backward"):
            return backward(tensor, *args, **kwargs)

    def marked_step():
        with (
            torch.profiler.record_function("va_t93_step"),
            patch.object(torch.Tensor, "backward", marked_backward),
        ):
            return step()

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.profiler, "profile", profiling))
        stack.enter_context(patch.object(scans, "_dispatch", scan))
        stack.enter_context(patch.object(stem, "_launch_tensor_core_stem", timed_stem))
        stack.enter_context(patch.object(reader, "_launch_reader_cuda", reader_launch))
        for name in (
            "_hopper_reader_spatial", "_hopper_reader_weight", "_wave25b_spatial",
            "_wave25b_weight", "_operation_vector_plan",
        ):
            selector = getattr(reader, name, None)
            if selector is not None:
                stack.enter_context(patch.object(reader, name, reader_plan(name, selector)))
        result = original(marked_step, **options)

    fields = (
        "height",
        "width",
        "modes",
        "vertical",
        "backward",
        "coarse",
        "compact",
        "vector_weights",
        "register_path",
        "register_batch2",
        "horizontal_unroll8",
        "mma_path",
        "backward_batch2",
        "horizontal_forward_unroll",
        "backward_batch_tail",
    )
    if "kernels" in result:
        result.pop("top_kernels", None)
        annotations = [
            row for row in result["kernels"]
            if row["name"] in {"va_t93_step", "va_t101_backward"}
            or row["name"].startswith(("## Call CompiledFxGraph ", "Optimizer.step#"))
        ]
        result["cuda_annotations"] = annotations
        result["kernels"] = [row for row in result["kernels"] if row not in annotations]
        groups = Counter()
        for row in result["kernels"]:
            name = row["name"]
            if name.startswith("reader_"):
                operation = next(
                    (op for op in ("reduce", "dinput", "dweight", "forward") if op in name),
                    name,
                )
                groups[f"reader_{operation}"] += row["self_cuda_ms"]
            elif name.startswith("_nr4_correlation"):
                groups["_nr4_correlation_kernel"] += row["self_cuda_ms"]
        result["major_kernel_ms"] = dict(groups)
        result["self_cuda_ms_by_bucket"]["other"] -= sum(
            row["self_cuda_ms"] for row in annotations
        )
        result["dispatches_by_profiler_bucket"]["other"] -= sum(
            row["count"] for row in annotations
        )
    result["scan_dispatch"] = [
        {**dict(zip(fields, key, strict=True)), "count": count}
        for key, count in Counter(scan_dispatch).items()
    ]
    result["reader_dispatch"] = [
        {**json.loads(key), "count": count}
        for key, count in Counter(json.dumps(row) for row in reader_dispatch).items()
    ]
    result["stem_tensor_core_event_ms"] = {
        f"stage{i}": sum(start.elapsed_time(end) for channels, start, end in stem_launches
                         if channels == expected)
        for i, expected in enumerate((3, 32))
    }
    try:
        result["swiglu_breakdown"] = swiglu_breakdown(profiles[0])
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        result["swiglu_trace_error"] = repr(exc)
    events = profiles[0].events()
    mark = next(event.time_range for event in events if event.name == "va_t93_step")
    result["step_synchronizations"] = dict(
        Counter(
            event.name
            for event in events
            if "synchronize" in event.name.lower()
            and mark.start <= event.time_range.start < mark.end
        )
    )
    return result


def main():
    command = sys.argv[1]
    bench = Path(os.environ["VA_T93_BENCH"])
    namespace = runpy.run_path(str(bench))["main"].__globals__
    orthogonal = []
    if command == "train-arm":
        build_model = namespace["build_model"]
        measure = namespace["measure"]
        profile_step = namespace["profile_step"]
        kernel_bucket = namespace["kernel_bucket"]

        def build(variant):
            model = build_model(variant)
            for name, module in model.named_modules():
                if type(module).__name__ == "CaptureSafeMatrixExpOrthogonal":
                    orthogonal.append(
                        {
                            "name": name,
                            "degree": module.taylor_degree,
                            "scaling_steps": module.scaling_steps,
                            "base_dtype": str(module.base.dtype),
                            "compute_dtype": str(module.compute_dtype),
                        }
                    )
            return model

        def measured(*args, **kwargs):
            metrics, losses = measure(*args, **kwargs)
            metrics["loss_samples"] = [float(loss) for loss in losses]
            metrics["orthogonal"] = orthogonal
            return metrics, losses

        def bucket(name):
            if name.startswith("conv_stem_"):
                return "fused_stem"
            if name.startswith(("_rmsnorm_", "_swiglu_vec2d_")):
                return "triton"
            return kernel_bucket(name)

        namespace.update(
            build_model=build,
            measure=measured,
            kernel_bucket=bucket,
            profile_step=lambda step, **kw: instrument(step, profile_step, **kw),
        )
    namespace["main"]()


if __name__ == "__main__":
    main()
