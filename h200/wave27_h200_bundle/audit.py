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
                        )
                    ),
                )
            )

    def reader_plan(name, selector):
        @functools.wraps(selector)
        def select(projected, real, *args, **kwargs):
            picked = selector(projected, real, *args, **kwargs)
            if picked is not None:
                if name == "_wave27b_weight":
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

    def marked_step():
        with torch.profiler.record_function("va_t93_step"):
            return step()

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.profiler, "profile", profiling))
        stack.enter_context(patch.object(scans, "_dispatch", scan))
        stack.enter_context(patch.object(stem, "_launch_tensor_core_stem", timed_stem))
        stack.enter_context(patch.object(reader, "_launch_reader_cuda", reader_launch))
        for name in ("_wave27b_spatial", "_wave27b_weight", "_wave25b_spatial", "_wave25b_weight",
                     "_operation_vector_plan"):
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
    )
    if "kernels" in result:
        result.pop("top_kernels", None)
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
            return "fused_stem" if name.startswith("conv_stem_") else kernel_bucket(name)

        namespace.update(
            build_model=build,
            measure=measured,
            kernel_bucket=bucket,
            profile_step=lambda step, **kw: instrument(step, profile_step, **kw),
        )
    namespace["main"]()


if __name__ == "__main__":
    main()
