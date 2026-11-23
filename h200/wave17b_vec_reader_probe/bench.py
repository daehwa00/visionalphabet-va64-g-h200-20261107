"""Time the dispatched reader kernels with CUDA Graphs and check reference parity."""

# The probe intentionally calls private CUDA dispatch and graph-capture interfaces.
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, FBT002, SLF001
import argparse
import importlib.util
import itertools
import json
import os
import statistics
import sys
from pathlib import Path

import torch

from lnet import factorized_reader_kernels as reader

STANDARD = (
    (56, 56, 160, 4), (28, 28, 320, 4), (14, 14, 640, 4), (7, 7, 640, 4),
    (128, 192, 160, 4), (96, 128, 320, 4), (512, 768, 160, 4),
)
PROBE_EDGES = (1, 2, 3, 4, 6, 17, 30, 33, 34, 48)


def inputs(batch, shape, dtype, strict=False):
    h, w, m, r = shape
    options = {"device": "cuda", "dtype": dtype}
    projected = torch.randn((batch, h, w, m * r * (2 if strict else 1)), **options) * 0.2
    grad = torch.randn((batch, h, w, 2 * m), **options) * 0.2
    real, imag = (torch.randn((2, m, r, 3, 3), **options) * 0.2).unbind()
    return projected, grad, real, imag


def graph_time(launches, module, samples):
    launch = module._launch_reader_cuda

    def replay():
        for args, kwargs in launches:
            stream = module._reader_stream(args[0], args[3][0].device)
            launch(*args, **{**kwargs, "stream": stream})

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay()
    for _ in range(4):
        graph.replay()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(samples)]
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    raw = [start.elapsed_time(end) for start, end in events]
    med = statistics.median(raw)
    q1, _, q3 = statistics.quantiles(raw, n=4, method="inclusive")
    logical_bytes = sum(t.numel() * t.element_size() for args, _ in launches for t in args[3])
    return {"samples_ms": raw, "median_ms": med, "iqr_ms": q3 - q1,
            "logical_bytes": logical_bytes, "effective_gb_s": logical_bytes / (med * 1e6),
            "launches": [(args[1], args[2], kwargs.get("threads", module._DEFAULT_THREADS))
                         for args, kwargs in launches]}


def record_launches(module, p, g, real, imag):
    calls = []
    launch = module._launch_reader_cuda

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return launch(*args, **kwargs)

    module._launch_reader_cuda = capture
    try:
        module._packed_spatial(p, real, imag, strict_complex=False)
        module._packed_spatial_backward(g, p, real, imag, strict_complex=False)
    finally:
        module._launch_reader_cuda = launch
    return calls


def time_case(batch, shape, dtype, samples, modules):
    h, w, m, r = shape
    # Leave room for all operation buffers and partial reductions.
    element_bytes = torch.empty((), dtype=dtype).element_size()
    free, _ = torch.cuda.mem_get_info()
    batch = min(batch, max(1, int(free * 0.48) // (h * w * m * (2 * r + 4) * element_bytes)))
    p, g, real, imag = inputs(batch, shape, dtype)
    result = {"shape": (batch, h, w, m, r), "dtype": str(dtype), "arms": {}}
    for name, module in modules:
        historical_arms = {"baseline": "1", "wave17b": "auto", "wave18b": "auto"}
        os.environ["LNET_READER_VEC"] = historical_arms.get(name, name)
        calls = record_launches(module, p, g, real, imag)
        groups = {
            "forward": calls[:1], "dinput": calls[-1:], "dweight": calls[1:-1], "total": calls,
        }
        result["arms"][name] = {
            key: graph_time(value, module, samples) for key, value in groups.items()
        }
        if module is reader:
            result.setdefault("plans", {})[name] = {}
            for operation in ("forward", "dinput", "dweight"):
                gradient = None if operation == "forward" else g
                selected = reader._operation_vector_plan(
                    p, real, gradient, operation, strict=False,
                )
                if selected:
                    geometry, compiled = selected
                    result["plans"][name][operation] = {
                        **geometry._asdict(),
                        "resident_blocks": reader._reader_resident(
                            compiled, operation, geometry.warps,
                        ),
                    }
        del calls, groups
    return result


def load_baseline(path, arm="wave14a"):
    if path is None:
        return None
    name = f"lnet._{arm}_reader"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    # Custom-op names are process-global, so give Wave 14A independent registrations.
    code = path.read_text().replace('"va_reader::', f'"va_reader_{arm}::')
    exec(compile(code, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def compute(module, p, g, real, imag, strict):
    out = module._packed_spatial(p, real, imag, strict)
    dp, dw = module._packed_spatial_backward(g, p, real, imag, strict)
    return out, dp, dw


def compare(actual, expected, dtype):
    report = {}
    for name, actual_tensor, expected_tensor in zip(
        ("forward", "dinput", "dweight"), actual, expected, strict=True,
    ):
        ref = expected_tensor.float()
        difference = actual_tensor.float() - ref
        absolute = float(difference.abs().max())
        relative = float(difference.norm() / ref.norm().clamp_min(1e-20))
        rel_limit = {torch.bfloat16: 0.005, torch.float16: 0.001, torch.float32: 2e-5}[dtype]
        if name == "dweight":
            rel_limit = 2e-5
        report[name] = {"max_abs": absolute, "relative_l2": relative,
                        "passed": relative <= rel_limit}
    return report


def parity_case(batch, shape, dtype, strict, baseline):
    p, g, real, imag = inputs(batch, shape, dtype, strict)
    ref = (reader._spatial_reference(p, real, imag, strict_complex=strict),
           *reader._spatial_reference_backward(g, p, real, imag, strict_complex=strict))
    base = compute(baseline, p, g, real, imag, strict) if baseline else None
    arms = {}
    for vec in ("1", "2", "4", "auto"):
        os.environ["LNET_READER_VEC"] = vec
        actual = compute(reader, p, g, real, imag, strict)
        arms[vec] = {"reference": compare(actual, ref, dtype)}
        if base is not None:
            arms[vec]["baseline"] = compare(actual, base, dtype)
    return {"shape": (batch, *shape), "dtype": str(dtype), "strict": strict, "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bench", "parity"))
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--wave17b", type=Path)
    parser.add_argument("--wave18b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=4)
    parser.add_argument("--case", type=int)
    parser.add_argument("--arms", nargs="+", choices=("1", "2", "4", "auto"),
                        default=("1", "2", "4", "auto"))
    args = parser.parse_args()
    torch.manual_seed(644)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    baseline = load_baseline(args.baseline)
    shapes = [args.shape] if args.shape else STANDARD
    if args.case is not None:
        shapes = [shapes[args.case]]
    dtypes = (torch.bfloat16, torch.float16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with args.out.open("w") as out:
        if args.mode == "bench":
            modules = [("baseline", baseline)] if baseline else []
            if args.wave17b:
                modules.append(("wave17b", load_baseline(args.wave17b, "wave17b")))
            if args.wave18b:
                modules.append(("wave18b", load_baseline(args.wave18b, "wave18b")))
            modules.extend((vec, reader) for vec in args.arms)
            cases = (
                time_case(args.batch, shape, dtype, args.samples, modules)
                for shape, dtype in itertools.product(shapes, dtypes)
            )
        else:
            edges = [
                ((1, 3, 7)[i % 3], ((5, 13, 17)[i % 3], (7, 15, 19)[i % 3], m,
                                     (1, 2, 4)[i % 3]))
                for i, m in enumerate(PROBE_EDGES)
            ]
            cases = (
                parity_case(batch, shape, dtype, strict, baseline)
                for (batch, shape), dtype, strict in itertools.product(
                    [(1, shape) for shape in shapes] + edges,
                    (*dtypes, torch.float32), (False, True),
                )
            )
        for record in cases:
            print(json.dumps(record), file=out, flush=True)
            if args.mode == "parity":
                failed |= any(
                    not metric["passed"] for arm in record["arms"].values()
                    for comparison in arm.values() for metric in comparison.values()
                )
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
