"""Capture reader CUDA event durations grouped by launch shape."""
from collections.abc import Callable
from unittest.mock import patch

import torch

from h200.liverms_v2_opt import bench_h200 as bench

# The probe intentionally wraps the private reader launch interface.
# ruff: noqa: SLF001
from lnet import factorized_reader_kernels as reader

_PROFILE = bench.profile_step
_LAUNCH = reader._launch_reader_cuda


def profile(step: Callable) -> dict:
    launches = []

    def track(
        module: reader._CudaModule, op: str, grid: tuple[int, int, int],
        tensors: tuple[torch.Tensor, ...], sizes: tuple[int, ...],
        *, stream: int, threads: int = reader._DEFAULT_THREADS,
    ) -> None:
        if op not in {"forward", "dinput", "dweight"}:
            _LAUNCH(module, op, grid, tensors, sizes, stream=stream, threads=threads)
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _LAUNCH(module, op, grid, tensors, sizes, stream=stream, threads=threads)
        end.record()
        batch = tensors[0].shape[0]
        h, w, m = sizes[:3]
        rank = sizes[-1]
        launches.append((op, (batch, h, w, m, rank), grid, start, end))

    with patch.object(reader, "_launch_reader_cuda", track):
        result = _PROFILE(step)
    grouped = {}
    for op, shape, grid, start, end in launches:
        key = (op, shape, grid)
        row = grouped.setdefault(key, {"op": op, "shape": shape, "grid": grid, "ms": []})
        row["ms"].append(start.elapsed_time(end))
    result["reader_launch_shapes"] = list(grouped.values())
    return result


if __name__ == "__main__":
    bench.profile_step = profile
    bench.main()
