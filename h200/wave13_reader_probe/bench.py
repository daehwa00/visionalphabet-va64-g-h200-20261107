"""Measure packed reader stages and verify their outputs."""

# The probe intentionally exercises the private launch interface.
# ruff: noqa: SLF001
import argparse
import json
from pathlib import Path
from typing import NamedTuple

import torch

from h200.liverms_v2_opt.bench_h200 import closeness, event_median
from lnet import factorized_reader_kernels as reader


def measure(
    module: reader._CudaModule, op: str, grid: tuple[int, int, int],
    tensors: tuple[torch.Tensor, ...], sizes: tuple[int, ...], warps: int,
) -> dict:
    stream = reader._reader_stream(module, tensors[0].device)

    def call() -> None:
        reader._launch_reader_cuda(
            module, op, grid, tensors, sizes, stream=stream, threads=warps * 32,
        )

    ms = event_median(call)
    size = sum(t.numel() * t.element_size() for t in tensors)
    return {"ms": ms, "effective_gb_s": size / (ms * 1e6), "grid": grid}


class Geometry(NamedTuple):
    spatial_rows: int
    spatial_warps: int
    rank_tile: int
    weight_rows: int
    weight_warps: int
    weight_waves: int


def candidate(
    shape: tuple[int, int, int, int], data: tuple[torch.Tensor, ...], modes_per_thread: int,
    geometry: Geometry | None = None,
) -> tuple[dict, tuple[torch.Tensor, ...]]:
    batch, height, width, modes = shape
    projected, grad, real, imag = data
    rank = real.shape[1]
    device = projected.device.index
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    plan = reader._rank_four_plan(projected, real, grad, strict=False)
    if modes_per_thread > 1:
        sr, sw, tile, wr, ww = 2, 4, (1 if modes_per_thread == 4 else 2), 2, 4
        register = keep = 0
        sv = 2
    else:
        sr, sw = plan.spatial if plan else reader._spatial_geometry(
            width, rank, 3, projected.dtype, strict=False,
        )
        tile, wr, ww = plan.weight if plan else reader._weight_geometry(
            projected, modes, rank, grad.dtype,
        )
        register = keep = 0 if plan else -1
        sv = 1
    waves = plan.weight_waves if plan else 5
    if geometry is not None:
        sr, sw, tile, wr, ww, waves = geometry
        register = keep = 0
    mode_tiles = reader._ceildiv(modes, 32 * modes_per_thread)
    available = reader._ceildiv(batch * reader._ceildiv(height, wr), ww)
    workers = min(available, reader._ceildiv(waves * sms, mode_tiles))
    if ww == reader._WIDE_WEIGHT_WARPS and rank <= reader._CENTERED_WEIGHT_TILE:
        workers = max(workers, min(
            reader._ceildiv(available * 3, 5),
            reader._ceildiv(reader._MAX_STREAMING_WEIGHT_WAVES * sms, mode_tiles),
        ))
    module = reader._reader_cuda_module(
        device, reader._reader_compile_shape(projected, modes), reader._reader_columns(width),
        rank, 3, ("__nv_bfloat16",) * 3, strict=False, spatial_geometry=(sr, sw),
        weight_geometry=(tile, wr, ww), register_filters=register, keep_real=keep,
        spatial_vec=sv, weight_vec=modes_per_thread,
    )
    out, dp = torch.empty_like(grad), torch.empty_like(projected)
    partial = projected.new_empty((workers, 2, rank, 3, 3, modes), dtype=torch.float32)
    dw = real.new_empty((2, *real.shape), dtype=torch.float32)
    sizes = height, width, modes, rank
    grid = reader._reader_spatial_grid(batch, height, width, modes, sr, sw, sv)
    results = {
        "forward": measure(module, "forward", grid, (projected, real, imag, out), sizes, sw),
        "dinput": measure(module, "dinput", grid, (grad, real, imag, dp), sizes, sw),
        "dweight": measure(
            module, "dweight", (mode_tiles * reader._ceildiv(rank, tile), workers, 1),
            (projected, grad, partial), (height, width, modes, batch, rank), ww,
        ),
    }
    parallel = workers >= reader._WEIGHT_PARALLEL_REDUCTION_WORKERS
    reduction = "dweight_reduce_parallel" if parallel else "dweight_reduce"
    results["reduce"] = measure(
        module, reduction, (reader._ceildiv(dw.numel(), 32 if parallel else 128), 1, 1),
        (partial, dw), (modes, rank, workers), 4,
    )
    return results, (out, dp, dw)


def run(
    shapes: list[tuple[int, int, int, int, int]], out_path: Path, *, sweep: bool,
) -> None:
    torch.manual_seed(52)
    torch.backends.cudnn.allow_tf32 = False
    results = {}
    baseline = [(str(vec), vec, None) for vec in (1, 2, 4)]
    configs = [
        (1, Geometry(4, 4, 2, 2, 8, 20)),
        (2, Geometry(1, 4, 1, 1, 4, 20)),
        (2, Geometry(2, 2, 2, 1, 4, 20)),
        (2, Geometry(2, 4, 4, 1, 4, 20)),
        (2, Geometry(4, 2, 2, 1, 8, 20)),
        (2, Geometry(4, 4, 2, 2, 8, 20)),
        (4, Geometry(2, 2, 1, 1, 4, 20)),
        (4, Geometry(4, 2, 2, 1, 4, 20)),
    ]
    sweeps = [
        ("v" + str(vec) + "-" + "x".join(map(str, cfg)), vec, cfg) for vec, cfg in configs
    ]
    for batch, height, width, modes, rank in shapes:
        key = f"{batch}x{height}x{width}x{modes}-r{rank}"
        data = (
            torch.randn((batch, height, width, rank * modes), device="cuda", dtype=torch.bfloat16),
            torch.randn((batch, height, width, 2 * modes), device="cuda", dtype=torch.bfloat16),
            *torch.randn((2, modes, rank, 3, 3), device="cuda", dtype=torch.bfloat16).unbind(),
        )
        rows = results[key] = {}
        reference = None
        for name, vec, config in (baseline + sweeps if sweep else baseline):
            timing, tensors = candidate((batch, height, width, modes), data, vec, config)
            if reference is None:
                reference = tensors
            rows[name] = {
                "stages": timing,
                "parity": {
                    op: closeness([actual], [expected])
                    for op, actual, expected in zip(
                        ("forward", "dinput", "dweight"), tensors, reference, strict=True,
                    )
                },
            }
            del tensors
        del reference, data
        out_path.write_text(json.dumps(results, indent=2))
        print(key, flush=True)  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    shapes = [
        (min(args.max_batch, 1024), side, side, 128 if side == 7 else 160, 4)
        for side in (56, 28, 14, 7)
    ]
    shapes.extend(((4, 128, 192, 160, 4), (4, 512, 768, 48, 4), (4, 96, 128, 80, 2)))
    run(shapes, args.out, sweep=args.sweep)


if __name__ == "__main__":
    main()
