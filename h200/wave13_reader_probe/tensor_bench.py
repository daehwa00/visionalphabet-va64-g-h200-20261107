"""Screen Tensor Core dweight using a tiled correlation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from h200.liverms_v2_opt.bench_h200 import closeness
from h200.wave13_reader_probe.bench import candidate, measure

# The probe intentionally exercises the private launch interface.
# ruff: noqa: SLF001
from lnet import factorized_reader_kernels as reader

_INSERT = 'extern "C" __global__ void reader_dweight('
_WEIGHT_DISPATCH = "    if constexpr (Rank && Kernel == 3 && !Strict)\n"
_TEMPLATE = Path(__file__).with_name("tensor_weight.cuh")
_BASE = reader._READER_CUDA


def use_tensor_source(tile_h: int, tile_w: int) -> None:
    dispatch = (
        f"    tensor_weight_gradient<Rank, {tile_h}, {tile_w}>(\n"
        "        reinterpret_cast<const unsigned short*>(projected),\n"
        "        reinterpret_cast<const unsigned short*>(grad), partial, h, w, m, batch);\n"
        "    return;\n"
    )
    source = _BASE.replace(_INSERT, _TEMPLATE.read_text() + "\n" + _INSERT, 1)
    reader._READER_CUDA = source.replace(_WEIGHT_DISPATCH, dispatch + _WEIGHT_DISPATCH, 1)
    reader._reader_cuda_module.cache_clear()


def tensor_weight(
    shape: tuple[int, int, int, int], data: tuple[torch.Tensor, ...],
    tile_h: int, tile_w: int, waves: int,
) -> tuple[dict, torch.Tensor]:
    batch, height, width, modes = shape
    projected, grad, real, _ = data
    rank = real.shape[1]
    device = projected.device.index
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    div = reader._ceildiv
    mode_tiles = div(modes, 16)
    workers = min(batch * div(height, tile_h) * div(width, tile_w), div(sms * waves, mode_tiles))
    use_tensor_source(tile_h, tile_w)
    try:
        module = reader._reader_cuda_module(
            device, reader._reader_compile_shape(projected, modes), reader._reader_columns(width),
            rank, 3, ("__nv_bfloat16",) * 3, strict=False,
        )
        partial = projected.new_empty((workers, 2, rank, 3, 3, modes), dtype=torch.float32)
        dw = real.new_empty((2, *real.shape), dtype=torch.float32)
        timing = {"dweight": measure(
            module, "dweight", (mode_tiles, workers, 1), (projected, grad, partial),
            (height, width, modes, batch, rank), 4,
        )}
        parallel = workers >= reader._WEIGHT_PARALLEL_REDUCTION_WORKERS
        reduction = "dweight_reduce_parallel" if parallel else "dweight_reduce"
        timing["reduce"] = measure(
            module, reduction, (div(dw.numel(), 32 if parallel else 128), 1, 1),
            (partial, dw), (modes, rank, workers), 4,
        )
    finally:
        reader._READER_CUDA = _BASE
        reader._reader_cuda_module.cache_clear()
    return timing, dw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    torch.manual_seed(52)
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    result = {}
    for rank, modes, side in ((4, 160, 56), (4, 160, 28), (2, 80, 96)):
        data = (
            torch.randn((args.batch, side, side, rank * modes), **kwargs),
            torch.randn((args.batch, side, side, 2 * modes), **kwargs),
            *torch.randn((2, modes, rank, 3, 3), **kwargs).unbind(),
        )
        shape = (args.batch, side, side, modes)
        base_timing, (_, _, ref) = candidate(shape, data, 1)
        rows = result["x".join(map(str, (*shape, rank)))] = {"scalar": base_timing}
        for h, w in ((2, 8), (2, 16), (4, 8), (4, 16), (4, 32)):
            for waves in (8, 20):
                key = f"{h}x{w}-w{waves}"
                try:
                    timing, dw = tensor_weight(shape, data, h, w, waves)
                    rows[key] = {"timing": timing, "parity": closeness([dw], [ref])}
                except RuntimeError as exc:
                    rows[key] = {"error": str(exc)}
        args.out.write_text(json.dumps(result, indent=2))
        del data


if __name__ == "__main__":
    main()
