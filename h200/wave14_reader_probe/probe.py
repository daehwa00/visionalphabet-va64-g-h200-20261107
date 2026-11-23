"""Measure isolated Wave 13B and 14B reader stages using paired CUDA graph replays."""

from __future__ import annotations

# The probe deliberately exercises private baseline launch and reference APIs.
# ruff: noqa: SLF001
import argparse
import ctypes as ct
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch
from sourcegen import RING, SHAPES, SPATIAL, WEIGHT, WEIGHT_STREAM, Tile, generate
from torch import Tensor

from lnet import factorized_reader_kernels as reader
from lnet.cuda_nvrtc import _check_nvrtc, _CudaModule, _include_dirs, _nvrtc

SHAPE = tuple[int, int, int, int, int]
DTYPE = torch.bfloat16
TYPES = ("__nv_bfloat16",) * 3
DIV = reader._ceildiv
OPS = ("forward", "dinput", "dweight", "dweight_reduce")


def compile_probe(shape: SHAPE, tiles: set[str]) -> _CudaModule:
    nvrtc = _nvrtc()
    program = ct.c_void_p()
    _check_nvrtc(
        nvrtc,
        nvrtc.nvrtcCreateProgram(
            ct.byref(program),
            generate(shape, tiles).encode(),
            b"wave14b_reader.cu",
            0,
            None,
            None,
        ),
    )
    major, minor = torch.cuda.get_device_capability()
    flags = [
        f"--gpu-architecture=compute_{major}{minor}".encode(),
        b"--std=c++17",
        *(f"--include-path={p}".encode() for p in _include_dirs()),
    ]
    try:
        options = (ct.c_char_p * len(flags))(*flags)
        _check_nvrtc(nvrtc, nvrtc.nvrtcCompileProgram(program, len(flags), options), program)
        size = ct.c_size_t()
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTXSize(program, ct.byref(size)), program)
        image = ct.create_string_buffer(size.value)
        _check_nvrtc(nvrtc, nvrtc.nvrtcGetPTX(program, image), program)
    finally:
        nvrtc.nvrtcDestroyProgram(ct.byref(program))
    return _CudaModule(image, torch.cuda.current_device())


@dataclass(frozen=True)
class Launch:
    module: _CudaModule
    name: str
    grid: tuple[int, int, int]
    tensors: tuple[Tensor, ...]
    sizes: tuple[int, ...]
    threads: int = 128
    smem: int = 0

    def __call__(self) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        args = [ct.c_void_p(t.data_ptr()) for t in self.tensors]
        args.extend(ct.c_int(value) for value in self.sizes)
        self.module.launch(self.name, self.grid, (self.threads, 1, 1), self.smem, stream, args)

    @property
    def logical_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors)


def timed_graphs(calls: dict[str, Launch], samples: int) -> dict:
    graphs = {}
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[name] = graph
    # Repeat rounds over all variants. Rotate the order to reduce clock drift bias.
    values: dict[str, list[float]] = {name: [] for name in graphs}
    order = list(graphs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(samples + 2):
        for name in order[i % len(order) :] + order[: i % len(order)]:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            if i >= 2:
                values[name].append(start.elapsed_time(end))
    return {
        name: {
            "median_ms": median(values[name]),
            "samples_ms": values[name],
            "logical_bytes": call.logical_bytes,
            "effective_gb_s": call.logical_bytes / (median(values[name]) * 1e6),
            "grid": call.grid,
            "threads": call.threads,
            "dynamic_smem_bytes": call.smem,
        }
        for name, call in calls.items()
    }


@dataclass(frozen=True)
class Inputs:
    projected: Tensor
    grad: Tensor
    real: Tensor
    imag: Tensor

    @property
    def shape(self) -> SHAPE:
        b, h, w, _ = self.projected.shape
        modes, rank, _, _ = self.real.shape
        return b, h, w, modes, rank

    @classmethod
    def create(cls, shape: SHAPE) -> Inputs:
        b, h, w, m, r = shape
        args = {"device": "cuda", "dtype": DTYPE}
        p = torch.randn((b, h, w, m * r), **args)
        g = torch.randn((b, h, w, m * 2), **args)
        r, i = (torch.randn((2, m, r, 3, 3), **args) * 0.125).unbind()
        return cls(p, g, r, i)


def baseline(inputs: Inputs) -> tuple[_CudaModule, tuple[int, int], tuple[int, int, int, int, int]]:
    p, g, r = inputs.projected, inputs.grad, inputs.real
    b, h, w, m, rank = inputs.shape
    props = torch.cuda.get_device_properties(p.device)
    plan = reader._rank_four_plan(p, r, g, strict=False)
    vec = reader._reader_weight_vector_width(
        p, r, g, strict=False, major=props.major, sms=props.multi_processor_count
    )
    module = reader._reader_module_for(p, r, g, strict=False, rank_four_plan=plan, weight_vec=vec)
    spatial = plan.spatial if plan else reader._spatial_geometry(w, rank, 3, DTYPE, strict=False)
    tile, rows, warps = plan.weight if plan else reader._weight_geometry(p, m, rank, DTYPE)
    waves = plan.weight_waves if plan else 5
    if (
        plan is None
        and rank <= 2
        and w > 2 * reader._reader_columns(w)
        and warps != 8
        and b * h * w >= reader._SPECIALIZE_MIN_PIXELS
    ):
        waves = 3
    if vec == 2:
        tile, rows, warps = reader._VECTOR_WEIGHT_GEOMETRY
    return module, spatial, (tile, rows, warps, vec, waves)


def spatial_call(
    inputs: Inputs,
    output: Tensor,
    op: str,
    module: _CudaModule,
    *,
    tile: Tile | None = None,
    geometry: tuple[int, int] = (2, 4),
) -> Launch:
    b, h, w, m, rank = inputs.shape
    src = inputs.projected if op == "forward" else inputs.grad
    smem = 0
    if tile is None:
        rows, warps = geometry
        cols = reader._reader_columns(w) * warps
        name = f"reader_{op}"
    else:
        rows, warps = tile.rows, tile.warps
        cols = tile.columns * warps if tile in RING else tile.columns
        name = f"reader_{tile.name}_{op}"
        if tile in SPATIAL:
            smem = tile.spatial_smem(rank, backward=op == "dinput")
        module.load_function(name, smem)
    return Launch(
        module,
        name,
        (DIV(m, 32), DIV(h, rows) * DIV(w, cols), b),
        (src, inputs.real, inputs.imag, output),
        (h, w, m, rank),
        warps * 32,
        smem,
    )


def weight_calls(
    inputs: Inputs,
    dw: Tensor,
    module: _CudaModule,
    reduce_module: _CudaModule,
    *,
    tile: Tile | None = None,
    geometry: tuple[int, int, int, int, int] = (2, 2, 4, 1, 5),
) -> tuple[Launch, Launch]:
    b, h, w, m, rank = inputs.shape
    rank_tile, rows, warps, vec, waves = geometry
    sms = torch.cuda.get_device_properties(inputs.projected.device).multi_processor_count
    if tile is None:
        tiles = DIV(m, 32 * vec)
        capacity = DIV(b * DIV(h, rows), warps)
        workers = min(capacity, DIV(waves * sms, tiles))
        if warps == 8 and rank <= 2:
            target = min(DIV(capacity * 3, 5), DIV(40 * sms, tiles))
            workers = min(capacity, max(workers, target))
        name, smem = "reader_dweight", 0
        mode_rank_tiles = tiles * DIV(rank, rank_tile)
    else:
        tiles, warps = DIV(m, 32), tile.warps
        capacity = b * DIV(h, tile.rows) * DIV(w, tile.columns)
        workers = min(capacity, DIV(waves * sms, tiles))
        name, smem = f"reader_{tile.name}_dweight", tile.weight_smem(rank)
        mode_rank_tiles = tiles
        module.load_function(name, smem)
    partial = inputs.projected.new_empty((workers, 2, rank, 3, 3, m), dtype=torch.float32)
    compute = Launch(
        module,
        name,
        (mode_rank_tiles, workers, 1),
        (inputs.projected, inputs.grad, partial),
        (h, w, m, b, rank),
        warps * 32,
        smem,
    )
    parallel = workers >= reader._WEIGHT_PARALLEL_REDUCTION_WORKERS
    reducer = Launch(
        reduce_module,
        "reader_dweight_reduce_parallel" if parallel else "reader_dweight_reduce",
        (DIV(dw.numel(), 32 if parallel else 128), 1, 1),
        (partial, dw),
        (m, rank, workers),
    )
    return compute, reducer


def compare(actual: Tensor, expected: Tensor, *, atol: float, rtol: float) -> dict:
    # Chunk over the batch so B1024 checks every element with bounded scratch.
    max_abs = mismatches = outside = 0
    size = actual[0].numel()
    chunk = max(1, 4_194_304 // size)
    for actual_chunk, expected_chunk in zip(
        actual.split(chunk), expected.split(chunk), strict=True
    ):
        a, b = actual_chunk.float(), expected_chunk.float()
        delta = (a - b).abs()
        max_abs = max(max_abs, float(delta.max()))
        mismatches += int(torch.count_nonzero(a != b))
        outside += int(torch.count_nonzero(~(delta <= atol + rtol * b.abs())))
    return {
        "exact": mismatches == 0,
        "elements": actual.numel(),
        "unequal_elements": mismatches,
        "max_abs": max_abs,
        "outside_tolerance": outside,
        "atol": atol,
        "rtol": rtol,
    }


def all_calls(
    inputs: Inputs,
    outputs: tuple[Tensor, Tensor, Tensor],
    names: set[str],
) -> dict[str, dict[str, Launch]]:
    b, h, w, m, rank = inputs.shape
    module, gs, gw = baseline(inputs)
    spatial = {
        op: {"wave13": spatial_call(inputs, output, op, module, geometry=gs)}
        for op, output in zip(OPS[:2], outputs[:2], strict=True)
    }
    weight, reduce = weight_calls(inputs, outputs[-1], module, module, geometry=gw)
    candidates = {**spatial, "dweight": {"wave13": weight}, "dweight_reduce": {"wave13": reduce}}
    props = torch.cuda.get_device_properties(inputs.projected.device)
    max_smem = props.shared_memory_per_block_optin
    tiles = [
        t
        for t in (*RING, *SPATIAL, *WEIGHT)
        if t.name in names
        and (
            t in RING
            or (t in SPATIAL and t.spatial_smem(rank, backward=False) <= max_smem)
            or (t in WEIGHT and t.weight_smem(rank) <= max_smem)
        )
    ]
    if tiles:
        tiled = compile_probe(inputs.shape, {t.name for t in tiles})
        for t in tiles:
            if t in WEIGHT:
                p, q = weight_calls(inputs, outputs[-1], tiled, module, tile=t)
                candidates["dweight"][t.name], candidates["dweight_reduce"][t.name] = p, q
            else:
                for op, out in zip(OPS[:2], outputs[:2], strict=True):
                    candidates[op][t.name] = spatial_call(inputs, out, op, tiled, tile=t)
    for name, (tile, rows, warps, vec, waves) in WEIGHT_STREAM.items():
        if name in names:
            streaming = reader._reader_cuda_module(
                torch.cuda.current_device(),
                (b, h, w, m),
                reader._reader_columns(w),
                rank,
                3,
                TYPES,
                strict=False,
                spatial_geometry=gs,
                weight_geometry=(tile, rows, warps),
                weight_vec=vec,
            )
            p, q = weight_calls(
                inputs, outputs[-1], streaming, module, geometry=(tile, rows, warps, vec, waves)
            )
            candidates["dweight"][name], candidates["dweight_reduce"][name] = p, q
    return candidates


def write_json(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def measure(inputs: Inputs, names: set[str], samples: int, *, reference: bool) -> dict:
    p, g, r, i = inputs.projected, inputs.grad, inputs.real, inputs.imag
    y, dx = torch.empty_like(g), torch.empty_like(p)
    dw = r.new_empty((2, *r.shape), dtype=torch.float32)
    calls = all_calls(inputs, (y, dx, dw), names)
    result = {op: timed_graphs(candidates, samples) for op, candidates in calls.items()}
    if reference:
        # PyTorch grouped convolution of one complete image for each geometry.
        torch.backends.cudnn.allow_tf32 = False
        sample = Inputs(p[-1:], g[-1:], r, i)
        sy = reader._spatial_reference(sample.projected, r, i, strict_complex=False)
        sdx, sdw = reader._spatial_reference_backward(
            sample.grad, sample.projected, r, i, strict_complex=False
        )
        sample_calls = all_calls(
            sample, (torch.empty_like(sy), torch.empty_like(sdx), torch.empty_like(sdw)), names
        )
    for op in OPS[:3]:
        options = calls[op]
        dest = options["wave13"].tensors[-1] if op != "dweight" else dw

        def run(name: str, operations: dict[str, dict[str, Launch]], op: str = op) -> None:
            operations[op][name]()
            if op == "dweight":
                operations["dweight_reduce"][name]()

        run("wave13", calls)
        baseline_output = dest.clone()
        for name, row in result[op].items():
            run(name, calls)
            if name != "wave13":
                row["wave13_parity"] = compare(
                    dest,
                    baseline_output,
                    atol=0.03 if op == "dweight" else 0,
                    rtol=2e-5 if op == "dweight" else 0,
                )
            if reference:
                run(name, sample_calls)
                ref = {"forward": sy, "dinput": sdx, "dweight": sdw}[op]
                out = sample_calls[op][name].tensors[-1]
                if op == "dweight":
                    out = sample_calls["dweight_reduce"][name].tensors[-1]
                row["pytorch_parity"] = compare(
                    out,
                    ref,
                    atol=0.006 if op == "dweight" else 0.002,
                    rtol=2e-5 if op == "dweight" else 0.008,
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--shapes", default="0,1,2,3,4,5,6")
    parser.add_argument("--names", default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(58)
    names = {tile.name for tile in (*SPATIAL, *RING, *WEIGHT)} | set(WEIGHT_STREAM)
    if args.names != "all":
        names &= set(args.names.split(","))
    props = torch.cuda.get_device_properties(0)
    rows: dict[str, dict] = {}
    result = {
        "device": props.name,
        "sms": props.multi_processor_count,
        "capability": [props.major, props.minor],
        "torch": torch.__version__,
        "shapes": rows,
    }
    for index in map(int, args.shapes.split(",")):
        b, *other = SHAPES[index]
        shape = (args.batch if b == 1024 else b, *other)
        key = "x".join(map(str, shape))
        print(key, flush=True)  # noqa: T201
        rows[key] = measure(Inputs.create(shape), names, args.samples, reference=args.reference)
        write_json(args.out, result)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
