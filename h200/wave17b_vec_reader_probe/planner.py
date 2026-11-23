"""Replay reader dispatch with NVRTC, ptxas and offline SM resource estimates."""

# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, PLR0915, S603, SLF001
import argparse
import itertools
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
from bench import STANDARD
from offline import OfflineModule

from lnet import factorized_reader_kernels as reader

SM = {
    "86": (80, 48, 16, 100 * 1024, 99 * 1024),
    "89": (128, 48, 24, 100 * 1024, 99 * 1024),
    "90": (132, 64, 32, 228 * 1024, 227 * 1024),
}


def resources(log):
    kernels = {}
    for part in log.split("Compiling entry function ")[1:]:
        name = part.split("'")[1].removeprefix("reader_")
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                          r"(\d+) bytes spill loads", part)
        registers = re.search(r"Used (\d+) registers", part)
        shared = re.search(r"(\d+) bytes smem", part)
        kernels[name] = {
            "regs": int(registers[1]),
            "smem": int(shared[1]) if shared else 0,
            **dict(zip(("stack", "spill_stores", "spill_loads"), map(int, frame.groups()),
                       strict=True)),
        }
    return kernels


def replay(job):
    target, case, dtype, rank, out, cuda = job
    sms, sm_warps, max_blocks, smem, block_smem = SM[target]
    batch = 1024 if target == "90" else 128
    height, width, modes, _ = case
    shape = batch, height, width, modes
    directory = out / "-".join(map(str, (target, dtype, rank, height, width, modes)))
    directory.mkdir(exist_ok=True)
    props = SimpleNamespace(major=int(target[0]), minor=int(target[1]),
                            multi_processor_count=sms, shared_memory_per_block=block_smem)
    torch.cuda.get_device_properties = lambda _: props
    torch.cuda.get_device_capability = lambda _: (props.major, props.minor)
    reader._CudaModule = OfflineModule
    reader._reader_cuda_module.cache_clear()
    reader._vector_plan.cache_clear()
    records = []
    original = reader._reader_cuda_module

    def assemble(*args, **kwargs):
        module = original(*args, **kwargs)
        if hasattr(module, "resources"):
            return module
        prefix = directory / str(len(records))
        ptx = prefix.with_suffix(".ptx")
        ptx.write_bytes(module.image)
        process = subprocess.run(
            [cuda / "ptxas", "-v", f"-arch=sm_{target}", ptx, "-o",
             prefix.with_suffix(".cubin")], capture_output=True, text=True, check=True,
        )
        prefix.with_suffix(".log").write_text(process.stderr)
        module.resources = resources(process.stderr)
        records.append({"options": kwargs, "columns": args[2], "resources": module.resources})
        return module

    def resident(module, op, warps):
        meta = module.resources[op]
        if meta["stack"] or meta["spill_stores"] or meta["spill_loads"]:
            return 0
        per_warp = reader._ceildiv(meta["regs"] * reader._WARP_SIZE, 256) * 256
        # Warps are allocated to four SM processing blocks. CUDA also
        # reserves shared memory for CTA state.
        reg_blocks = (65536 // (per_warp * 4) * 4) // warps
        cta_shared = reader._ceildiv(meta["smem"] + 1024, 256) * 256
        return min(max_blocks, sm_warps // warps, reg_blocks, smem // cta_shared)

    reader._reader_cuda_module = assemble
    reader._reader_resident = resident
    projected = SimpleNamespace(
        shape=(*shape[:3], modes * rank), dtype=dtype, device=torch.device("cuda:0"),
    )
    real = SimpleNamespace(shape=(modes, rank, 3, 3), dtype=dtype)
    scalar_plan = reader._rank_four_plan(projected, real, strict=False)
    compile_shape = reader._reader_compile_shape(projected, modes)
    types = (reader._CUDA_TYPES[dtype],) * 3
    dispatch = {}
    for arm in ("auto", "1", "2", "4"):
        dispatch[arm] = {}
        for op in ("forward", "dinput", "dweight"):
            selected = None if arm == "1" else reader._vector_plan(
                0, shape, compile_shape, rank, types, 4 if arm == "4" else 2, op, arm == "auto",
            )
            if selected:
                launch, module = selected
            else:
                module = reader._reader_module_for(
                    projected, real, strict=False, rank_four_plan=scalar_plan,
                )
                if op == "dweight":
                    tile, rows, warps = scalar_plan.weight if scalar_plan else (
                        reader._weight_geometry(projected, modes, rank, dtype)
                    )
                else:
                    rows, warps = scalar_plan.spatial if scalar_plan else reader._spatial_geometry(
                        width, rank, 3, dtype, strict=False,
                    )
                    tile = rank
                register_filters = scalar_plan is None and (
                    rank <= reader._RESIDENT_FILTER_RANK_LIMIT
                    or (rows, warps) == reader._COMPACT_SPATIAL_GEOMETRY
                )
                keep_real = op != "dweight" and (register_filters or (
                    scalar_plan is None and op == "dinput"
                    and compile_shape[2] >= 4 * reader._COMPACT_COLUMNS
                ))
                launch = reader._VectorLaunch(
                    1, reader._reader_columns(width), rows, warps, tile, keep_real,
                )
            active = resident(module, op, launch.warps)
            if op == "dweight":
                mode_tiles = reader._ceildiv(modes, reader._WARP_SIZE * launch.vec)
                row_tiles = reader._ceildiv(height, launch.rows)
                available = reader._ceildiv(batch * row_tiles, launch.warps)
                waves = scalar_plan.weight_waves if launch.vec == 1 and scalar_plan else 5
                blocks = min(available, reader._ceildiv(waves * sms, mode_tiles))
                blocks *= mode_tiles * reader._ceildiv(rank, launch.rank_tile)
            else:
                blocks = reader._reader_spatial_grid(
                    *shape, launch.rows, launch.warps, launch.vec, launch.columns,
                )
                blocks = blocks[0] * blocks[1] * blocks[2]
            dispatch[arm][op] = {
                **launch._asdict(), **module.resources[op], "resident_blocks_estimate": active,
                "traffic_score": reader._traffic_score(launch, op, rank, width, blocks, active, sms)
                if active else 0,
            }
    reader._reader_cuda_module = original
    return {"sm": target, "shape": shape, "dtype": str(dtype), "rank": rank,
            "dispatch": dispatch, "assembled": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda/bin"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs="+", default=(4,))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = (
        (*case, args.output, args.cuda)
        for case in itertools.product(SM, STANDARD, (torch.bfloat16, torch.float16), args.ranks)
    )
    with (args.output / "plans.jsonl").open("w") as output, ProcessPoolExecutor(
        max_workers=args.jobs,
    ) as pool:
        for record in pool.map(replay, jobs):
            print(json.dumps(record), file=output, flush=True)


if __name__ == "__main__":
    main()
