"""Capture production launch geometry while assembling its CUDA kernels offline."""

# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, S101, S603, SLF001
import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import cache
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from bench import load_baseline
from offline import OfflineModule
from screen import GENERAL, REAL

from lnet import factorized_reader_kernels as reader

SM = {"86": (80, 48, 16, 100, 99), "89": (128, 48, 24, 100, 99),
      "90": (132, 64, 32, 228, 227)}


def resources(log):
    kernels = {}
    for part in log.split("Compiling entry function ")[1:]:
        name = part.split("'")[1].removeprefix("reader_")
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                          r"(\d+) bytes spill loads", part)
        registers = re.search(r"Used (\d+) registers", part)
        shared = re.search(r"(\d+) bytes smem", part)
        kernels[name] = {
            "regs": int(registers[1]), "smem": int(shared[1]) if shared else 0,
            "stack_spill": list(map(int, frame.groups())),
        }
    return kernels


class TensorSpec(SimpleNamespace):
    def data_ptr(self):
        return 0

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()

    def contiguous(self):
        return self

    def new_empty(self, shape, dtype=None):
        return TensorSpec(shape=shape, device=self.device, dtype=dtype or self.dtype)

    def numel(self):
        return prod(self.shape)


def replay(module, target, cases, directory, ptxas):  # noqa: C901, PLR0915
    sms, sm_warps, max_blocks, smem, block_smem = SM[target]
    props = SimpleNamespace(
        major=int(target[0]), minor=int(target[1]), multi_processor_count=sms,
        shared_memory_per_block=block_smem * 1024, shared_memory_per_multiprocessor=smem * 1024,
        regs_per_multiprocessor=65536, max_threads_per_multi_processor=sm_warps * 32,
    )
    original = module._reader_cuda_module
    original.cache_clear()
    for attr in ("_vector_plan", "_wave25b_spatial_plan"):
        if hasattr(module, attr):
            getattr(module, attr).cache_clear()

    def assemble(*args, **kwargs):
        loaded = original(*args, **kwargs)
        if not hasattr(loaded, "resources"):
            ptx = directory / "reader.ptx"
            ptx.write_bytes(loaded.image)
            process = subprocess.run(
                [ptxas, "-v", f"-arch=sm_{target}", ptx, "-o", directory / "reader.cubin"],
                capture_output=True, text=True, check=False,
            )
            if process.returncode:
                raise RuntimeError(process.stderr)
            loaded.resources = resources(process.stderr)
            disassembly = subprocess.run(
                [str(Path(ptxas).with_name("cuobjdump")), "-sass", directory / "reader.cubin"],
                capture_output=True, text=True, check=True,
            ).stdout
            for entry in disassembly.split("Function : ")[1:]:
                name = entry.split()[0].removeprefix("reader_")
                widths = re.findall(r"\b(?:LDG|STG)(?:\.[A-Z0-9]+)*", entry)
                loaded.resources[name]["sass_global"] = dict(Counter(widths))
                loaded.resources[name]["sass_sha256"] = hashlib.sha256(entry.encode()).hexdigest()
            loaded.args, loaded.options = args, kwargs
        return loaded

    def resident(loaded, op, warps):
        meta = loaded.resources[op]
        if any(meta["stack_spill"]):
            return 0
        registers = module._ceildiv(meta["regs"] * 32, 256) * 256
        reg_blocks = (65536 // (registers * 4) * 4) // warps
        cta_shared = module._ceildiv(meta["smem"] + 1024, 256) * 256
        return min(max_blocks, sm_warps // warps, reg_blocks, smem * 1024 // cta_shared)

    def capture(loaded, op, grid, *_args, threads=128, **_kwargs):
        args, options = loaded.args, loaded.options
        columns, rank = args[2:4]
        weight = op == "dweight"
        if op.startswith("dweight_reduce"):
            dispatch[op] = {"grid": grid, **loaded.resources[op]}
            return
        rows, warps = options.get("spatial_geometry", module._DEFAULT_SPATIAL_GEOMETRY)
        tile = rank
        vec = options.get("spatial_vec", 1)
        if weight:
            tile, rows, warps = options.get("weight_geometry") or module._DEFAULT_WEIGHT_GEOMETRY
            tile, rows, warps = (2, 1, 4) if options["strict"] else (tile, rows, warps)
            vec = options.get("weight_vec", 1)
        keep = options.get("keep_real", -1)
        filters = options.get("register_filters", -1)
        resident_filter = filters == 1 or (filters == -1 and (
            (rank <= module._RESIDENT_FILTER_RANK_LIMIT and args[5][0] != "float")
            or (rows, warps) == module._COMPACT_SPATIAL_GEOMETRY
        ))
        keep_real = (
            not weight and vec == 1 and (
                (not options["strict"] and resident_filter)
                or (op == "dinput" and (keep == 1 or
                                        (keep == -1 and args[1][2] >= 4 * module._COMPACT_COLUMNS)))
            )
        )
        assert warps * 32 == threads
        dispatch[op] = dict(vec=vec, rows=rows, cols=columns, warps=warps, rank_tile=tile,
                            keep_real=int(keep_real), grid=grid,
                            mode_lanes=options.get("mode_lanes", 32),
                            rank_lanes=options.get("rank_lanes", 1),
                            resident=resident(loaded, op, warps), **loaded.resources[op])

    with (
        patch.object(torch.cuda, "get_device_properties", lambda _: props),
        patch.object(torch.cuda, "get_device_capability", lambda _: (props.major, props.minor)),
        patch.object(torch.cuda, "device", lambda _: nullcontext()),
        patch.object(torch, "empty_like", lambda x: x),
        patch.object(module, "_CudaModule", OfflineModule),
        patch.object(module, "_reader_cuda_module", assemble),
        patch.object(module, "_reader_resident", resident, create=True),
        patch.object(module, "_use_reader_cuda", lambda *_: True),
        patch.object(module, "_reader_stream", lambda *_: None),
        patch.object(module, "_launch_reader_cuda", capture),
    ):
        for shape, dtypes, strict in cases:
            b, h, w, m, r = shape
            p = TensorSpec(shape=(b, h, w, m * r * (2 if strict else 1)),
                           dtype=dtypes[0], device=torch.device("cuda", 0))
            g = p.new_empty((b, h, w, 2 * m), dtypes[2])
            f = p.new_empty((m, r, 3, 3), dtypes[1])
            dispatch = {}
            module._packed_spatial._init_fn(p, f, f, strict)
            module._packed_spatial_backward._init_fn(g, p, f, f, strict)
            yield {"sm": target, "shape": shape, "dtypes": list(map(str, dtypes)), "strict": strict,
                       "dispatch": dispatch}


@cache
def load_matrix_baseline(path):
    return load_baseline(path)


def matrix_job(job):
    target, shape, arm, directory, ptxas, reference = job
    dtype = (torch.bfloat16, torch.float16, torch.float32)
    dtypes = tuple(itertools.product(dtype, repeat=3))
    cases = [(shape, types, strict) for types, strict in itertools.product(dtypes, (False, True))]
    os.environ["LNET_READER_W25B"] = arm
    module = load_matrix_baseline(reference) if arm == "wave21" else reader
    scratch = directory / "-".join(map(str, (target, *shape, arm)))
    scratch.mkdir(exist_ok=True)
    records = [
        {"arm": arm, **record} for record in replay(module, target, cases, scratch, ptxas)
    ]
    (scratch / "reader.ptx").unlink()
    (scratch / "reader.cubin").unlink()
    scratch.rmdir()
    return records


def matrix(args):
    shapes = (*REAL, *GENERAL, (9, 11, 34, 4))
    if args.height:
        shapes = tuple(shape for shape in shapes if shape[0] in args.height)
    reference = args.refs / "wave21.py" if args.refs else None
    jobs = (
        (target, (1024 if target == "90" else 128, h, w, m, rank), arm,
         args.out, args.ptxas, reference)
        for target, (h, w, m, _), rank, arm in itertools.product(
            args.sm, shapes, args.ranks, args.arms,
        )
    )
    failed = False
    with (args.out / "matrix.jsonl").open("w") as output, ProcessPoolExecutor(
        max_workers=args.jobs,
    ) as pool:
        for records in pool.map(matrix_job, jobs):
            for record in records:
                print(json.dumps(record), file=output, flush=True)
                failed |= any(any(op["stack_spill"]) for op in record["dispatch"].values())
    raise SystemExit(int(failed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sm", choices=SM, nargs="+", default=tuple(SM))
    parser.add_argument("--ptxas", default="/usr/local/cuda/bin/ptxas")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--height", type=int, nargs="+")
    parser.add_argument("--ranks", type=int, choices=range(1, 5), nargs="+", default=range(1, 5))
    parser.add_argument("--arms", choices=("on", "off", "wave21"), nargs="+", default=("on", "off"))
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.matrix:
        if "wave21" in args.arms and args.refs is None:
            parser.error("--refs is required for the wave21 matrix")
        matrix(args)
    if args.refs is None:
        parser.error("--refs is required for historical comparisons")
    historical = load_baseline(args.refs / "wave20.py")
    with TemporaryDirectory() as tmp:
        wave21 = Path(tmp) / "wave21.py"
        source = (args.refs / "wave21.py").read_text()
        wave21.write_text(source.replace("va_reader::", "va_wave21::"))
        baseline = load_baseline(wave21)
    with (args.out / "plans.jsonl").open("w") as output:
        for target in args.sm:
            batch = 1024 if target == "90" else 128
            cases = [((batch, *s), (torch.bfloat16,) * 3, False) for s in REAL]
            arms = (("wave21", baseline), ("off", reader), ("on", reader), ("wave20", historical))
            for name, module in arms:
                os.environ["LNET_READER_W25B"] = name
                for record in replay(module, target, cases, args.out, args.ptxas):
                    print(json.dumps(dict(wave=name, **record)), file=output, flush=True)
    (args.out / "reader.ptx").unlink()
    (args.out / "reader.cubin").unlink()


if __name__ == "__main__":
    main()
