"""Replay reader launches on CPU and audit their CUDA 12.8 PTX with ptxas."""

# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, S101, S603, SLF001
import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lnet import factorized_reader_kernels as reader

PTXAS = "/usr/local/cuda-12.8/bin/ptxas"
SM = {86: (80, 48, 16, 100), 89: (128, 48, 24, 100), 90: (132, 64, 32, 228),
      100: (148, 64, 32, 228)}
SHAPES = ((56, 56, 160, 4), (28, 28, 160, 4), (14, 14, 160, 4),
          (7, 7, 128, 4), (9, 11, 34, 4), (13, 19, 63, 3), (13, 19, 64, 6))
DTYPES = (torch.bfloat16, torch.float16)


class OfflineModule:
    def __init__(self, image, _device) -> None:
        self.ptx = image.raw

    def load_function(self, *_args):
        pass


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
        return math.prod(self.shape)


def assemble(loaded, sm, directory):
    if hasattr(loaded, "resources"):
        return
    ptx = directory / "reader.ptx"
    ptx.write_bytes(loaded.ptx)
    log = subprocess.run(
        [PTXAS, "-v", f"-arch=sm_{sm}", ptx, "-o", directory / "reader.cubin"],
        capture_output=True, text=True, check=True,
    ).stderr
    loaded.resources = {}
    for part in log.split("Compiling entry function ")[1:]:
        op = part.split("'")[1].removeprefix("reader_")
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                          r"(\d+) bytes spill loads", part)
        shared = re.search(r"(\d+) bytes smem", part)
        loaded.resources[op] = {
            "regs": int(re.search(r"Used (\d+) registers", part)[1]),
            "smem": int(shared[1]) if shared else 0,
            "stack_spill": list(map(int, frame.groups())),
        }


def resident(meta, sm, warps):
    _, sm_warps, max_blocks, smem = SM[sm]
    if any(meta["stack_spill"]):
        return 0
    regs_per_warp = math.ceil(meta["regs"] / 8) * 256
    reg_warps = (16384 // regs_per_warp) * 4
    cta_smem = math.ceil((meta["smem"] + 1024) / 256) * 256
    return min(max_blocks, sm_warps // warps, reg_warps // warps, smem * 1024 // cta_smem)


def replay(module, sm, shape, dtype, strict, directory, assemblies):
    sms, warps, _, shared = SM[sm]
    props = SimpleNamespace(
        major=sm // 10, minor=sm % 10, multi_processor_count=sms,
        total_memory=({86: 12, 89: 24, 90: 140, 100: 180}[sm]) * 1024**3,
        max_threads_per_multi_processor=warps * 32, regs_per_multiprocessor=65536,
        shared_memory_per_block=(shared - 1) * 1024, shared_memory_per_multiprocessor=shared * 1024,
    )
    original = module._reader_cuda_module
    module._wave25b_spatial_plan.cache_clear()
    launches = {}
    images = {}

    def build(*args, **kwargs):
        loaded = original(*args, **kwargs)
        if loaded.ptx in assemblies:
            loaded.resources = assemblies[loaded.ptx]
        else:
            assemble(loaded, sm, directory)
            assemblies[loaded.ptx] = loaded.resources
        loaded.args, loaded.options = args, kwargs
        images[repr((args, sorted(kwargs.items())))] = loaded.ptx
        return loaded

    def capture(loaded, op, grid, tensors, sizes, *, threads=128, **_kwargs):  # noqa: ARG001
        meta = loaded.resources[op]
        warps = threads // 32
        blocks = resident(meta, sm, warps)
        options = loaded.options
        key = op if op not in launches else f"{op}:{grid[1]}"
        launches[key] = dict(
            grid=grid, vec=options.get("weight_vec" if op == "dweight" else "spatial_vec", 1),
            columns=loaded.args[2], geometry=options.get("weight_geometry" if op == "dweight"
                                                        else "spatial_geometry"),
            warps=warps, resident_warps=blocks * warps,
            waves=math.prod(grid) / (sms * blocks) if blocks else None,
            partial_mib=tensors[-1].numel() * 4 / 1024**2
            if op in ("dweight", "dweight_reduce_chunks") else None,
            ptx_sha256=hashlib.sha256(loaded.ptx).hexdigest(), **meta,
        )

    def active_blocks(loaded, op, warps):
        return resident(loaded.resources[op], sm, warps)

    with (
        patch.object(torch.cuda, "get_device_properties", return_value=props),
        patch.object(torch.cuda, "get_device_capability", return_value=(props.major, props.minor)),
        patch.object(torch.cuda, "device", return_value=nullcontext()),
        patch.object(torch.cuda, "memory_allocated", return_value=0),
        patch.object(torch, "empty_like", side_effect=lambda x: x),
        patch.object(module, "_CudaModule", OfflineModule),
        patch.object(module, "_reader_cuda_module", build),
        patch.object(module, "_reader_resident", side_effect=active_blocks),
        patch.object(module, "_use_reader_cuda", return_value=True),
        patch.object(module, "_reader_stream", return_value=None),
        patch.object(module, "_launch_reader_cuda", capture),
    ):
        b, h, w, m, r = shape
        p = TensorSpec(shape=(b, h, w, m * r * (2 if strict else 1)),
                       dtype=dtype, device=torch.device("cuda", 0))
        g = p.new_empty((b, h, w, 2 * m))
        f = p.new_empty((m, r, 3, 3))
        module._packed_spatial._init_fn(p, f, f, strict)
        module._packed_spatial_backward._init_fn(g, p, f, f, strict)
    return launches, images


@cache
def load_base(path):
    spec = spec_from_file_location("lnet._reader_w31b_base", path)
    module = module_from_spec(spec)
    code = compile(path.read_text().replace('"va_reader::', '"va_reader_w31b_base::'),
                   str(path), "exec")
    exec(code, module.__dict__)  # noqa: S102
    return module


def audit(job):
    sm, shape, dtype, strict, base_path, legacy = job
    os.environ.update(
        LNET_READER_VEC="auto", LNET_READER_W25B="auto", LNET_READER_W27B="off",
        LNET_READER_W28B="bwd_hiocc_v2", LNET_READER_W29B="hiocc_dinput_v2",
        LNET_READER_W30B="off", LNET_READER_W31B="off",
    )
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        reader._reader_cuda_module.cache_clear()
        reader._wave31b_forward_plan.cache_clear()
        reader._wave31b_weight_plan.cache_clear()
        base = load_base(base_path)
        base._reader_cuda_module.cache_clear()
        args = sm, shape, dtype, strict, directory, {}
        baseline, base_images = replay(base, *args)
        off, off_images = replay(reader, *args)
        assert off_images == base_images, (sm, shape, dtype, strict, "PTX")
        assert off == baseline, (sm, shape, dtype, strict, "dispatch")
        rows = {"off": off}
        for arm in (*reader._W31B_PRESETS, "auto"):
            os.environ["LNET_READER_W31B"] = arm
            rows[arm], _ = replay(reader, *args)
        assert rows["auto"] == rows["full_w31b"]
        if not strict and shape[1] * shape[2] * shape[0] >= reader._SPECIALIZE_MIN_PIXELS:
            assert all(arm["dinput"] == off["dinput"] for arm in rows.values())
            assert rows["wide_fwd_only"]["dweight"] == off["dweight"]
            assert rows["dweight_vram_split"]["forward"] == off["forward"]
        else:
            assert all(arm == off for arm in rows.values())
        legacy_checks = check_legacy(base, args) if legacy else 0
    return {"sm": sm, "shape": shape, "dtype": str(dtype), "strict": strict, "arms": rows,
            "legacy_checks": legacy_checks}


def check_legacy(base, args):
    os.environ["LNET_READER_W31B"] = "off"
    presets = {27: reader._WAVE27B_PRESETS, 28: reader._WAVE28B_PRESETS,
               29: reader._WAVE29B_PRESETS, 30: reader._WAVE30B_PRESETS}
    keys = {wave: f"LNET_READER_W{wave}B" for wave in presets}
    off = dict.fromkeys(keys.values(), "off")
    os.environ.update(off, LNET_READER_W25B="off")
    assert replay(base, *args) == replay(reader, *args), (*args[:4], "all off")
    os.environ["LNET_READER_W25B"] = "auto"
    checks = 1
    for wave, arms in presets.items():
        os.environ.update(off)
        for arm in (*arms, "auto"):
            os.environ[keys[wave]] = arm
            assert replay(base, *args) == replay(reader, *args), (*args[:4], wave, arm)
            checks += 1
    return checks


def candidate(job):
    sm, h, dtype, op, vec, rows, tile = job
    modes = 128 if h == 7 else 160
    kind = reader._CUDA_TYPES[dtype]
    weight = op == "dweight"
    columns = reader._reader_vector_columns(h, vec) if weight else h
    warps = 4 if weight else 1
    with (
        tempfile.TemporaryDirectory() as scratch,
        patch.object(torch.cuda, "get_device_capability", return_value=divmod(sm, 10)),
        patch.object(reader, "_CudaModule", OfflineModule),
    ):
        module = reader._reader_cuda_module(
            0, (1024, h, h, modes), columns, 0 if weight else 4, 3, (kind,) * 3, strict=False,
            weight_geometry=(tile, rows, warps) if weight else None,
            weight_vec=vec if weight else 1, weight_read_only=vec != 2,
            split_rank=4 if weight else 0,
            spatial_geometry=(rows, warps) if not weight else reader._DEFAULT_SPATIAL_GEOMETRY,
            spatial_vec=1 if weight else vec, updated_spatial=not weight,
            register_filters=-1 if weight else 0, keep_real=-1 if weight else 0,
        )
        assemble(module, sm, Path(scratch))
        meta = module.resources[op]
    return dict(sm=sm, height=h, dtype=str(dtype), op=op, vec=vec, rows=rows, tile=tile,
                resident_warps=resident(meta, sm, warps) * warps, **meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--sm", nargs="+", type=int, choices=SM, default=(86, 89, 90))
    args = parser.parse_args()
    if args.base:
        inputs = itertools.product(args.sm, SHAPES, DTYPES, (False, True))
        jobs = (
            (sm, (1024, *shape), dtype, strict, args.base,
             args.legacy and not strict and shape in SHAPES[:4])
            for sm, shape, dtype, strict in inputs
        )
        run = audit
    else:
        jobs = (
            (90, h, dtype, op, vec, rows, tile)
            for (h, *_), dtype, (op, vec, rows, tile) in itertools.product(
                SHAPES[:4], DTYPES, (
                    ("dweight", 1, 2, 4), ("dweight", 1, 4, 4), ("dweight", 1, 7, 4),
                    ("dweight", 1, 8, 4), ("dweight", 2, 4, 2), ("dweight", 2, 8, 2),
                    ("forward", 2, 4, 4), ("forward", 2, 7, 4), ("forward", 2, 8, 4),
                ),
            )
        )
        run = candidate
    failed = False
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(run, jobs):
            print(json.dumps(result), flush=True)  # noqa: T201
            if args.base:
                failed |= any(any(meta["stack_spill"]) for arm in result["arms"].values()
                              for meta in arm.values())
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
