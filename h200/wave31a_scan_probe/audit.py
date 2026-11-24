"""Compile Wave 31A and replay SM 9.0 launch selection with CUDA hidden."""

from __future__ import annotations

import argparse
import ctypes as ct
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

from lnet import general_cuda_scans as scans

if TYPE_CHECKING:
    from types import ModuleType

# ruff: noqa: FBT001, S101, S603, S607, SLF001, T201
BASE = "4122ea206011d539b2f29597e20f6e982a0333ba"
PTXAS = "/usr/local/cuda-12.8/bin/ptxas"
ARCHES = (86, 89, 90)
NEW_ARCHES = (*ARCHES, 100)
DTYPES = (torch.bfloat16, torch.float16)
LAYOUTS = tuple(product((False, True), repeat=2))
SMS, SMEM_LIMIT, WARP_LIMIT = 132, 227 * 1024, 64
DEVICES = {
    86: (82, 99 * 1024, 48),
    89: (128, 99 * 1024, 48),
    90: (SMS, SMEM_LIMIT, WARP_LIMIT),
    100: (148, 227 * 1024, 64),
}


def baseline(folder: Path) -> ModuleType:
    path = folder / "baseline.py"
    path.write_bytes(subprocess.check_output([
        "git", "show", f"{BASE}:src/lnet/general_cuda_scans.py",
    ]))
    spec = importlib.util.spec_from_file_location("lnet.wave31a_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Wave 30 baseline")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def old_keys(profile: dict) -> set[scans.Kernel]:
    keys = set()
    for row in profile["scan_shape_cuda_events"]:
        flags = next(flag for flag in profile["scan_dispatch"] if all(
            flag[k] == row[k]
            for k in ("height", "width", "modes", "vertical", "backward", "coarse")
        ))
        vertical, backward, coarse = (row[k] for k in ("vertical", "backward", "coarse"))
        for plan in row["plans"]:
            for dtype, (ri, ii), (rt, it) in product(
                DTYPES, LAYOUTS, ((False, False),) if vertical else LAYOUTS,
            ):
                keys.add(scans.Kernel(
                    dtype, plan["warps"], backward, vertical=vertical, coarse=coarse,
                    hidden8=vertical, real_interleaved=ri, imag_interleaved=ii,
                    real_transposed=rt, imag_transposed=it, compact=flags["compact"],
                    precise=vertical and not backward and not flags["vector_weights"],
                    chunk=0 if plan["full_trajectory"] else plan["chunk"],
                    cache_weights=plan["cache_weights"], cache_line=plan["cache_line"],
                    cache_source=plan["cache_source"], shared_acc=plan["shared_acc"],
                    global_checkpoints=plan["global_checkpoints"], batch_path=plan["batch_path"],
                    **{f: flags[f] for f in (
                        "vector_weights", "register_path", "register_batch2", "horizontal_unroll8",
                        "horizontal_forward_unroll", "backward_batch2", "backward_batch_tail",
                    )},
                ))
    keys.update(scans.Kernel(
        torch.float32, 4, backward=True, vertical=vertical, reduction=True,
    ) for vertical in (False, True))
    return keys


def stages(
    states: torch.Tensor, coarse: bool, preset: str, layout: tuple[bool, bool],
) -> list[tuple[scans.Kernel, int, int]]:
    _, b, h, w, _ = states.shape
    lines = b * (w // (1 + coarse))
    cells = lines * (h // (1 + coarse))
    fused = preset == "saved_fused"
    result = []
    for backward in (False, True):
        scan = scans._wave31a_scan_key(
            states, coarse, fused, backward=backward, compact=coarse and backward,
            precise=False, interleaved=layout if fused else (False, False),
        )
        result.append((scan, lines, (94 if fused else 4) if backward else 0))
        if not fused:
            project = scans._wave31a_project_key(
                states, coarse, scans._WAVE31A_PRESETS[preset], backward=backward,
                precise=False, interleaved=layout,
            )
            result.append((project, cells, 90 if backward else 0))
    return result


def compile_one(job: tuple[int, scans.Kernel]) -> tuple[tuple[int, scans.Kernel], dict]:
    sm, key = job
    ptx = scans._compile_ptx(key, divmod(sm, 10)).value
    with TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "scan.ptx", Path(tmp) / "scan.cubin"
        src.write_bytes(ptx)
        report = subprocess.run(
            [PTXAS, "-v", "-arch", f"sm_{sm}", str(src), "-o", str(dst)],
            capture_output=True, check=True, text=True,
        ).stderr
    regs = re.search(r"Used (\d+) registers", report)
    spill = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", report,
    )
    if regs is None or spill is None:
        raise ValueError(report)
    return (job, {
        "sm": sm, "kernel": scans._kernel_name(key), "registers": int(regs[1]),
        "stack_spill": tuple(map(int, spill.groups())),
        "packed_loads": ptx.count(b"ld.global.v2.b32"),
        "packed_stores": ptx.count(b"st.global.v2.b32"),
        "key": {**vars(key), "dtype": str(key.dtype)},
    })


def off_one(job: tuple[int, scans.Kernel], base: ModuleType) -> dict:
    sm, key = job
    old = base.Kernel(**{field.name: getattr(key, field.name) for field in fields(base.Kernel)})
    cap = divmod(sm, 10)
    assert base._kernel_source(old) == scans._kernel_source(key)
    equal = scans._compile_ptx(key, cap).value == base._compile_ptx(old, cap).value
    return {
        "sm": sm, "key": {**vars(key), "dtype": str(key.dtype)}, "byte_identical": equal,
    }


def record_dispatch(
    calls: Counter, reports: dict, sm: int, layout: tuple[bool, bool],
) -> list[dict]:
    sms, shared_limit, warp_limit = DEVICES[sm]

    def resident(_device: int, key: scans.Kernel, smem: int) -> int:
        report = reports[sm, key]
        if any(report["stack_spill"]):
            return 0
        warp_registers = math.ceil(report["registers"] / 8) * 256
        warps = min(warp_limit, 4 * (16384 // warp_registers))
        shared = math.ceil((smem + 1024) / 128) * 128
        return min(32, warps // key.warps, (shared_limit + 1024) // shared)

    scans._wave31a_plan.cache_clear()
    result = []
    with (
        patch.object(scans, "device_limits", return_value=(sms, shared_limit, warp_limit)),
        patch.object(scans, "_resident", resident),
    ):
        for (h, w, m, coarse), dtype, preset in product(calls, DTYPES, scans._WAVE31A_PRESETS):
            shape = (1024, h, w, m)
            states = torch.empty((4, *shape), dtype=dtype, device="meta")
            out_h = h // (1 + coarse)
            items = stages(states, coarse, preset, layout)
            if sm == 90 and not any(layout) and preset == "saved_decoupled_v2":
                pure = next(entry for entry in items if entry[0].saved_scan and entry[0].backward)
                items.extend((replace(pure[0], saved_direction=d), *pure[1:]) for d in (1, 2))
            for base, work, n_fields in items:
                key, grid, rows, smem = scans._wave31a_plan(
                    base, 0, work, m, out_h, 2, n_fields,
                )
                ctas = resident(0, key, smem)
                result.append({
                    "shape": shape, "coarse": coarse, "preset": preset, "warps": key.warps,
                    "count": calls[h, w, m, coarse], "grid": grid,
                    "layout": layout, "dynamic_smem": smem,
                    "ctas_per_sm": ctas, "warps_per_sm": ctas * key.warps,
                    "waves": grid / (sms * ctas), "partials_bytes": rows * n_fields * m * 4,
                    **reports[sm, key],
                })
    return result


def new_keys(calls: Counter) -> set[scans.Kernel]:
    keys = set()
    for (h, w, m, coarse), dtype, preset, layout in product(
        calls, DTYPES, scans._WAVE31A_PRESETS, LAYOUTS,
    ):
        states = torch.empty((4, 1024, h, w, m), dtype=dtype, device="meta")
        for base, _, _ in stages(states, coarse, preset, layout):
            for key in scans._wave31a_candidates(base):
                keys.add(key)
                if key.saved_scan and key.backward and not key.saved_fused:
                    keys.add(replace(key, compact=False))
                    keys.update(replace(key, saved_direction=d) for d in (1, 2))
                if key.saved_fused and key.backward:
                    keys.add(replace(key, compact=False))
    keys.add(scans.Kernel(
        torch.float32, 4, backward=True, vertical=True, reduction=True, projection_reduction=True,
    ))
    return keys


def dump(path: Path, result: list[dict]) -> None:
    path.write_text(json.dumps(result, indent=2) + "\n")


def load_compiler(path: Path) -> dict:
    reports = {}
    for row in json.loads(path.read_text()):
        key = scans.Kernel(**{**row["key"], "dtype": getattr(torch, row["key"]["dtype"][6:])})
        row["key"] = {**vars(key), "dtype": str(key.dtype)}
        reports[row["sm"], key] = row
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--reuse-off", action="store_true")
    parser.add_argument("--reuse-compiler", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    major, minor = ct.c_int(), ct.c_int()
    scans._nvrtc().nvrtcVersion(ct.byref(major), ct.byref(minor))
    assert (major.value, minor.value) == (12, 8)
    profile = json.loads(args.profile.read_text())["profiler"]
    calls = Counter({
        (r["height"], r["width"], r["modes"], r["coarse"]): r["count"]
        for r in profile["scan_shape_cuda_events"] if r["vertical"] and r["backward"]
    })
    assert sum(calls.values()) == 11
    with TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        keys = old_keys(profile)
        if args.reuse_off:
            off = json.loads((args.output / "off.json").read_text())
        else:
            base = baseline(Path(tmp))
            off = list(pool.map(lambda job: off_one(job, base), product(ARCHES, keys)))
            dump(args.output / "off.json", off)
        assert len(off) == len(ARCHES) * len(keys)
        assert all(row["byte_identical"] for row in off)
        print(f"{len(off)} byte-identical off PTX images", flush=True)
        jobs = set(product(NEW_ARCHES, new_keys(calls)))
        reports = load_compiler(args.output / "compiler.json") if args.reuse_compiler else {}
        reports.update(pool.map(compile_one, jobs - reports.keys()))
        reports = {job: reports[job] for job in jobs}
    dump(args.output / "compiler.json", list(reports.values()))
    records = [
        row
        for sm, layout in product(NEW_ARCHES, LAYOUTS)
        for row in record_dispatch(calls, reports, sm, layout)
    ]
    dump(args.output / "dispatch.json", records)
    assert all(not any(row["stack_spill"]) for row in records)
    assert all(not any(reports[sm, scans.Kernel(
        torch.float32, 4, backward=True, vertical=True, reduction=True, projection_reduction=True,
    )]["stack_spill"]) for sm in NEW_ARCHES)
    used = {
        (row["sm"], tuple(row["key"].items()))
        for row in records if row["key"]["saved_direction"] == 0
    }
    print(f"{len(reports)} CUDA 12.8 assemblies")
    print(f"{len(used)} selected images clean")


if __name__ == "__main__":
    main()
