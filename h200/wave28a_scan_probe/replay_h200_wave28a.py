"""Replay Whole H200 #810 scan dispatches with CUDA 12.8 compiler checks."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import torch
from replay import records, resources, summarize

from lnet import general_cuda_scans as scans

# ruff: noqa: S603, S607, SLF001

BASE = "42b343cbbf62f6accc6ff3a8b25b8be341400853"
PRESETS = ("off", "wp_all_waves", "wp_batch2", "wp_batch2_all", "wp_vscan_opt", "auto")
ARCHES = (86, 89, 90)
DTYPES = (torch.bfloat16, torch.float16)


def resident(_device: int, key: scans.Kernel, smem: int) -> int:
    row = resources(90, key)
    if any(row["stack_spill"]):
        return 0
    reg_warps = 256 // ((row["registers"] + 7) // 8)
    return min(32, min(reg_warps, 64) // key.warps, (228 * 1024) // (smem + 1024))


def baseline_module(folder: Path) -> ModuleType:
    path = folder / "baseline.py"
    path.write_bytes(subprocess.check_output([
        "git", "show", f"{BASE}:src/lnet/general_cuda_scans.py",
    ]))
    spec = importlib.util.spec_from_file_location("lnet.wave28a_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compile_and_check(
    job: tuple[int, scans.Kernel], baseline: ModuleType, off: set[scans.Kernel],
) -> dict:
    sm, key = job
    row = resources(sm, key)
    row["off_ptx_equal"] = None
    if key in off:
        fields = vars(key).copy()
        fields.pop("backward_batch_tail")
        cap = divmod(sm, 10)
        expected = baseline._compile_ptx(baseline.Kernel(**fields), cap).value
        row["off_ptx_equal"] = scans._compile_ptx(key, cap).value == expected
    return row


def original_launch(name: str, launch: scans.Launch) -> scans.Launch:
    # #810 measured Wave 27A wave_plan. Recover the six #806 original
    # worker counts to exercise W28 from W27=off, as in a bundled A/B arm.
    workers = {
        "vscan_bwd_56_d": 1690, "vscan_bwd_56_c": 1692,
        "vscan_bwd_28_d": 1692, "vscan_bwd_28_c": 1692,
        "vscan_bwd_14_d": 636, "vscan_bwd_14_c": 636,
    }.get(name)
    if workers is None:
        return launch
    return launch._replace(workers=workers, partials=workers // launch.plan.warps)


def precompile_hopper(entries: list[tuple]) -> None:
    h200_keys = set()
    for _, _, _, _, shape, launch in entries:
        for dtype in DTYPES:
            key = replace(launch.key, dtype=dtype)
            h200_keys.add(key)
            if key.vertical and key.backward:
                height = shape[1]
                h200_keys.add(replace(
                    key, backward_batch2=True,
                    backward_batch_tail=bool(launch.plan.chunk % (2 * (1 + key.coarse))),
                ))
                if height == 56:
                    h200_keys.add(replace(key, chunk=14))
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda key: resources(90, key), h200_keys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as folder:
        baseline = baseline_module(Path(folder))
    profile = json.loads(args.profile.read_text())["profiler"]
    old_keys, new_keys = set(), set()
    rows = []
    entries = list(records(profile))
    precompile_hopper(entries)
    with (
        patch.object(scans, "_resident", resident),
        patch.object(scans, "device_limits", return_value=(132, 232448, 64)),
        patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        patch.dict(os.environ, {"LNET_SCAN_W27A": "off"}),
    ):
        for name, index, count, layout, shape, recorded in entries:
            for dtype in DTYPES:
                launch = original_launch(name, recorded)
                launch = launch._replace(key=replace(launch.key, dtype=dtype))
                for preset in PRESETS:
                    os.environ["LNET_SCAN_W28A"] = preset
                    selected = scans._with_wave28a(launch, 0, shape)
                    new_keys.add(selected.key)
                    if preset == "off":
                        old_keys.add(launch.key)
                    rows.append({
                        "name": name, "index": index, "count": count, "layout": layout,
                        "preset": preset, "dtype": str(dtype),
                        **summarize(selected, shape),
                        "backward_batch_tail": selected.key.backward_batch_tail,
                    })
                if shape[1] == 56 and launch.key.vertical and launch.key.backward:
                    os.environ["LNET_SCAN_W27A"] = "bwd_batch2"
                    old_keys.add(scans._with_wave27a(launch, 0, shape).key)
                    os.environ["LNET_SCAN_W27A"] = "off"
    (args.output / "dispatch.json").write_text(json.dumps(rows, indent=2))
    jobs = list(product(ARCHES, old_keys | new_keys))
    with (
        ThreadPoolExecutor(max_workers=12) as pool,
        (args.output / "compiler.jsonl").open("w") as out,
    ):
        reports = list(pool.map(lambda job: compile_and_check(job, baseline, old_keys), jobs))
        for row in reports:
            out.write(json.dumps(row) + "\n")
    failures = [r for r in reports if any(r["stack_spill"]) or r["off_ptx_equal"] is False]
    identical = sum(r["off_ptx_equal"] is True for r in reports)
    print(  # noqa: T201
        f"{len(reports)} ptxas jobs, {identical} identical PTX images, "
        f"{len(failures)} failed checks",
    )
    if failures:
        raise ValueError(f"{len(failures)} failed checks; see compiler.jsonl")


if __name__ == "__main__":
    main()
