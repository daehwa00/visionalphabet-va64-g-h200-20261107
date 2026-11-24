"""Replay #806 dispatches and verify CUDA 12.8 compiler resources off-device."""

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import cache
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import torch
from replay_h200 import layouts, recorded_launch
from resources import measure

from lnet import general_cuda_scans as scans

# ruff: noqa: S607, SLF001, T201

PRESETS = ("off", "auto", "wave_plan", "bwd_chunk", "bwd_batch2", "hscan_unroll")


@cache
def resources(sm: int, key: scans.Kernel) -> dict:
    return measure((sm, key))


def resident(_device: int, key: scans.Kernel, smem: int) -> int:
    row = resources(90, key)
    if any(row["stack_spill"]):
        return 0
    reg_warps = 65536 // (math.ceil(row["registers"] / 8) * 256)
    allocated = math.ceil((smem + 1024) / 128) * 128
    return min(32, min(reg_warps, 64) // key.warps, (228 * 1024) // allocated)


def records(profile: dict) -> Iterator[tuple]:
    for row in profile["scan_shape_cuda_events"]:
        for index, plan in enumerate(row["plans"]):
            for layout in layouts(vertical=row["vertical"]):
                launch, shape = recorded_launch(row, plan, layout)
                flags = next(
                    f for f in profile["scan_dispatch"]
                    if all(f[k] == row[k] for k in (
                        "height", "width", "modes", "vertical", "backward", "coarse",
                    ))
                    and f["compact"] == (launch.key.compact and launch.key.backward)
                )
                launch = launch._replace(key=replace(launch.key, **{
                    f: flags[f] for f in (
                        "vector_weights", "register_path", "register_batch2", "horizontal_unroll8",
                        "compact",
                    )
                }, precise=row["vertical"] and not row["backward"] and not flags["vector_weights"]))
                yield row["name"], index, plan["count"], layout, shape, launch


def baseline_module(folder: Path) -> ModuleType:
    name = "lnet.wave27a_baseline"
    path = folder / "baseline.py"
    path.write_bytes(subprocess.check_output([
        "git", "show", "a670a752a01ecf32e90b6016777750569e422148:src/lnet/general_cuda_scans.py",
    ]))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def certify(job: tuple[int, scans.Kernel, ModuleType | None]) -> dict:
    sm, key, baseline = job
    row = resources(sm, key)
    if any(row["stack_spill"]):
        raise ValueError(row)
    if baseline is not None:
        cap = divmod(sm, 10)
        fields = vars(key).copy()
        fields.pop("backward_batch2")
        old_ptx = baseline._compile_ptx(baseline.Kernel(**fields), cap).value
        if scans._compile_ptx(key, cap).value != old_ptx:
            raise ValueError(("PTX mismatch", sm, key))
    return row


def summarize(launch: scans.Launch, shape: tuple) -> dict:
    plan, key, workers, partials = launch
    ctas = resident(0, key, plan.smem)
    grid = math.ceil(workers / plan.warps) * math.ceil(shape[-1] / 32)
    waves = grid / (132 * ctas)
    fields = (94 if key.vertical else 4) if key.backward else 0
    scratch = scans._scratch_per_worker(
        plan, shape[1] if key.vertical else shape[2], vertical=key.vertical,
        coarse=key.coarse, backward=key.backward, itemsize=2, compact=key.compact,
    )
    return {
        **plan._asdict(), "workers": workers, "partials": partials,
        "registers": resources(90, key)["registers"], "ctas_per_sm": ctas, "grid": grid,
        "waves": waves, "wave_fill_pct": 100 * waves / math.ceil(waves),
        "last_wave_fill_pct": 100 * (waves - math.ceil(waves) + 1),
        "partial_bytes": partials * shape[-1] * fields * 4,
        "scratch_bytes": scratch * workers * shape[-1],
        "backward_batch2": key.backward_batch2,
        "hscan_unroll": 8 if key.horizontal_unroll8 else key.horizontal_forward_unroll,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as folder:
        baseline = baseline_module(Path(folder))
    profile = json.loads(args.profile.read_text())["profiler"]
    old_keys = set()
    new_keys = set()
    rows = []
    entries = list(records(profile))
    with patch.object(scans, "_resident", resident), patch.object(
        scans, "device_limits", lambda _: (132, 232448, 64),
    ):
        for preset in PRESETS:
            os.environ["LNET_SCAN_W27A"] = preset
            for name, index, count, layout, shape, original in entries:
                candidate = scans._with_wave27a(original, 0, shape)
                for dtype in (torch.bfloat16, torch.float16):
                    old_keys.add(replace(original.key, dtype=dtype))
                    new_keys.add(replace(candidate.key, dtype=dtype))
                if preset != "off":
                    rows.append({
                        "name": name, "index": index, "count": count, "layout": layout,
                        "preset": preset, "old": summarize(original, shape),
                        "new": summarize(candidate, shape),
                    })
    (args.output / "dispatch.json").write_text(json.dumps(rows, indent=2))
    jobs = [(sm, key, baseline if key in old_keys else None)
            for sm, key in product((86, 89, 90), old_keys | new_keys)]
    with (
        ThreadPoolExecutor(max_workers=12) as pool,
        (args.output / "compiler.jsonl").open("w") as out,
    ):
        for result in pool.map(certify, jobs):
            print(json.dumps(result), file=out, flush=True)
    print(f"{len(old_keys) * 3} identical PTX images; {len(jobs)} clean assemblies")


if __name__ == "__main__":
    main()
