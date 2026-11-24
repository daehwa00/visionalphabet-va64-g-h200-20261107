"""Replay #821 scan dispatches using CUDA 12.8 and no CUDA device."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from replay import records, resident, resources, summarize
from replay_h200_wave28a import baseline_module, original_launch

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001

PRESETS = ("off", "surgical_w29", "chunk28_nocache", "smem_full_56", "l2_full_56", "auto")
ARCHES = (86, 89, 90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as folder, patch(
        "replay_h200_wave28a.BASE", "6853f51c1eda2ecc51201b32e276a38efdf6de96",
    ):
        baseline = baseline_module(Path(folder))
    profile = json.loads(args.profile.read_text())["profiler"]
    entries = list(records(profile))

    def replay(entry: tuple) -> list[tuple]:
        name, index, count, layout, shape, recorded = entry
        rows = []
        for dtype in (torch.bfloat16, torch.float16):
            launch = original_launch(name, recorded)
            launch = launch._replace(key=replace(launch.key, dtype=dtype))
            for preset in PRESETS:
                selected = scans._with_wave30a(launch, 0, shape, preset=preset)
                plan = selected.plan
                row = {
                    "name": name, "index": index, "count": count, "layout": layout,
                    "dtype": str(dtype), "preset": preset, **summarize(selected, shape),
                    "segments": scans._ceil(shape[1], plan.chunk) if (
                        selected.key.vertical and selected.key.backward
                    ) else None,
                    "register_path": selected.key.register_path,
                    "readonly": selected.key.backward_readonly,
                    "global_seg_bytes": (
                        shape[1] * selected.workers * shape[-1] * 16
                        if selected.key.global_trajectory else 0
                    ),
                    "uncached_trajectory": selected.key.uncached_trajectory,
                }
                row["warps_per_sm"] = row["ctas_per_sm"] * plan.warps
                rows.append((row, selected.key))
        return rows

    with (
        patch.object(scans, "_resident", resident),
        patch.object(scans, "device_limits", return_value=(132, 232448, 64)),
        patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        patch.dict(os.environ, {"LNET_SCAN_W27A": "off", "LNET_SCAN_W28A": "off"}),
        ThreadPoolExecutor(max_workers=12) as pool,
    ):
        reports = [row for rows in pool.map(replay, entries) for row in rows]
    (args.output / "dispatch.json").write_text(json.dumps([r for r, _ in reports], indent=2))
    keys = {k for _, k in reports}
    off = {k for r, k in reports if r["preset"] == "off"}
    reductions = {
        scans.Kernel(torch.float32, 4, backward=True, vertical=vertical, reduction=True)
        for vertical in (False, True)
    }
    keys.update(reductions)
    off.update(reductions)

    def certify(job: tuple[int, scans.Kernel]) -> dict:
        sm, key = job
        row = dict(resources(sm, key))
        if key in off:
            fields = vars(key).copy()
            fields.pop("uncached_trajectory")
            fields.pop("global_trajectory")
            cap = divmod(sm, 10)
            row["off_ptx_equal"] = (
                scans._compile_ptx(key, cap).value
                == baseline._compile_ptx(baseline.Kernel(**fields), cap).value
            )
        return row

    compiler = args.output / "compiler.jsonl"
    with ThreadPoolExecutor(max_workers=12) as pool, compiler.open("w") as out:
        for row in pool.map(certify, product(ARCHES, keys)):
            print(json.dumps(row), file=out, flush=True)
    failures = [
        row for row in map(json.loads, compiler.read_text().splitlines())
        if any(row["stack_spill"]) or row.get("off_ptx_equal") is False
    ]
    print(f"{len(keys) * 3} ptxas jobs, {len(off) * 3} off checks, {len(failures)} failures")  # noqa: T201
    if failures:
        raise ValueError(f"{len(failures)} compiler checks failed")


if __name__ == "__main__":
    main()
