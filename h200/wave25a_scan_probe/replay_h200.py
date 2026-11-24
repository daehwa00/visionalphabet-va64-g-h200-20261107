"""Replay Wave 25A selection against every recorded P1/NR4 launch in #798."""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
from resources import measure

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001

SMS = 132
SHARED_PER_SM = 228 * 1024
REGISTER_FILE = 65536
WARPS_PER_SM = 64
BLOCKS_PER_SM = 32


@cache
def resources(key: scans.Kernel) -> dict:
    return measure((90, key))


def resident(_device: int, key: scans.Kernel, smem: int) -> int:
    row = resources(key)
    if any(row["stack_spill"]):
        return 0
    regs_per_warp = math.ceil(row["registers"] / 8) * 256
    resident_warps = min(REGISTER_FILE // regs_per_warp, WARPS_PER_SM)
    allocated_smem = math.ceil((smem + 1024) / 128) * 128
    return min(BLOCKS_PER_SM, resident_warps // key.warps, SHARED_PER_SM // allocated_smem)


def layouts(*, vertical: bool) -> list[tuple[bool, ...]]:
    # #798 records geometry and launch counts, but omits source-stride flags.
    if vertical:
        return [(flag, flag, False, False) for flag in (False, True)]
    return list(product((False, True), repeat=4))


def recorded_launch(
    row: dict, recorded: dict, layout: tuple[bool, ...],
) -> tuple[scans.Launch, tuple[int, int, int, int]]:
    vertical, backward, coarse = (row[k] for k in ("vertical", "backward", "coarse"))
    h, w, modes = (row[k] for k in ("height", "width", "modes"))
    shape = (1024, h, w, modes)
    plan = scans.ScanPlan(**{k: recorded[k] for k in scans.ScanPlan._fields})
    threads = plan.warps * 32
    compact = coarse if vertical else backward and plan.smem == 4 * w * threads
    vector = False
    if vertical:
        base_plans = scans._scan_plans(
            h, 232448, vertical=True, backward=backward, coarse=coarse, hidden=8,
            itemsize=2, compact=compact,
        )
        packed = [candidate.smem for candidate in base_plans if candidate[2:] == plan[2:]
                  and candidate.warps == plan.warps]
        if packed:
            vector = plan.cache_weights and plan.smem - packed[0] == 11 * 8 * 32 * 2
    key = scans.Kernel(
        torch.bfloat16, plan.warps, backward, vertical=vertical, coarse=coarse,
        hidden8=vertical, vector_weights=vector, precise=vertical and not backward and not vector,
        compact=compact, chunk=0 if plan.full_trajectory else plan.chunk,
        cache_weights=plan.cache_weights, global_checkpoints=plan.global_checkpoints,
        cache_line=plan.cache_line, shared_acc=plan.shared_acc, cache_source=plan.cache_source,
        batch_path=recorded["batch_path"],
        real_interleaved=layout[0], imag_interleaved=layout[1],
        real_transposed=layout[2], imag_transposed=layout[3],
    )
    return scans.Launch(plan, key, recorded["workers"], recorded["partials"]), shape


def replay(job: tuple[dict, dict, tuple[bool, ...]]) -> dict:
    row, recorded, layout = job
    baseline, shape = recorded_launch(row, recorded, layout)
    plan, key, _, _ = baseline
    selected = scans._with_wave25a(baseline, 0, shape)
    changed = selected != baseline
    old_ctas = resident(0, key, plan.smem) if changed else None
    new_ctas = resident(0, selected.key, selected.plan.smem) if changed else None
    return {
        "name": row["name"], "shape": shape, "count": recorded["count"], "layout": layout,
        "compact": key.compact, "vector_weights": key.vector_weights, "warps": plan.warps,
        "batch_path": key.batch_path,
        "selected_cells_per_batch": 2 if selected.key.register_batch2 else
        4 if selected.key.batch_path else 1,
        "workers": baseline.workers, "partials": baseline.partials,
        "fires": changed, "register_path": selected.key.register_path,
        "horizontal_unroll8": selected.key.horizontal_unroll8,
        "horizontal_forward_unroll": selected.key.horizontal_forward_unroll,
        "smem": [plan.smem, selected.plan.smem], "resident_ctas_estimate": [old_ctas, new_ctas],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    events = json.loads(args.profile.read_text())["profiler"]["scan_shape_cuda_events"]
    jobs = [
        (row, plan, layout)
        for row in events
        for plan in row["plans"]
        for layout in layouts(vertical=row["vertical"])
    ]
    with (
        patch.object(scans, "_resident", resident),
        patch.object(scans, "device_limits", lambda _: (SMS, 232448, WARPS_PER_SM)),
        ThreadPoolExecutor(max_workers=8) as pool,
        args.output.open("w") as stream,
    ):
        for row in pool.map(replay, jobs):
            print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    main()
