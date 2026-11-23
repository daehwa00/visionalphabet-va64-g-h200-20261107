"""Measure isolated projection CUDA Graphs using the Wave 17A payload convention."""

import argparse
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wave17a_decoupled_tc_vscan_probe"))
from bench import CASES, graph, pair_graph

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001


def measure(case: int, dtype: torch.dtype, arm: str, repeats: int) -> dict:
    h, w, modes, coarse = CASES[case]
    cells = 128 * (h // (1 + coarse)) * (w // (1 + coarse))
    params = ((modes, 8, 8), (modes, 8), (modes, 2, 8), (modes, 2))
    path = [torch.randn(shape, device="cuda").mul_(0.17) for shape in params]
    source = torch.empty((), dtype=dtype, device="cuda")
    packed = arm != "legacy"
    features = scans._empty_features(source, cells, modes, cell_major=False, packed=packed)
    features.uniform_(-0.5, 0.5)
    grads = [source.new_empty(cells, modes).uniform_(-0.2, 0.2) for _ in range(2)]
    outs = [torch.empty_like(grad) for grad in grads]
    itemsize = source.element_size()
    feature_bytes = features.numel() * itemsize
    output_bytes = 2 * cells * modes * itemsize
    result = pair_graph(
        partial(scans._project_path, features, path, *outs,
                backward=False, precise=True, projection_tc=packed),
        partial(scans._project_path, features, path, *grads,
                backward=True, precise=False, projection_tc=packed),
        repeats, (feature_bytes + output_bytes, 2 * feature_bytes + output_bytes),
    )
    if packed:
        key = scans.Kernel(dtype, 0, backward=True, projection=True, projection_tc=True,
                           hidden8=True, packed_features=True)
        launch_key, groups, smem = scans._plan_projection(
            source.device.index or 0, key, cells, modes, 8,
        )
        partials = torch.zeros((groups, 90, modes), dtype=torch.float32, device=source.device)
        result["reduction"] = graph(
            partial(scans._reduce_partials, partials, path, 8, projection=True), repeats,
        )
        result["launch"] = {"warps": launch_key.warps, "groups": groups, "smem": smem}
    return {"case": case, "shape": (128, h, w, modes, coarse), "dtype": str(dtype),
            "arm": arm, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", required=True, choices=("bfloat16", "float16"))
    parser.add_argument("--arm", required=True, choices=("legacy", "wave17a", "wave18a"))
    parser.add_argument("--case", type=int, nargs="+", default=list(range(len(CASES))))
    parser.add_argument("--tile", type=int, nargs=2, metavar=("WARPS", "CELLS"))
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    original_plan = scans._plan_projection

    def plan(device: int, base: scans.Kernel, cells: int, modes: int, hidden: int) -> tuple:
        if args.tile is None or not base.projection_tc:
            return original_plan(device, base, cells, modes, hidden)
        warps, tile_cells = args.tile
        key = replace(base, warps=warps)
        if args.arm == "wave18a":
            key = replace(key, projection_cells=tile_cells)
            smem = scans._tc_smem(warps, base.backward, tile_cells)
        else:
            smem = scans._tc_smem(warps, base.backward)
        sms, _, _ = scans.device_limits(device)
        resident = scans._resident(device, key, smem)
        if not resident:
            raise RuntimeError(f"infeasible tile: {args.tile}")
        tiles = scans._ceil(modes, warps)
        groups = max(1, min(scans._ceil(cells, tile_cells), scans._ceil(4 * sms * resident, tiles)))
        if base.backward:
            bytes_per_group = (11 * hidden + 2) * modes * 4
            groups = min(groups, max(1, scans._MAX_PARTIAL_BYTES // bytes_per_group))
        return key, groups, smem

    torch.manual_seed(431)
    with patch.object(scans, "_plan_projection", plan), args.out.open("a") as log:
        for case in args.case:
            result = measure(case, getattr(torch, args.dtype), args.arm, args.repeats)
            print(json.dumps({**result, "tile": args.tile}), file=log, flush=True)


if __name__ == "__main__":
    main()
