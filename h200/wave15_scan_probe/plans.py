# ruff: noqa: SLF001, T201
"""Record attempted occupancy and scratch budgets without allocating image tensors."""

import json
from itertools import product

import torch

from lnet import general_cuda_scans as scans

ARMS = {
    "wave14": {},
    "scratch": {"local_scratch": True},
    "single": {"state_layout": 2, "single_store": True, "local_scratch": True},
    "shared2": {"state_layout": 2, "shared_hidden": 2},
}


def main() -> None:
    device = torch.cuda.current_device()
    sms, smem_limit, max_warps = scans.device_limits(device)
    l2_bytes = scans._l2_size(device)
    dtype = torch.bfloat16
    geometries = product((128, 1024), (7, 14, 28, 56), (False, True), (False, True))
    for b, h, coarse, backward in geometries:
        if coarse and h % 2:
            continue
        for name, opts in ARMS.items():
            if name in ("single", "shared2") and not backward:
                continue
            m = 160
            shape = (b, h, h, m)
            launch = scans._launch_for_shape(
                device, dtype, shape, 4 * b * h * h * m, vertical=True, backward=backward,
                coarse=coarse, compact=backward and coarse, hidden=8, precise=not backward,
                **opts,
            )
            if launch is None:
                continue
            plan, key, workers, _ = launch
            resident = scans._resident(device, key, plan.smem)
            grids = scans._ceil(workers, plan.warps) * scans._ceil(m, 32)
            rows = h if key.single_store else h // (1 + coarse)
            span = 8 if key.single_store else 4
            scratch = 0 if plan.cache_line else workers * m * rows * span * dtype.itemsize
            checkpoint = (
                workers * m * ((h - 1) // plan.chunk) * 16 if plan.global_checkpoints else 0
            )
            print(json.dumps({
                "gpu": torch.cuda.get_device_name(device), "shape": shape, "arm": name,
                "backward": backward, "coarse": coarse, "sms": sms, "ctas": grids,
                "warps": plan.warps, "resident_ctas_per_sm": resident,
                "max_warps_per_sm": max_warps,
                "resident_warp_fraction": resident * plan.warps / max_warps,
                "grid_resident_waves": grids / (sms * resident), "workers": workers,
                "smem_limit_bytes": smem_limit, "smem_bytes": plan.smem, "l2_bytes": l2_bytes,
                "scratch_bytes": scratch, "checkpoint_bytes": checkpoint,
                "scratch_l2_fraction": (scratch + checkpoint) / l2_bytes,
            }), flush=True)


if __name__ == "__main__":
    main()
