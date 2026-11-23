"""Check horizontal backward occupancy at the existing H=56 dense layer."""

import argparse
import json
from dataclasses import replace
from functools import lru_cache
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
from bench import compare, measure, sample
from variants import Flag, Shape, original_launch_for_shape

from lnet import general_cuda_scans as scans

# ruff: noqa: SLF001


def launch(
    base: scans.Launch, shape: Shape, device: int, *,
    warps: int, cache: bool,
) -> scans.Launch | None:
    b, h, w, m = shape
    sms, limit, _ = scans.device_limits(device)
    smem = max(16 * 32 * warps if warps > 1 else 0, 8 * w * 32 * warps if cache else 0)
    if smem > limit:
        return None
    key = replace(base.key, warps=warps, cache_line=cache)
    plan = scans.ScanPlan(warps, smem, cache_line=cache)
    count = scans._resident(device, key, smem)
    extra = scans._extra_workspace_budget(device)
    workers = scans._workers(
        b * h, m, 4, warps, sms * count * scans._LONG_HSCAN_WAVES,
        scans._MAX_PARTIAL_BYTES + extra,
    )
    if not cache:
        workers = min(workers, (scans._MAX_SCRATCH_BYTES + extra) // (8 * w * m))
    if workers and count:
        return scans.Launch(plan, key, workers, scans._ceil(workers, warps))
    return None


def probe(batch: int) -> dict:
    torch.manual_seed(731)
    dtype = torch.bfloat16
    shape = (batch, 56, 56, 160)
    source = [sample(shape, dtype, 0.4) for _ in range(2)]
    pole = [sample((shape[-1],), torch.float32, 0.12) for _ in range(4)]
    states = scans.horizontal_forward(pole, *source)
    dy = sample((4, *shape), dtype, 0.1)

    def operation() -> list[torch.Tensor]:
        return scans.horizontal_backward(pole, source, states, dy)

    reference, expected = measure(operation, 7, (), vertical=False)
    result = {"shape": shape, "baseline": reference, "trials": {}}
    for warps, cache in product(scans._SCAN_WARPS, (False, True)):
        @lru_cache(maxsize=8)
        def forced(
            device: int, dtype: torch.dtype, shape: Shape, bound: int, *,
            chosen_warps: int = warps, cached: bool = cache, **flags: Flag,
        ) -> scans.Launch | None:
            base = original_launch_for_shape(device, dtype, shape, bound, **flags)
            return None if base is None else launch(
                base, shape, device, warps=chosen_warps, cache=cached,
            )

        with patch.object(scans, "_launch_for_shape", forced):
            observed, actual = measure(operation, 7, (), vertical=False)
        observed["parity"] = compare(actual, expected)
        result["trials"][f"w{warps}-cache{int(cache)}"] = observed
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(probe(args.batch), indent=2) + "\n")


if __name__ == "__main__":
    main()
