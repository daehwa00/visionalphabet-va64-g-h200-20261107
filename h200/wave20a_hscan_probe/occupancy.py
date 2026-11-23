"""Render a theoretical CTA occupancy table for the offline kernel matrix."""

# ruff: noqa: T201, SLF001
import argparse
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import torch

from lnet import general_cuda_scans as scans

# SM resources (KiB shared memory, maximum warps, maximum CTAs).
RESOURCES = {"86": (100, 48, 16), "89": (100, 48, 24), "90": (228, 64, 32)}
PROFILES = {"86": (128, 80, 6), "89": (128, 128, 72), "90": (1024, 132, 50)}
SHAPES = (
    (56, 56, 160), (28, 28, 160), (14, 14, 160), (7, 7, 128),
    (64, 96, 160), (37, 53, 192), (128, 192, 160),
)


def resources(key: scans.Kernel, smem: int, row: dict) -> int:
    total_kib, max_warps, max_blocks = RESOURCES[row["sm"]]
    reg_per_warp = scans._ceil(row["registers"] * 32, 256) * 256
    shared_per_cta = scans._ceil(smem, 256) * 256 + 1024
    return min(max_blocks, max_warps // key.warps, 65536 // (reg_per_warp * key.warps),
               total_kib * 1024 // shared_per_cta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    args = parser.parse_args()
    table = {}
    for line in args.matrix.read_text().splitlines():
        row = json.loads(line)
        if row["layout"] == ["plain", "plain"]:
            lookup = tuple(row[k] for k in (
                "sm", "dtype", "warps", "backward", "compact", "cache", "paired", "index32",
            ))
            table[lookup] = row

    print("| SM | B x H x W x M | phase | compact | warps | SMEM B | registers off → auto "
          "| CTAs/SM off → auto | LDG/STG/LDS/STS/FFMA/MUFU off → auto |")
    print("|---|---|---|---|---:|---:|---:|---:|---|")
    for sm, shape, (backward, compact) in itertools.product(
        PROFILES, SHAPES, ((False, False), (True, False), (True, True)),
    ):
        if compact and shape[1] % 2:
            continue
        batch, sms, l2_mib = PROFILES[sm]
        total_kib, max_warps, _ = RESOURCES[sm]
        l2 = l2_mib * 1024**2

        def profile(key: scans.Kernel, *, paired: bool = False, target: str = sm) -> dict:
            return table[target, str(key.dtype), key.warps, key.backward,
                         key.compact, key.cache_line, paired, key.index32]

        with (
            patch.object(scans, "device_limits", return_value=(sms, (total_kib - 1) * 1024,
                                                             max_warps)),
            patch.object(scans, "_horizontal_cache_budget", return_value=l2 * 7 // 8),
            patch.object(scans, "_extra_workspace_budget", return_value=l2 if sm == "90" else 0),
            patch.object(
                scans, "_resident", side_effect=lambda _d, k, s: resources(k, s, profile(k)),
            ),
        ):
            scans._launch_for_shape.cache_clear()
            b, h, w, m = batch, *shape
            launch = scans._launch_for_shape(
                0, torch.bfloat16, (b, h, w, m), 4 * b * h * w * m,
                backward=backward, vertical=False, coarse=False, hidden=0,
                precise=False, compact=compact,
            )
        plan, key, _, _ = launch
        off = profile(key)
        candidate = profile(key, paired=True) if backward and compact and key.cache_line else off
        a, b = resources(key, plan.smem, off), resources(key, plan.smem, candidate)
        auto = candidate if b >= a and candidate["stack"] == 0 else off
        before, after = ("/".join(str(n) for n in item["sass"].values()) for item in (off, auto))
        print(f"| {sm} | {batch} x {h} x {w} x {m} | {'bwd' if backward else 'fwd'} "
              f"| {compact} | {key.warps} | {plan.smem} "
              f"| {off['registers']} → {auto['registers']} "
              f"| {a} → {resources(key, plan.smem, auto)} | {before} → {after} |")


if __name__ == "__main__":
    main()
