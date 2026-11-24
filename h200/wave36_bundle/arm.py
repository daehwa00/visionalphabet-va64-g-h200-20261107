"""Execute a single Wave 36 GPU benchmark arm."""
# ruff: noqa: BLE001

import argparse
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch

from h200.liverms_v2_opt import bench_h200 as bench
from h200.wave36_bundle.metrics import BUCKETS, EXPECTED_11, SUBGROUPS, classify
from lnet.model_w36b import model_w36b_features

VARIANT = "17-p1-nr4-1024d-v3"


def profile_step(step: Callable[[], torch.Tensor]) -> dict:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        step()
        torch.cuda.synchronize()

    kernels = Counter()
    counts = Counter()
    buckets = dict.fromkeys(BUCKETS, 0.0)
    groups = {group: {"self_cuda_ms": 0.0, "calls": 0} for group in SUBGROUPS}
    generic_stem = [event for event in profiler.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                    and event.name.startswith("stem_conv_tensor_stages")]
    # Stage 0 is the only fused-conv stage. Its tensor-core kernel is
    # specialized twice under the same symbol, with forward before backward.
    stem_directions = {}
    if len(generic_stem) == 2:
        for event, direction in zip(
            sorted(generic_stem, key=lambda entry: entry.time_range.start), ("fwd", "bwd"),
            strict=True,
        ):
            stem_directions[id(event)] = f"fused_stem_stage0_{direction}"

    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        elapsed = event.self_device_time_total / 1000
        name = event.name
        bucket, subgroup = classify(name)
        subgroup = stem_directions.get(id(event), subgroup)
        buckets[bucket] += elapsed
        if subgroup:
            groups[subgroup]["self_cuda_ms"] += elapsed
            groups[subgroup]["calls"] += event.count
        kernels[name] += elapsed
        counts[name] += event.count

    total = sum(buckets.values())
    graph_launches = sum(calls for name, calls in counts.items() if "graphlaunch" in name.lower())
    graph_arm = "cuda_graphs" in model_w36b_features()
    coverage = total > 0 and (
        not graph_arm or bool(buckets["general_vscan"] and buckets["factorized_reader"])
    )
    return {
        "buckets": buckets if coverage else {},
        "subkernels": groups if coverage else {},
        "total_self_cuda_ms": total if coverage else None,
        "profile_cuda_coverage": coverage,
        "profile_graph_launches": graph_launches,
        "top_kernels": [{"name": name, "calls": counts[name], "self_cuda_ms": elapsed}
                        for name, elapsed in kernels.most_common(20)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.variant, args.batch, args.profile = VARIANT, 1024, True
    try:
        row = bench.train_arm(args, profiler=profile_step)
        profile = row.pop("profiler", {})
        row.update(status="ok", **profile)
        row.update(
            img_per_sec=row["images_per_second"],
            loss_finite=row["finite_losses"],
            grad_finite=row["finite_gradients"],
            finite=row["finite_losses"] and row["finite_gradients"],
            loss_value=row["losses"]["last"],
        )
        total = row.get("total_self_cuda_ms")
        row["wall_minus_cuda_ms"] = None if total is None else row["median_ms"] - total
        groups = row.get("subkernels", {})
        row["wave36a_11_call_check"] = {
            name: groups[name]["calls"] == 11 for name in EXPECTED_11 if name in groups
        }
    except Exception as exc:
        row = {"status": "oom" if "out of memory" in str(exc).lower() else "error",
               "error": traceback.format_exc()}
    bench.write_json(args.out, row)


if __name__ == "__main__":
    main()
