"""Execute a single Wave 39 GPU benchmark arm."""
# ruff: noqa: BLE001

import argparse
import json
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import torch

from h200.liverms_v2_opt import bench_h200 as bench
from h200.wave39_h200_bundle.metrics import BUCKETS, SUBGROUPS, classify

VARIANT = "17-p1-nr4-1024d-v3"


def profile_step(
    step: Callable[[], torch.Tensor], eager_scan_diagnostic: Callable[[], None] | None,
) -> dict:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as profiler:
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

    shapes = []
    horizontal_events = []
    shape_error = None
    if eager_scan_diagnostic is not None:
        try:
            with bench.track_scan_events() as diagnostics:
                eager_scan_diagnostic()
                torch.cuda.synchronize()
            _, shapes = bench.summarize_scan_events(diagnostics)
            horizontal_events = [
                {"name": row["name"], "cuda_event_ms": round(row["cuda_event_ms"], 5),
                 "calls": row["count"]}
                for row in shapes if row["name"].startswith("hscan_")
            ]
        except Exception:
            shape_error = traceback.format_exc()
    total = sum(buckets.values())
    graph_launches = sum(event.count for event in profiler.events()
                         if "graphlaunch" in event.name.lower())
    coverage = total > 0 and bool(
        buckets["factorized_reader"] and (buckets["general_vscan"] or groups["scan"]["calls"])
    )
    return {
        "buckets": buckets if coverage else {},
        "subkernels": groups if coverage else {},
        "total_self_cuda_ms": total if coverage else None,
        "profile_cuda_coverage": coverage,
        "scan_shape_cuda_events": compact_shapes(shapes),
        "legacy_hscan_cuda_events": horizontal_events,
        "scan_shape_events_source": "eager_same_scan_policy" if eager_scan_diagnostic else None,
        "scan_shape_events_error": shape_error,
        "profile_graph_launches": graph_launches,
        "top_kernels": [{"name": name, "calls": counts[name], "self_cuda_ms": elapsed}
                        for name, elapsed in kernels.most_common(20)],
    }


def compact_shapes(shapes: list[dict]) -> list[dict]:
    return [
        {"name": row["name"], "cuda_event_ms": round(row["cuda_event_ms"], 5),
         "count": row["count"], "plan": row["plans"]}
        for row in shapes if row["name"].startswith(("vscan_fwd_", "vscan_bwd_"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scan-events", action="store_true")
    args = parser.parse_args()
    args.variant, args.batch, args.profile = VARIANT, 1024, True
    try:
        def profile(step: Callable, scan: Callable) -> dict:
            return profile_step(step, scan if args.scan_events else None)

        row = bench.train_arm(args, profiler=profile)
        profile = row.pop("profiler", {})
        row.update(status="ok", **profile)
        replays = {"forward": 0, "backward": 0, "update": 0}
        replays.update(row.get("cudagraph_replays_by_phase", {}))
        row.update(
            img_per_sec=row["images_per_second"],
            loss_finite=row["finite_losses"],
            grad_finite=row["finite_gradients"],
            finite=(row["finite_losses"] and row["finite_gradients"]
                    and row["grad_tensor_count"] == 193),
            loss_value=row["losses"]["last"],
            cudagraph_replays=replays,
        )
        total = row.get("total_self_cuda_ms")
        row["wall_minus_cuda_ms"] = None if total is None else row["median_ms"] - total
    except Exception as exc:
        partial = json.loads(args.out.read_text()) if args.out.exists() else {}
        row = {**partial, "status": "oom" if "out of memory" in str(exc).lower() else "error",
               "error": traceback.format_exc()}
    bench.write_json(args.out, row)


if __name__ == "__main__":
    main()
