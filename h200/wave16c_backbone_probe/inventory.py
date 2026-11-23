"""Join every GPU kernel in a Chrome trace to the host launch and operator."""

import argparse
import json
import runpy
from collections import defaultdict
from pathlib import Path

SUITE = runpy.run_path("h200/liverms_v2_opt/bench_h200.py")
SOURCES = (
    "memory_backbone.py(", "baseline_backbone.py(", "va-t62-baseline.py(",
    "fused_stem_ops.py(", "image_layers.py(", "terminal_imaginary_correlation.py(",
    "fused_backbone_ops.py(", "va_model_builder.py(", "/cache/",
)


def build_inventory(trace: dict) -> list[dict]:
    events = trace["traceEvents"]
    host = {}
    launches = {}
    frames = defaultdict(list)
    for event in events:
        args = event.get("args", {})
        category = event.get("cat")
        external = args.get("External id")
        if external is not None and category in {"cpu_op", "user_annotation"}:
            host.setdefault(external, []).append({
                "name": event["name"],
                "shape": args.get("Input Dims"),
                "stack": args.get("Call stack"),
            })
        if category == "python_function" and any(source in event["name"] for source in SOURCES):
            frames[event["pid"], event["tid"]].append(event)
        if category in {"cuda_runtime", "cuda_driver"} and "correlation" in args:
            launches[args["correlation"]] = event
    rows = []
    for event in events:
        category = event.get("cat")
        if category not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        args = event.get("args", {})
        launch = launches.get(args.get("correlation"), {})
        external = args.get("External id", launch.get("args", {}).get("External id"))
        launch_time = launch.get("ts", 0)
        source_frames = sorted(
            (
                frame for frame in frames[launch.get("pid"), launch.get("tid")]
                if frame["ts"] <= launch_time <= frame["ts"] + frame["dur"]
            ),
            key=lambda frame: frame["ts"],
        )
        rows.append({
            "name": event["name"],
            "category": category,
            "bucket": SUITE["kernel_bucket"](event["name"]),
            "duration_us": event["dur"],
            "timestamp_us": event["ts"],
            "grid": args.get("grid"),
            "block": args.get("block"),
            "registers_per_thread": args.get("registers per thread"),
            "local_memory_bytes": args.get("local memory per thread"),
            "external_id": external,
            "launch_api": launch.get("name"),
            "operators": host.get(external, []),
            "python_frames": [frame["name"] for frame in source_frames],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = build_inventory(json.loads(args.trace.read_text()))
    args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
