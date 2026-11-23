"""Flatten raw Wave 15A samples into graph, model, and ptxas tables."""

import argparse
import csv
import json
from pathlib import Path

PHASES = ("combined", "hscan_fwd", "vscan_fwd", "vscan_bwd", "hscan_bwd")


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        table = csv.DictWriter(handle, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def graph_rows(rows: list[dict], source: str) -> list[dict]:
    result = []
    for case in rows:
        shape = case.get("shape")
        if shape is None or "results" not in case:
            continue
        for name, arm in case["results"].items():
            samples = arm.get("samples_ms", {})
            for phase in PHASES:
                times = samples.get(phase, [])
                if not times:
                    continue
                plan = arm["plan"].get(phase, {})
                result.append({
                    "source": source, "shape": "x".join(map(str, shape[:4])),
                    "coarse": shape[4], "dtype": case["dtype"], "arm": name, "phase": phase,
                    "median_ms": arm["median_ms"][phase],
                    "samples_ms": " ".join(map(str, times)),
                    "warps": plan.get("warps"), "workers": plan.get("workers"),
                    "smem_bytes": plan.get("smem"),
                    "scratch_bytes": plan.get("estimated_scratch_bytes"),
                    "peak_allocated_bytes": arm["peak_allocated_bytes"],
                })
    return result


def model_row(model: dict, source: str) -> dict:
    event = model.get("profiler", {}).get("scan_axis_cuda_event_ms", {})
    return {
        "source": source, "batch": model.get("batch"), "status": model.get("status"),
        "median_ms": model.get("median_ms"), "images_per_second": model.get("images_per_second"),
        "peak_allocated_gib": model.get("peak_allocated_gib"),
        "finite_losses": model.get("finite_losses"),
        "finite_gradients": model.get("finite_gradients"),
        "hscan_ms": event.get("general_hscan"), "vscan_ms": event.get("general_vscan"),
    }


def profile_rows(model: dict, source: str) -> list[dict]:
    return [
        {
            "source": source, "batch": model.get("batch"),
            **{field: row[field] for field in (
                "name", "vertical", "backward", "coarse", "height", "width", "modes", "count",
                "cuda_event_ms",
            )},
            "mean_event_ms": row["cuda_event_ms"] / row["count"],
            "plans": json.dumps(row["plans"], separators=(",", ":")),
        }
        for row in model.get("profiler", {}).get("scan_shape_cuda_events", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    root = parser.parse_args().folder
    graph, model, resources, profiles = [], [], [], []
    for path in sorted(root.rglob("*.jsonl")):
        relative = str(path.relative_to(root))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if path.name.startswith("ptxas"):
            resources.extend(
                {
                    "source": relative,
                    **{key: value for key, value in row.items() if key not in (
                        "log", "spills", "memory_instructions",
                    )},
                    "spill_store_bytes": row["spills"][0], "spill_load_bytes": row["spills"][1],
                    **row["memory_instructions"],
                } for row in rows
            )
        else:
            graph.extend(graph_rows(rows, relative))
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("model"):
            row = json.loads(path.read_text())
            source = str(path.relative_to(root))
            model.append(model_row(row, source))
            profiles.extend(profile_rows(row, source))
        if path.name.startswith("h200-") and path.name.endswith("-result.json"):
            result = json.loads(path.read_text())
            for name, rows in result.items():
                if isinstance(rows, list):
                    graph.extend(graph_rows(rows, f"{path.name}/{name}"))
            for name, row in result.get("arms", {}).items():
                source = f"{path.name}/{name}"
                model.append(model_row(row, source))
                profiles.extend(profile_rows(row, source))
    write(root / "cuda-graph.csv", graph)
    write(root / "p1-nr4.csv", model)
    write(root / "ptxas.csv", resources)
    write(root / "scan-profiler.csv", profiles)


if __name__ == "__main__":
    main()
