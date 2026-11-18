"""Turn measured component records into proposals, never final speed decisions."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast


def _shape(value: object) -> list[int] | None:
    if isinstance(value, list) and len(value) == 4 and all(type(x) is int and x > 0 for x in value):
        return cast("list[int]", value)
    return None


def _post_shape(node: dict[str, object]) -> list[int] | None:
    if node["kind"] == "same_resolution":
        return _shape(node.get("output_shape"))
    output = node.get("output_shape")
    return _shape(output[0]) if isinstance(output, list) and output else None


def _positive(value: object) -> float | None:
    if type(value) in (int, float) and math.isfinite(cast("float", value)) and cast("float", value) > 0:
        return float(cast("float", value))
    return None


def measured_component_hints(
    catalog: list[dict[str, object]], directory: Path, *, device_name: str,
) -> dict[str, object]:
    post: dict[str, dict[str, object]] = {}
    projection: dict[str, dict[str, object]] = {}
    scan: dict[str, dict[str, object]] = {}
    post_gain: dict[str, float] = {}
    projection_gain: dict[str, float] = {}
    scan_gain: dict[str, float] = {}
    sources = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("device", payload.get("gpu")) != device_name:
            continue
        if not ("records" in payload or "candidates" in payload):
            continue
        sources.append(str(path))
        raw_records = payload.get("records", [])
        records = [row for row in raw_records if isinstance(row, dict)] if isinstance(raw_records, list) else []
        for node in catalog:
            site = cast("str", node["site"])
            shape = _shape(node.get("input_shape"))
            if shape is None:
                continue
            batch, height, width, _ = shape
            post_shape = _post_shape(node)
            post_site = node.get("post_site")
            if post_site is not None and post_shape is not None and "spatial_multiplicity" in payload:
                if payload.get("modes") != post_shape[-1] // 2 or payload.get("hidden_modes") != node["hidden"]:
                    continue
                matches = [row for row in records if row.get("batch") == batch
                           and row.get("spatial") == post_shape[1]]
                baseline = next((_positive(row.get("latency_ms")) for row in matches
                                 if row.get("variant") == "baseline"), None)
                if baseline is not None:
                    for row in matches:
                        elapsed = _positive(row.get("latency_ms"))
                        error = row.get("relative_l2_error")
                        if (elapsed is None or type(error) not in (int, float)
                                or not math.isfinite(cast("float", error)) or not 0 <= cast("float", error) <= 0.01):
                            continue
                        variant = row.get("variant")
                        if variant not in ("baseline", "joint_gated", "gate_output", "full"):
                            continue
                        gain = baseline - elapsed
                        key = cast("str", post_site)
                        if gain >= post_gain.get(key, 0.0):
                            post[key] = {"variant": "none" if variant == "baseline" else variant}
                            post_gain[key] = gain
            component_shape = payload.get("shape")
            if node["memory_projection"] and post_shape is not None and isinstance(component_shape, dict):
                matches_shape = all(component_shape.get(key) == value for key, value in {
                    "batch": batch, "height": post_shape[1], "width": post_shape[2],
                    "inputs": node["poles"], "outputs": post_shape[-1] // 2,
                }.items())
                baseline = _positive(payload.get("baseline_ms"))
                if matches_shape and baseline is not None:
                    for row in payload.get("candidates", []):
                        elapsed = _positive(row.get("ms"))
                        if row.get("passed") is not True or elapsed is None:
                            continue
                        gain = baseline - elapsed
                        if gain > projection_gain.get(site, 0.0):
                            projection[site] = {"projection": row["fusion"],
                                                "projection_block_rows": row["block_rows"],
                                                "projection_output_tile": row["output_tile"],
                                                "projection_num_warps": row["num_warps"]}
                            projection_gain[site] = gain
            for row in records:
                if row.get("valid") is not True or row.get("batch") != batch:
                    continue
                if row.get("gain") != node.get("gain_normalization") or row.get("swiglu") != node.get("path_swiglu"):
                    continue
                baseline, elapsed = _positive(row.get("baseline_ms")), _positive(row.get("candidate_ms"))
                if baseline is None or elapsed is None:
                    continue
                gain = baseline - elapsed
                if gain <= scan_gain.get(site, 0.0):
                    continue
                if "epilogue" in row:
                    epilogue = ("mean_energy" if node.get("readout_mode") == "mean_energy" else "descriptor") if node["kind"] == "terminal" else "path"
                    if (node["kind"] == "transition" or row.get("epilogue") != epilogue
                            or row.get("size") != height or height != width
                            or row.get("modes") != node["poles"]
                            or row.get("planar") != bool(node["memory_projection"])):
                        continue
                    scan[site] = {"scan": "scan2d", "scan_block_modes": row["block_modes"],
                                  "scan_num_warps": row["warps"]}
                elif "output_tile" in row:
                    if (node["kind"] != "same_resolution" or not node["memory_projection"]
                            or row.get("height") != height or row.get("width") != width
                            or row.get("poles") != node["poles"]
                            or row.get("outputs") != shape[-1] // 2):
                        continue
                    scan[site] = {"scan": "vertical_projection", "vertical_pole_tile": row["pole_tile"],
                                  "vertical_output_tile": row["output_tile"],
                                  "vertical_num_warps": row["num_warps"]}
                else:
                    continue
                scan_gain[site] = gain
    return {"post": post, "projection": projection, "scan": scan,
            "provenance": sources,
            "scope": "component-guided proposals; whole-model measurements determine selection"}
