"""Join Kineto CUDA kernels to ATen operators and compute logical rooflines."""

# ruff: noqa: PLC0415
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

GEMM = re.compile(r"nvjet_|gemm|cublas|cutlass|xmma", re.IGNORECASE)
CONV = re.compile(
    r"cudnn|convol(?:ve|ution)|conv[123]d|(?:^|_)(?:fprop|dgrad|wgrad)(?:_|\b)", re.IGNORECASE
)
MATRIX_OPS = {"aten::mm": 0, "aten::addmm": 1, "aten::bmm": 0, "aten::baddbmm": 1}
PEAKS = {"rtx3080ti": (136.4, 912), "h200": (989, 4800)}


def roofline(flops: int, logical_bytes: int, elapsed_us: float | None = None) -> dict:
    result = {
        "flops": flops,
        "logical_bytes": logical_bytes,
        "flops_per_byte": flops / logical_bytes,
    }
    for hardware, (tflops, gbps) in PEAKS.items():
        compute = flops / (tflops * 1e6)
        memory = logical_bytes / (gbps * 1e3)
        result[hardware] = {
            "compute_us": compute,
            "memory_us": memory,
            "floor_us": max(compute, memory),
            "roofline_limit": "compute" if compute > memory else "memory",
        }
        if elapsed_us:
            result[hardware].update(
                bf16_peak_percent=100 * compute / elapsed_us,
                logical_dram_peak_percent=100 * memory / elapsed_us,
            )
    if elapsed_us:
        result.update(
            tflops=flops / elapsed_us / 1e6, logical_gbps=logical_bytes / elapsed_us / 1e3
        )
    return result


def matrix_metric(name: str | None, args: dict) -> dict:
    shapes = args.get("Input Dims", [])
    index = MATRIX_OPS.get(name)
    if index is None or len(shapes) < index + 2:
        return {}
    left, right = shapes[index : index + 2]
    if len(left) not in (2, 3) or len(left) != len(right) or left[-1] != right[-2]:
        return {}
    m, k = left[-2:]
    n = right[-1]
    batch = left[0] if len(left) == 3 else 1
    types = args.get("Input type", args.get("Input Types", []))[index : index + 2]
    strides = args.get("Input Strides", [])[index : index + 2]
    return {"m": m, "n": n, "k": k, "batch": batch, "types": types, "strides": strides}


def analyze(path: Path, fx: tuple[Path, Path] | None = None) -> dict:  # noqa: C901, PLR0912, PLR0915
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        events = json.load(stream)["traceEvents"]
    matched = {}
    if fx:
        from inductor import compiled_calls

        matched = compiled_calls(events, fx)
    by_external = {}
    matrix_exp_scopes = collections.defaultdict(list)
    launches = {}
    scopes = collections.defaultdict(list)
    for event in events:
        args = event.get("args", {})
        category = event.get("cat")
        external = args.get("External id")
        if category == "cpu_op" and external is not None:
            by_external[external] = event
            if event["name"] == "aten::linalg_matrix_exp":
                matrix_exp_scopes[event["tid"]].append(event)
        elif category in {"cuda_runtime", "cuda_driver"} and "correlation" in args:
            launches[args["correlation"]] = event
        elif category == "user_annotation" and event["name"].startswith("wave22a."):
            scopes[event["tid"]].append(event)
    rows = []
    for event in events:
        if event.get("cat") != "kernel":
            continue
        name = event["name"]
        args = event["args"]
        launch = launches.get(args.get("correlation"), {})
        external = args.get("External id", launch.get("args", {}).get("External id"))
        aten = by_external.get(external, {})
        op = aten.get("name")
        if CONV.search(name) or "nhwcAddPaddingKernel" in name or "convolution" in (op or ""):
            bucket = "conv"
        elif GEMM.search(name):
            bucket = "gemm"
        elif "multi_tensor_apply_kernel" in name:
            bucket = "multi_tensor"
        elif "_static_separable_variance_kernel" in name:
            bucket = "variance"
        elif op == "aten::fill_":
            bucket = "fill"
        elif "elementwise" in name:
            bucket = "elementwise"
        else:
            continue
        phase = next(
            (
                scope["name"]
                for scope in scopes[launch.get("tid")]
                if scope["ts"] <= launch["ts"] <= scope["ts"] + scope["dur"]
            ),
            None,
        )
        row = {
            "bucket": bucket,
            "kernel": name,
            "op": op,
            "scope": phase,
            "elapsed_us": event["dur"],
            "input_dims": aten.get("args", {}).get("Input Dims"),
            "launch_us": launch.get("dur"),
            "external": external,
        }
        if bucket == "gemm":
            row.update(matrix_metric(op, aten.get("args", {})))
            if op == "aten::bmm" and any(
                scope["ts"] <= aten["ts"] <= scope["ts"] + scope["dur"]
                for scope in matrix_exp_scopes[aten["tid"]]
            ):
                row["parent_op"] = "aten::linalg_matrix_exp"
            call = matched.get(external)
            if call:
                left, right = call["dims"]
                shape = {"m": left[-2], "n": right[-1], "k": left[-1]}
                if all(row.get(axis, length) == length for axis, length in shape.items()):
                    row.setdefault("types", call["types"])
                    row.update(**shape, batch=left[0] if len(left) == 3 else 1, fx=call)
        rows.append(row)
    grouped = collections.defaultdict(list)
    for row in rows:
        if row["bucket"] == "gemm":
            grouped[row["external"]].append(row)
    operators = []
    for kernels in grouped.values():
        row = kernels[0]
        if "m" not in row:
            continue
        operator = {key: row[key] for key in ("external", "op", "m", "n", "k", "batch", "types")}
        operator["fx"] = row.get("fx")
        operator["parent_op"] = row.get("parent_op")
        operator["strides"] = row.get("strides") or (row.get("fx") or {}).get("strides")
        operator["kernel_count"] = len(kernels)
        operator["elapsed_us"] = sum(kernel["elapsed_us"] for kernel in kernels)
        types = row["types"]
        if len(types) == 2 and types[0] == types[1]:
            size = {
                "bfloat16": 2,
                "half": 2,
                "float": 4,
                "c10::BFloat16": 2,
                "c10::Half": 2,
                "bf16": 2,
                "f32": 4,
            }.get(types[0])
            if size is not None:
                m, n, k, batch = (row[key] for key in ("m", "n", "k", "batch"))
                operator["roofline"] = roofline(
                    2 * batch * m * n * k,
                    size * batch * (m * k + k * n + m * n),
                    operator["elapsed_us"],
                )
                batch_axes = {128 * side**2 for side in (7, 14, 28, 56)}
                source = (row.get("fx") or {}).get("source") or ""
                if "self.standardizer(descriptor)" in source:
                    batch_axes.add(128)
                m, n, k = [d * 8 if d in batch_axes else d for d in (m, n, k)]
                operator["b1024_model"] = {
                    "m": m,
                    "n": n,
                    "k": k,
                    **roofline(2 * batch * m * n * k, size * batch * (m * k + k * n + m * n)),
                }
        operators.append(operator)
    summary = {}
    for row in rows:
        group = summary.setdefault(row["bucket"], {})
        key = row["scope"] or row["op"] or "unattributed"
        entry = group.setdefault(key, {"count": 0, "ms": 0.0})
        entry["count"] += 1
        entry["ms"] += row["elapsed_us"] / 1000
    return {"summary": summary, "kernels": rows, "matrix_operators": operators}


def stems() -> list[dict]:
    rows = []
    for batch in (128, 1024):
        for inputs, outputs, size in ((3, 32, 224), (32, 64, 112)):
            output_size = size // 2
            valid_taps = (3 * output_size - 1) ** 2
            logical_bytes = 2 * (
                batch * (inputs * size**2 + outputs * output_size**2) + 9 * inputs * outputs
            )
            rows.append(
                {
                    "batch": batch,
                    "c_in": inputs,
                    "c_out": outputs,
                    "h_in": size,
                    "per_pass_nominal": roofline(
                        2 * batch * output_size**2 * 9 * inputs * outputs, logical_bytes
                    ),
                    "per_pass_valid_taps": roofline(
                        2 * batch * valid_taps * inputs * outputs, logical_bytes
                    ),
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--fx", type=Path, nargs=2, metavar=("FORWARD", "BACKWARD"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"stems": stems()}
    if args.trace:
        result.update(analyze(args.trace, tuple(args.fx) if args.fx else None))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
