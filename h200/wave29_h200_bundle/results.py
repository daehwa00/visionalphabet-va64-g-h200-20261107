"""Select eligible arms, validate gates, and pack the H200 job transcript."""
# ruff: noqa: ANN001, ANN201, T201

import io
import json
import math
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

BASE = "w28_base"
COMBO = "w29_best_combo"
ARMS = {
    BASE: ("off", "off", "off"),
    "w29a_compose_w28": ("compose_w28", "off", "off"),
    "w29a_compose_ckpt": ("compose_ckpt", "off", "off"),
    "w29a_vscan_fwd_opt": ("vscan_fwd_opt", "off", "off"),
    "w29a_vscan_bwd_deep": ("vscan_bwd_deep", "off", "off"),
    "w29b_hiocc_dweight": ("off", "hiocc_dweight", "off"),
    "w29b_fwd_stream_vec2": ("off", "fwd_stream_vec2", "off"),
    "w29b_hiocc_dinput_v2": ("off", "hiocc_dinput_v2", "off"),
    "w29b_full": ("off", "full_w29b", "off"),
    "w29c_swiglu_vec2d": ("off", "off", "swiglu_vec2d"),
    "w29c_rmsnorm_bwd": ("off", "off", "rmsnorm_bwd"),
    "w29c_rmsnorm_full": ("off", "off", "rmsnorm_full"),
    "w29c_combined_bwd": ("off", "off", "combined_bwd"),
    "w29c_combined_full": ("off", "off", "combined_full"),
}


def read(data, name):
    path = data / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write(data, name, value):
    (data / f"{name}.json").write_text(json.dumps(value, separators=(",", ":")) + "\n")


def valid(row):
    return bool(
        row.get("status") == "ok"
        and row.get("finite_losses") is True
        and row.get("finite_gradients") is True
        and math.isfinite(row.get("median_ms", math.nan))
        and row["median_ms"] > 0
        and math.isfinite(row.get("peak_allocated_gib", math.nan))
    )


def eligible(base, row):
    return (
        valid(base)
        and valid(row)
        and row["peak_allocated_gib"] <= base["peak_allocated_gib"] + 0.05
        and row["median_ms"] <= base["median_ms"] - 0.10
    )


def select(data):
    base = read(data, BASE)
    winners = {}
    contenders = {}
    for index, track in enumerate("abc"):
        rows = {
            flags[index]: read(data, arm)
            for arm, flags in ARMS.items() if flags[index] != "off"
        }
        contenders[track] = sorted(
            (preset for preset, row in rows.items() if eligible(base, row)),
            key=lambda preset: rows[preset]["median_ms"],
        )
        winners[track] = next(iter(contenders[track]), "off")
    c_memory_target = {
        preset: (row["peak_allocated_gib"] - base["peak_allocated_gib"]) * 1024 <= 20
        for arm, (*_, preset) in ARMS.items() if preset != "off"
        and valid(base) and valid(row := read(data, arm))
    }
    write(data, "selection", {
        "winners": winners, "eligible_by_track": contenders,
        "c_memory_target_le_20_mib": c_memory_target,
    })
    print(*winners.values())


def reader_parity(data, path):
    records = []
    peak = defaultdict(float)
    failures = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break  # A timeout may interrupt the final write.
        records.append({key: row[key] for key in ("shape", "dtype", "strict")})
        for preset, arm in row["arms"].items():
            for reference, comparisons in arm.items():
                for operation, metrics in comparisons.items():
                    label = f"{preset}/{row['dtype']}/{reference}/{operation}"
                    peak[label] = max(peak[label], metrics["relative_l2"])
                    if not metrics["passed"]:
                        failures.append({**records[-1], "check": label, **metrics})
    rc = int((data / "processes.tsv").read_text().splitlines()[-1].split("\t")[1])
    write(data, "parity_b", {
        "exit_code": rc, "passed": rc == 0 and bool(records) and not failures,
        "cases": records, "max_relative_l2": peak, "failures": failures,
    })


def whole_h200(row):
    return (
        row.get("status") == "ok"
        and "H200" in row.get("gpu", "")
        and row.get("sm_count") == 132
        and row.get("vram_gib", 0) > 130
        and row.get("capability") == [9, 0]
    )


def summarize(data):
    names = (*ARMS, COMBO, f"{COMBO}_rev", f"{BASE}_rev")
    rows = {name: read(data, name) for name in names}
    columns = (
        "status", "median_ms", "images_per_second", "peak_allocated_gib",
        "peak_reserved_gib", "finite_losses", "finite_gradients",
    )
    lines = ["\t".join(("arm", *columns))]
    for name, row in rows.items():
        lines.append("\t".join((name, *(str(row.get(key, "")) for key in columns))))
    (data / "arms.tsv").write_text("\n".join(lines) + "\n")
    dynamo, fold = read(data, "dynamo"), read(data, "fold")
    graphs = dynamo.get("graphs", {})
    highest = fold.get("fp32_head_parity_by_precision", {}).get("highest", {})
    gates = {
        "whole_h200": whole_h200(read(data, "device")),
        "screen_complete": all(valid(rows[name]) and rows[name].get("profiler") for name in ARMS),
        "forward_ab": eligible(rows[BASE], rows[COMBO]) if rows[COMBO] else None,
        "reverse_ab": (
            eligible(rows[f"{BASE}_rev"], rows[f"{COMBO}_rev"])
            if rows[f"{COMBO}_rev"] or rows[f"{BASE}_rev"] else None
        ),
        "dynamo": (
            len(graphs) == 6
            and all(set(modes) == {"train", "eval", "folded"} for modes in graphs.values())
            and all(pair == [1, 0] for modes in graphs.values() for pair in modes.values())
        ) if dynamo else None,
        "infer_fold": (
            highest.get("max_abs_logit_diff", math.inf) < 1e-6
            and fold.get("strict_reload_max_abs_logit_diff") == 0.0
            and fold.get("folded_train_guard") is True
        ) if fold else None,
    }
    gates["reader_parity"] = read(data, "parity_b").get("passed")
    for track, preset in read(data, "selection")["winners"].items():
        if preset != "off" or track == "c":
            gates[f"pytest_{track}"] = pytest_ok(
                read(data, "pytest_parity_c" if track == "c" else f"pytest_{track}")
            )
        if track == "a" and preset != "off":
            gates["scan_parity"] = read(data, "parity_a").get("passed")
    gates["screen_complete"] = bool(gates["screen_complete"])
    write(data, "gates", gates)


def pytest_ok(record):
    if not record:
        return None
    counts = record.get("counts") or {}
    return record.get("exit_code") == 0 and counts.get("passed", 0) > 0 and not any(
        counts.get(key, 0) for key in ("skipped", "failed", "errors", "xfailed", "xpassed")
    )


def compact_json(path, precision, detail):
    row = json.loads(path.read_text())
    profile = row.get("profiler", {})
    if precision is not None:
        for kernel in profile.get("kernels", []):
            kernel["self_cuda_ms"] = round(kernel["self_cuda_ms"], precision)
    if detail != "full":
        profile["kernels"] = [
            kernel for kernel in profile.get("kernels", [])
            if kernel["name"].startswith(("reader_", "_nr4_correlation"))
            or any(
                part in kernel["name"]
                for part in ("silu_split", "swiglu", "reduce_hscan", "reduce_vscan", "rmsnorm")
            )
        ] if profile else []
        if detail == "summary":
            for scan in profile.get("scan_shape_cuda_events", []):
                scan.pop("plans", None)
            for field in ("reader_dispatch", "scan_dispatch", "cuda_annotations"):
                profile.pop(field, None)
            if "tail" in row:
                row["tail"] = row["tail"][-8:]
    return (json.dumps(row, separators=(",", ":")) + "\n").encode()


def transport(data, out):
    archive = out / "task105.tar.xz"
    paths = sorted(data.iterdir())
    for tail_bytes, precision, detail in (
        (256, None, "full"), (80, 5, "full"), (100, 6, "major"), (64, 5, "summary"),
    ):
        with tarfile.open(archive, "w:xz", preset=9 | (1 << 31)) as tar:
            for path in paths:
                if path.suffix == ".log":
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - tail_bytes))
                        content = stream.read()
                    if not content:
                        continue
                elif path.suffix == ".json":
                    content = compact_json(path, precision, detail)
                else:
                    content = path.read_bytes()
                info = tarfile.TarInfo(path.name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            metadata = (
                f"kernel_ms_decimal_places={precision}\nlog_tail_bytes={tail_bytes}\n"
                f"kernel_detail={detail}\n"
            ).encode()
            info = tarfile.TarInfo("transport.txt")
            info.size = len(metadata)
            tar.addfile(info, io.BytesIO(metadata))
        if 4 * ((archive.stat().st_size + 2) // 3) < 45_000:
            return
    sys.exit("Archive exceeds comment cap; full results retained on the H200 volume")


def main():
    command, *args = sys.argv[1:]
    if command == "arms":
        for arm, flags in ARMS.items():
            print(arm, *flags)
        return
    data = Path(args[0])
    if command == "device":
        if not whole_h200(read(data, "device")):
            sys.exit("Expected a whole H200 SM 9.0")
    elif command == "select":
        select(data)
    elif command == "summarize":
        summarize(data)
    elif command == "reader-parity":
        reader_parity(data, Path(args[1]))
    elif command == "transport":
        transport(data, Path(args[1]))
    else:
        sys.exit(f"Unknown command {command}")


if __name__ == "__main__":
    main()
