"""Select eligible arms, validate gates, and pack the H200 job transcript."""
# ruff: noqa: ANN001, ANN201, T201

import io
import json
import math
import sys
import tarfile
from pathlib import Path

TRACKS = {
    "a": ("bwd_batch2", "bwd_chunk", "wave_plan", "hscan_unroll"),
    "b": ("dweight_2cta", "scalar_hiocc", "full_hopper"),
    "c": ("stage0_opt", "stage0_v2", "stage01"),
}
BASE = "w26_base"
COMBO = "w27_best_combo"
ARMS = {
    BASE: ("off", "off", "off"),
    **{
        f"w27{track}_{preset}": tuple(preset if key == track else "off" for key in TRACKS)
        for track, presets in TRACKS.items()
        for preset in presets
    },
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
    for track, presets in TRACKS.items():
        rows = {preset: read(data, f"w27{track}_{preset}") for preset in presets}
        contenders[track] = sorted(
            (preset for preset, row in rows.items() if eligible(base, row)),
            key=lambda preset: rows[preset]["median_ms"],
        )
        winners[track] = next(iter(contenders[track]), "off")
    write(data, "selection", {"winners": winners, "eligible_by_track": contenders})
    print(*winners.values())


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
            highest.get("passed") is True
            and fold.get("strict_reload_max_abs_logit_diff") == 0.0
            and fold.get("folded_train_guard") is True
        ) if fold else None,
    }
    for track, preset in read(data, "selection")["winners"].items():
        if preset == "off":
            continue
        record = read(data, f"pytest_{track}")
        gates[f"pytest_{track}"] = (
            record.get("exit_code") == 0 and record.get("counts", {}).get("passed", 0) > 0
        ) if record else None
    gates["screen_complete"] = bool(gates["screen_complete"])
    write(data, "gates", gates)


def compact_json(path, precision):
    row = json.loads(path.read_text())
    if precision is not None:
        for kernel in row.get("profiler", {}).get("kernels", []):
            # The profiler exposes CUDA activity timestamps at nanosecond resolution.
            kernel["self_cuda_ms"] = round(kernel["self_cuda_ms"], precision)
    return (json.dumps(row, separators=(",", ":")) + "\n").encode()


def transport(data, out):
    archive = out / "task97.tar.xz"
    paths = sorted(data.iterdir())
    for tail_bytes, precision in ((768, None), (400, 6), (128, 6), (128, 5), (128, 4)):
        with tarfile.open(archive, "w:xz", preset=9 | (1 << 31)) as tar:
            for path in paths:
                if path.suffix == ".log":
                    content = path.read_bytes()[-tail_bytes:]
                    if not content:
                        continue
                elif path.suffix == ".json":
                    content = compact_json(path, precision)
                else:
                    content = path.read_bytes()
                info = tarfile.TarInfo(path.name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            metadata = (
                f"kernel_ms_decimal_places={precision}\nlog_tail_bytes={tail_bytes}\n"
            ).encode()
            info = tarfile.TarInfo("transport.txt")
            info.size = len(metadata)
            tar.addfile(info, io.BytesIO(metadata))
        if 4 * ((archive.stat().st_size + 2) // 3) < 45_000:
            return
    sys.exit("Full-kernel archive exceeds comment cap; retained on H200 job volume")


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
    elif command == "transport":
        transport(data, Path(args[1]))
    else:
        sys.exit(f"Unknown command {command}")


if __name__ == "__main__":
    main()
