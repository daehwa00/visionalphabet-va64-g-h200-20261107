"""Select Wave 35 arms and package compact benchmark results."""

import json
import math
import sys
import tarfile
from io import BytesIO
from pathlib import Path

BASE = "w33_base"
COMBO = "w35_best_combo"
OFF = ("off", "off", "off")
ARMS = {
    BASE: OFF,
    "w35a_coop_bwd": ("vscan_coop_bwd", "off", "off"),
    "w35a_decoupled_simt": ("vscan_decoupled_simt", "off", "off"),
    "w35a_hscan_vec2": ("hscan_vec2", "off", "off"),
    "w35b_dweight_rt4": ("off", "dweight_rt4", "off"),
    "w35b_dweight_rows4": ("off", "dweight_rows4", "off"),
    "w35b_fwd_dinput_vec4": ("off", "fwd_dinput_vec4", "off"),
    "w35c_stem_v2": ("off", "off", "stem_v2"),
    "w35c_swiglu_v3": ("off", "off", "swiglu_v3"),
    "w35c_rmsnorm_bwd_inductor": ("off", "off", "rmsnorm_bwd_inductor"),
    "w35c_inductor_tune": ("off", "off", "inductor_tune"),
}
EXCLUSIVE = [
    ("w35a_coop_bwd", "w35a_decoupled_simt"),
    ("w35b_dweight_rt4", "w35b_dweight_rows4"),
]


def read(data: Path, name: str) -> dict:
    path = data / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def valid(row: dict) -> bool:
    median = row.get("median_ms")
    allocated = row.get("peak_allocated_gib")
    return (
        row.get("status") == "ok"
        and row.get("finite_losses") is True
        and row.get("finite_gradients") is True
        and type(median) in (int, float) and math.isfinite(median) and median > 0
        and type(allocated) in (int, float) and math.isfinite(allocated) and allocated >= 0
    )


def select(rows: dict) -> dict:
    base = rows[BASE]
    wins = {
        name for name, row in rows.items()
        if name != BASE and valid(base) and valid(row) and row["peak_allocated_gib"] <= 130
        and (row["median_ms"] < base["median_ms"] - 0.15 if name.startswith("w35b_")
             else row["median_ms"] <= base["median_ms"] - 0.15)
    }
    for pair in EXCLUSIVE:
        candidates = wins.intersection(pair)
        if candidates:
            winner = min(candidates, key=lambda name: (rows[name]["median_ms"], name))
            wins.difference_update(candidates - {winner})

    def subgroup(row: dict, name: str) -> float | None:
        return row.get("profiler", {}).get("major_kernel_ms", {}).get(name)

    before = subgroup(base, "reader_forward")
    after = subgroup(rows["w35b_fwd_dinput_vec4"], "reader_forward")
    delta = after - before if before and after else None
    arms = [name for name in ARMS if name in wins]
    presets = [
        ",".join(ARMS[name][i] for name in arms if ARMS[name][i] != "off") or "off"
        for i in range(3)
    ]
    return {
        "arms": arms, "presets": presets,
        "reader_forward_delta_ms": delta,
        "reader_forward_won": delta is not None and delta < 0,
    }


def transport(data: Path, out: Path) -> None:
    archive = out / "task128.tar.xz"
    paths = sorted(path for path in data.iterdir() if path.suffix != ".tmp")
    for log_bytes in (1024, 128, 0):
        with tarfile.open(archive, "w:xz", preset=6) as tar:
            for path in paths:
                if path.suffix == ".log":
                    if not log_bytes:
                        continue
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - log_bytes))
                        content = stream.read()
                elif path.suffix == ".json":
                    row = read(data, path.stem) or {"status": "unreadable"}
                    row.get("profiler", {}).pop("kernels", None)
                    if not log_bytes:
                        row["error"] = row.get("error", "")[-512:]
                        row.pop("tail", None)
                    content = json.dumps(row, separators=(",", ":")).encode()
                else:
                    content = path.read_bytes()
                info = tarfile.TarInfo(path.name)
                info.size = len(content)
                tar.addfile(info, BytesIO(content))
        if 4 * ((archive.stat().st_size + 2) // 3) < 45_000:
            return
    sys.exit("Archive exceeded cap; full results retained on the H200 volume")


def verify(data: Path) -> bool:
    selection = read(data, "selection")
    graphs = read(data, "dynamo").get("graphs", {})
    fold = read(data, "fold")
    graph_counts = [counts for modes in graphs.values() for counts in modes.values()]
    gates = {
        "individual_arms": all(
            valid(row) and bool(row.get("profiler"))
            for row in (read(data, name) for name in ARMS)
        ),
        "dynamo": len(graph_counts) == 18 and all(counts == [1, 0] for counts in graph_counts),
        "fold": fold.get("fp32_head_max_abs_logit_diff", math.inf) < 1e-6
        and fold.get("strict_reload_passed") is True and fold.get("folded_train_guard") is True,
        **{
            f"parity_{track}": read(data, f"parity_{track}").get("passed") is True
            for track in "ab"
        },
        **{
            f"pytest_{track}": read(data, f"pytest_{track}").get("status") == "ok"
            for track in "abc"
        },
    }
    if selection.get("arms"):
        combo = read(data, f"{COMBO}_no_decoupled") or read(data, COMBO)
        reverse = read(data, f"{COMBO}_rev")
        gates["combo"] = all(
            valid(row) and row["peak_allocated_gib"] <= 130 for row in (combo, reverse)
        ) and valid(read(data, f"{BASE}_rev"))
    (data / "verification.json").write_text(json.dumps(gates))
    return all(gates.values())


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["arms"]:
            for arm, presets in ARMS.items():
                print(arm, *presets)  # noqa: T201
        case ["select", data]:
            path = Path(data)
            selection = select({name: read(path, name) for name in ARMS})
            (path / "selection.json").write_text(json.dumps(selection))
            print(*selection["presets"])  # noqa: T201
        case ["transport", data, out]:
            transport(Path(data), Path(out))
        case ["verify", data]:
            sys.exit(0 if verify(Path(data)) else 1)
        case _:
            sys.exit("Expected arms, select, transport or verify")
