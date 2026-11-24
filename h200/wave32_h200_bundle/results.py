"""Select Wave 32 arms and package compact benchmark results."""

import json
import math
import sys
import tarfile
from io import BytesIO
from pathlib import Path

BASE = "w30_base"
COMBO = "w32_best_combo"
ARMS = {
    BASE: ("off", "off"),
    "w32a_fused_tc": ("fused_tc", "off"),
    "w32a_saved_fused_tc": ("saved_fused_tc", "off"),
    "w32a_decoupled_tc": ("decoupled_tc", "off"),
    "w32b_wide_fwd_only": ("off", "wide_fwd_only"),
    "w32b_dweight_vec2_split": ("off", "dweight_vec2_split"),
    "w32b_dweight_vec2_4row": ("off", "dweight_vec2_4row"),
    "w32b_full": ("off", "full_w32b"),
}


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


def select(data: Path) -> None:
    base = read(data, BASE)
    winners = []
    names = []
    for index in range(2):
        candidates = []
        for name, presets in ARMS.items():
            row = read(data, name)
            if (
                presets[index] != "off" and valid(base) and valid(row)
                and row["peak_allocated_gib"] <= 130.0
                and row["median_ms"] <= base["median_ms"] - 0.15
            ):
                candidates.append((row["median_ms"], name, presets[index]))
        _, name, preset = min(candidates, default=(math.inf, BASE, "off"))
        names.append(name)
        winners.append(preset)
    forward = COMBO if all(win != "off" for win in winners) else next(
        (name for name in names if name != BASE), BASE,
    )
    (data / "selection.json").write_text(json.dumps({
        "winners": dict(zip("ab", winners, strict=True)),
        "individual_arms": dict(zip("ab", names, strict=True)),
        "forward_arm": forward,
    }))
    print(*winners)  # noqa: T201


def phase_two(a: str, b: str) -> list[tuple[str, str, str]]:
    if (a, b) == ("off", "off"):
        return []
    runs = [(COMBO, a, b)] if a != "off" and b != "off" else []
    return [*runs, (f"{COMBO}_rev", a, b), (f"{BASE}_rev", "off", "off")]


def transport(data: Path, out: Path) -> None:
    archive = out / "task117.tar.xz"
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
                    for field in ("kernels", "top_kernels"):
                        row.get("profiler", {}).pop(field, None)
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


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["arms"]:
            for arm, presets in ARMS.items():
                print(arm, *presets)  # noqa: T201
        case ["phase-two", a, b]:
            for run in phase_two(a, b):
                print(*run)  # noqa: T201
        case ["select", data]:
            select(Path(data))
        case ["transport", data, out]:
            transport(Path(data), Path(out))
        case _:
            sys.exit("Expected arms, phase-two, select or transport")
