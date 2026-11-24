"""Select Wave 34 arms and package compact benchmark results."""

import json
import math
import sys
import tarfile
from io import BytesIO
from pathlib import Path

BASE = "w33_base"
COMBO = "w34_best_combo"
ARMS = {
    "w33_base": ("off", "off", "off"),
    "w34a_modefirst_tc": ("modefirst_tc", "off", "off"),
    "w34a_modefirst_tc_gs1": ("modefirst_tc_gs1", "off", "off"),
    "w34b_dweight_wpr": ("off", "dweight_wpr", "off"),
    "w34b_dweight_wpr_waves": ("off", "dweight_wpr_waves", "off"),
    "w34b_dinput_wpr": ("off", "dinput_wpr", "off"),
    "w34b_full": ("off", "full_w34b", "off"),
    "w34c_rmsnorm_fwd_v2": ("off", "off", "rmsnorm_fwd_v2"),
    "w34c_fused_res_norm": ("off", "off", "fused_res_norm"),
    "w34c_full": ("off", "off", "full_w34c"),
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
    for index in range(3):
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
    forward = COMBO if sum(win != "off" for win in winners) >= 2 else next(
        (name for name in names if name != BASE), BASE,
    )
    (data / "selection.json").write_text(json.dumps({
        "winners": dict(zip("abc", winners, strict=True)),
        "individual_arms": dict(zip("abc", names, strict=True)),
        "forward_arm": forward,
    }))
    print(*winners)  # noqa: T201


def phase_two(a: str, b: str, c: str) -> list[tuple[str, str, str, str]]:
    count = sum(preset != "off" for preset in (a, b, c))
    if count == 0:
        return []
    runs = [(COMBO, a, b, c)] if count >= 2 else []
    return [*runs, (f"{COMBO}_rev", a, b, c), (f"{BASE}_rev", *ARMS[BASE])]


def compact_json(data: Path, name: str, *, keep_logs: bool, keep_shapes: bool) -> bytes:
    row = read(data, name) or {"status": "unreadable"}
    profiler = row.get("profiler", {})
    for field in ("kernels", "top_kernels"):
        profiler.pop(field, None)
    if not keep_shapes:
        profiler.pop("scan_shape_cuda_events", None)
    if not keep_logs:
        if "error" in row:
            row["error"] = row["error"][-512:]
        row.pop("tail", None)
    return json.dumps(row, separators=(",", ":")).encode()


def transport(data: Path, out: Path) -> None:
    archive = out / "task124.tar.xz"
    paths = sorted(path for path in data.iterdir() if path.suffix != ".tmp")
    for log_bytes, keep_shapes in ((1024, True), (128, True), (0, True), (0, False)):
        with tarfile.open(archive, "w:xz", preset=6) as tar:
            for path in paths:
                if path.suffix == ".log":
                    if not log_bytes:
                        continue
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - log_bytes))
                        content = stream.read()
                elif path.suffix == ".json":
                    content = compact_json(
                        data, path.stem, keep_logs=bool(log_bytes), keep_shapes=keep_shapes,
                    )
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
        case ["phase-two", a, b, c]:
            for run in phase_two(a, b, c):
                print(*run)  # noqa: T201
        case ["select", data]:
            select(Path(data))
        case ["transport", data, out]:
            transport(Path(data), Path(out))
        case _:
            sys.exit("Expected arms, phase-two, select or transport")
