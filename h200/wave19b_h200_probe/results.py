"""Summarize Task 72 and enforce completeness and correctness on the runner."""
import json
import sys
from pathlib import Path

ARMS = ("baseline_w14a", "wave17b", "wave19b")
FULL = (*ARMS, "wave19b_rev", "wave17b_rev")
SHAPES = ("56x56x160x4", "28x28x320x4", "14x14x640x4", "7x7x640x4")


def read(out, name):
    return json.loads((out / f"{name}.json").read_text())


def device(data):
    return (
        data["status"] == "ok"
        and "H200" in data["gpu"]
        and data["sm_count"] == 132
        and data["vram_gib"] > 130
    )


def scalar_weight_launch(row):
    # Forced vec1 bypasses the vector planner. The scalar launch grid encodes
    # mode tiles times rank tiles. At M=640, rank 4, a grid of 40 is rank tile 2.
    operation = row["arms"]["1"]["dweight"]
    name, grid, threads = operation["launches"][0]
    assert name == "dweight" and grid[0] == 40 and threads == 128, operation["launches"]
    return {"vec": 1, "rank_tile": 2, "warps": 4, "grid": grid}


def check(out):
    failures = []
    for batch in (1024, 128, 256, 512):
        for arm in FULL if batch == 1024 else ARMS:
            name = f"b{batch}_{arm}"
            try:
                row = read(out, name)
                assert row["status"] == "ok", row.get("error")
                assert row["finite_losses"] and row["finite_gradients"], "nonfinite training"
                if batch == 1024:
                    assert row["profiler"]["general_cuda_scans_verified"], "missing scan profile"
                    assert row["profiler"]["top_kernels"], "empty profile"
            except (AssertionError, KeyError, OSError) as exc:
                failures.append(f"{name}: {exc}")
    for shape in SHAPES:
        try:
            rows = [json.loads(line) for line in (out / f"reader_micro_{shape}.jsonl").read_text().splitlines()]
            assert {r["dtype"] for r in rows} == {"torch.bfloat16", "torch.float16"}
            for row in rows:
                assert row["shape"] == [1024, *map(int, shape.split("x"))], row["shape"]
                assert set(row["arms"]) == {"baseline", "auto", "1", "2"}
                if shape.startswith("14x"):
                    assert row["plans"]["auto"]["dweight"]
                    scalar_weight_launch(row)
        except (AssertionError, KeyError, OSError) as exc:
            failures.append(f"reader {shape}: {exc}")
    dynamo = read(out, "dynamo")["graphs"]
    if len(dynamo) != 6 or any(
        set(modes) != {"train", "eval", "folded"} or any(v != [1, 0] for v in modes.values())
        for modes in dynamo.values()
    ):
        failures.append(f"Dynamo gate: {dynamo}")
    fold = read(out, "fold")
    if not (
        fold["fp32_head_parity_by_precision"]["highest"]["passed"]
        and fold["strict_reload_passed"] and fold["folded_train_guard"]
    ):
        failures.append("Fold correctness gate")
    if (out / "pytest_rc.txt").read_text().strip() != "0":
        failures.append("CUDA pytest gate")
    return failures


def summary(out):
    for batch in (1024, 128, 256, 512):
        for arm in FULL if batch == 1024 else ARMS:
            try:
                row = read(out, f"b{batch}_{arm}")
            except FileNotFoundError:
                continue
            print(f"{batch:4} {arm:15} " + " ".join(
                f"{row[key]:.3f}" for key in (
                    "median_ms", "images_per_second", "peak_allocated_gib", "peak_reserved_gib"
                ) if key in row
            ) + f" status={row['status']}")
    micro = out / "reader_micro_14x14x640x4.jsonl"
    if micro.exists():
        for line in micro.read_text().splitlines():
            row = json.loads(line)
            for arm in ("auto", "1"):
                timing = row["arms"][arm]["dweight"]
                plan = (scalar_weight_launch(row) if arm == "1"
                        else row.get("plans", {}).get(arm, {}).get("dweight"))
                print(f"14x14 {row['dtype']} {arm} dweight {timing['median_ms']:.6f} ms "
                      f"{timing['effective_gb_s']:.3f} GB/s plan={plan}")


if __name__ == "__main__":
    command, directory = sys.argv[1:]
    out = Path(directory)
    if command == "device":
        data = read(out, "device")
        print(json.dumps(data))
        raise SystemExit(0 if device(data) else "Expected a whole H200")
    if command == "gates":
        failures = check(out)
        for failure in failures:
            print(failure)
        raise SystemExit(bool(failures))
    summary(out)
