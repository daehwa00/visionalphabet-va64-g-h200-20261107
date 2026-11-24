"""Validate device, dispatch evidence, selection, and regression results."""
# ruff: noqa: ANN001, ANN201, T201

import json
import sys
from pathlib import Path

CANDIDATES = ("wave25bc", "wave25bc_hscan8", "wave25bc_hscan8_ps14")
ARMS = ("wave21c_base", *CANDIDATES)


def read(out, name):
    path = out / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def checks(arm, row):
    profile = row.get("profiler", {})
    scans = profile.get("scan_dispatch", [])
    reader = profile.get("reader_dispatch", [])
    tests = {
        "profiler": bool(profile.get("general_cuda_scans_verified") and profile.get("kernels"))
    }
    tests["no_register_56"] = not any(
        scan["register_path"] and scan["height"] == 56 for scan in scans
    )
    tests["no_mma"] = not any(scan["mma_path"] for scan in scans)
    tests["hscan_compact_unroll8"] = any(
        scan["horizontal_unroll8"] and scan["compact"] and scan["height"] == 56
        for scan in scans
    ) == (arm in CANDIDATES[1:])
    tests["vscan_register_28"] = any(
        scan["register_batch2"] and scan["height"] == 28 for scan in scans
    ) == (arm in CANDIDATES[1:])
    if arm in CANDIDATES:
        spatial = [r for r in reader if r.get("selector") == "_wave25b_spatial"]
        tests["reader_56_scalar_dinput"] = any(
            r["operation"] == "dinput" and r["shape"][1:3] == [56, 56] and r["plan"]["vec"] == 1
            for r in spatial
        )
        tests["reader_14x8_weight"] = any(
            r["operation"] == "dweight"
            and r.get("selector") == "_wave25b_weight"
            and r["plan"][1:] == [14, 8]
            for r in reader
        )
        tests.update(
            {
                f"reader_{size}_multirow_forward": any(
                    r["operation"] == "forward"
                    and r["shape"][1:3] == [size, size]
                    and r["plan"]["rows"] > 1
                    for r in spatial
                )
                for size in (56, 28, 7)
            }
        )
    if arm in CANDIDATES:
        tests["conv0_tc_forward_backward"] = any(
            "stem_conv_tensor_stages" in k["name"] and k["count"] >= 2
            for k in profile.get("kernels", [])
        )
    tests["fp32_ps14"] = any(
        mod["degree"] == 14
        and mod["scaling_steps"] == 2
        and mod["base_dtype"] == "torch.float32"
        and mod["compute_dtype"] in {"None", "torch.float32"}
        for mod in row.get("orthogonal", [])
    ) == (arm == "wave25bc_hscan8_ps14")
    return tests


def finite(row):
    return row.get("status") == "ok" and row.get("finite_losses") and row.get("finite_gradients")


def select(out):
    rows = {arm: read(out, f"b1024_{arm}") for arm in ARMS}
    base = rows["wave21c_base"]
    details = {arm: checks(arm, row) for arm, row in rows.items()}
    candidates = [
        arm
        for arm in CANDIDATES
        if finite(base)
        and finite(rows[arm])
        and all(details[arm].values())
        and rows[arm]["peak_allocated_gib"] <= base["peak_allocated_gib"]
    ]
    winner = min(candidates, key=lambda arm: rows[arm]["median_ms"], default="")
    (out / "selection.json").write_text(
        json.dumps({"winner": winner, "dispatch": details}, indent=2)
    )
    print(winner)


def regressions(out):
    graphs = read(out, "dynamo").get("graphs", {})
    fold = read(out, "fold")
    pytest = read(out, "pytest")
    gates = {
        "probe": whole_h200(read(out, "best-probe")),
        "dynamo": len(graphs) == 6
        and all(
            set(modes) == {"train", "eval", "folded"} and all(v == [1, 0] for v in modes.values())
            for modes in graphs.values()
        ),
        "fold": (
            fold.get("fp32_head_parity_by_precision", {}).get("highest", {}).get("passed") is True
            and fold.get("strict_reload_max_abs_logit_diff") == 0.0
            and fold.get("folded_train_guard") is True
        ),
        "cuda_pytest": pytest.get("exit_code") == 0
        and (pytest.get("counts") or {}).get("passed", 0) > 0,
    }
    selection = read(out, "selection")
    best = selection["winner"]
    gates["reverse"] = all(
        finite(row) and all(checks(arm, row).values())
        for arm, row in (
            (best, read(out, "b1024_best_rev")),
            ("wave21c_base", read(out, "b1024_wave21c_base_rev")),
        )
    )
    gates["sweep"] = all(
        (finite(row) and bool(profile.get("top_kernels") or profile.get("kernels")))
        or (batch == 2048 and row.get("status") == "oom")
        for batch in (128, 512, 1024, 2048)
        for arm in ("wave21c_base", best)
        for row in (read(out, f"b{batch}_{arm}_sweep"),)
        for profile in (row.get("profiler", {}),)
    )
    gates["no_timeouts"] = all(
        line.split("\t")[1] not in {"124", "137"}
        for line in (out / "processes.tsv").read_text().splitlines()
    )
    (out / "gates.json").write_text(json.dumps(gates, indent=2))
    if not all(gates.values()):
        sys.exit("One or more promotion gates failed")



def whole_h200(row):
    return (
        row.get("status") == "ok"
        and "H200" in row["gpu"]
        and row["sm_count"] == 132
        and row["vram_gib"] > 130
        and row["capability"] == [9, 0]
    )


if __name__ == "__main__":
    command, directory = sys.argv[1:]
    out = Path(directory)
    if command == "device":
        if not whole_h200(read(out, "device")):
            sys.exit("Expected a whole H200 SM 9.0")
    elif command == "select":
        select(out)
    elif command == "regressions":
        regressions(out)
    else:
        sys.exit(f"Unknown command {command}")
