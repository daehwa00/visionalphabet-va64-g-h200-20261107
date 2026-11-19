"""Run fixed-checkpoint updates and live-operand SM90 screening."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EFFECTIVE = HERE.parent / "effective_cli"
ROOT = Path(os.environ.get("RUN_ROOT", "/tmp"))
R6_FLAGS = ("LNET_R6_HBACK", "LNET_R6_CABWD")
R7_FLAGS = ("LNET_R7_AFWD", "LNET_R7_DIRECT")
ARMS = ("r4_control", "r5_combined", "r5_r6", "r5_r6_r7")
EXPECTED_CASES = {
    "H1-56-0", "H1-56-1", "H2-14-0", "H2-14-1", "H2-28-0", "H2-28-1",
    "H3-56-0", "H4-14-1", "H4-28-1", "H4-56-1",
    "H5-56-0-backward", "H5-56-1-backward", "H5-56-0-forward",
}


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def gpu_topology():
    import torch

    props = torch.cuda.get_device_properties(0)
    gib = props.total_memory / 2**30
    mig = gib < 80
    return dict(device_name=props.name, capability=[props.major, props.minor],
                multi_processor_count=props.multi_processor_count, total_memory_gib=gib,
                is_mig=mig, pool_mode="mig_1g18gb" if mig else "whole_h200")


def worker():
    import torch
    import triton
    from lnet.kernel_launch_config import workload_geometries
    from plan import workload

    arm, micro = sys.argv[2], int(sys.argv[3])
    assert torch.__version__ == "2.9.1+cu128" and triton.__version__ == "3.5.1"
    if arm == "screen":
        from screen import install

        install(micro, pilot=micro == 512)
    records = workload(micro, "r4" if arm in ("screen", ARMS[0]) else arm)
    worker_path = EFFECTIVE / "matched_worker.py"
    sys.argv = [str(worker_path), *sys.argv[4:]]
    with workload_geometries(records):
        runpy.run_path(str(worker_path), run_name="__main__")


def trace_summary(path):
    from collections import defaultdict

    totals = defaultdict(float)
    bins = defaultdict(lambda: [0, 0.0])
    for event in read(path)["traceEvents"]:
        if event.get("cat") != "kernel":
            continue
        name = event["name"]
        if name in ("_path_backward_kernel", "r7_direct_backward", "coarse_associative_backward") or "associative_backward" in name:
            family = "vback"
        elif name == "hback56" or "horizontal_path_backward" in name or "bidirectional_backward" in name:
            family = "hback"
        elif name == "_path_forward_kernel" or "associative_forward" in name:
            family = "vfwd"
        elif "bidirectional_forward" in name:
            family = "hfwd"
        else:
            family = "other"
        key = (family, name, tuple(event["args"].get("grid", [])), tuple(event["args"].get("block", [])))
        ms = event["dur"] / 1000
        bins[key][0] += 1
        bins[key][1] += ms
        totals[family] += ms
    rows = [
        dict(family=key[0], name=key[1], grid=key[2], block=key[3], calls=value[0], ms_update=value[1])
        for key, value in sorted(bins.items(), key=lambda item: -item[1][1])
    ]
    known = ("r7_associative_forward", "r7_direct_backward", "coarse_associative_backward", "hback56")
    dispatch = {name: sum(row["calls"] for row in rows if row["name"] == name) for name in known}
    return dict(path=str(path), dispatch_counts=dispatch, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                total_kernel_ms=sum(totals.values()), kernel_event_count=sum(r["calls"] for r in rows),
                families_ms=dict(totals), top=rows[:35])


def launch(label, arm, micro, seconds, *, profile=False):
    output = ROOT / "results" / f"{label}.json"
    if seconds <= 0:
        return dict(status="budget_skipped", microbatch=micro)
    source = ROOT / ("r4" if arm in (ARMS[0], "screen") else "current")
    cache = ROOT / "cache"
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(map(str, (source / "scripts", source / "src", HERE))),
        LNET_LAUNCH_CACHE=str(cache / "launch"), TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), PYTHONUNBUFFERED="1",
    )
    if arm in ARMS[2:]:
        env.update(dict.fromkeys(R6_FLAGS, "1"))
    if arm == ARMS[3]:
        env.update(dict.fromkeys(R7_FLAGS, "1"))
    fixture = Path(os.environ["COLD_INPUTS"])
    cmd = [sys.executable, str(HERE / "run_all.py"), "--worker", arm, str(micro),
           "--checkpoint", str(fixture / "checkpoint.pt"), "--fixture", str(fixture / "fixture.json"),
           "--contract", os.environ["CONTRACT"], "--output", str(output), "--microbatch", str(micro)]
    if arm == "screen":
        env["HOPPER_SCREEN_DEADLINE"] = str(int(time.time()) + max(0, seconds - 12))
        cmd.extend(("--warmup", "1", "--groups", "1", "--samples", "1"))
    else:
        cmd.append("--capture")
        if profile:
            cmd.append("--profile")
    logpath = output.with_suffix(".log")
    error = None
    with logpath.open("w") as log:
        try:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=seconds)
            if proc.returncode:
                error = f"worker exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            error = f"worker exceeded {seconds} s"
    try:
        row = read(output) if output.is_file() else {"microbatch": micro}
    except json.JSONDecodeError:
        row = {"microbatch": micro, "error": "worker result was interrupted during write"}
    row.pop("traceback", None)
    if error:
        row.update(status=row.get("status") if row.get("status") in ("oom", "error", "measured") else "error",
                   worker_error=error,
                   detail="\n".join(logpath.read_text(errors="replace").splitlines()[-5:])[-900:])
    if arm == "screen":
        # Interceptor replay synchronizes and launches extra kernels inside the
        # training step, so the screen driving run is never a benchmark arm.
        return {key: row[key] for key in ("status", "microbatch", "error", "worker_error", "detail") if key in row}
    row["flags"] = {name: env.get(name, "0") for name in (*R6_FLAGS, *R7_FLAGS)}
    trace = output.with_suffix(".trace.json")
    if row.get("status") in ("passed", "measured") and trace.is_file():
        try:
            row["trace"] = trace_summary(trace)
        except Exception as exc:
            row["trace_error"] = str(exc)[-400:]
    print("FULL_ARM", label, row.get("status"), row.get("median_ms"), error or "", flush=True)
    return row


def gate(control, candidate):
    folder = ROOT / "results"
    output = folder / f"{candidate}-vs-{control}.json"
    cmd = [sys.executable, str(EFFECTIVE / "matched_compare.py"), str(folder / f"{control}.json"),
           str(folder / f"{candidate}.json"), "--output", str(output)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    except subprocess.TimeoutExpired:
        return dict(passed=False, error="numerical comparison timeout")
    if not output.is_file():
        return dict(passed=False, error=proc.stderr[-600:])
    doc = read(output)
    if {k: len(v) for k, v in doc["groups"].items()} != {"outputs": 5, "grads": 193, "updates": 193}:
        return dict(passed=False, error="unexpected output, gradient or AdamW update count")
    groups = {}
    for group, tensors in doc["groups"].items():
        failed = [n for n, v in tensors.items() if not v["passed"]
                  or (group == "outputs" and v["max_abs"] > 1e-4)]
        groups[group] = dict(count=len(tensors), failed_count=len(failed), failed=failed[:12],
                             max_abs=max(v["max_abs"] for v in tensors.values()),
                             max_rel_l2=max(v["relative_l2"] for v in tensors.values()))
    passed = doc["passed"] and not any(item["failed_count"] for item in groups.values())
    return dict(passed=passed, groups=groups, scalars=doc["scalar_checks"],
                output_abs_limit=1e-4, adamw_abs_limit=3e-5, adamw_rel_l2_limit=.02)


def main():
    start = int(os.environ["HOPPER_R8_START"])
    topology = gpu_topology()
    mig = topology["is_mig"]
    deadline = int(os.environ["HOPPER_R8_DEADLINE"]) - 10
    dest = ROOT / "result.json"
    report = dict(gpu_topology=topology, topology_basis="visible CUDA memory and requested pool",
                  arms={}, gates={}, screen_runs={}, status="running",
                  start_epoch_s=start, deadline_epoch_s=deadline + 10)
    save(dest, report)
    if topology["capability"] != [9, 0] or topology["pool_mode"] != os.environ["HOPPER_REQUESTED_POOL"]:
        report.update(status="invalid_topology", requested_pool=os.environ["HOPPER_REQUESTED_POOL"])
        save(dest, report)
        raise RuntimeError("SM90 capability and pool must match the request")
    print("GPU_TOPOLOGY", json.dumps(topology), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    primary, extra = (128, 256) if mig else (1024, 512)

    def label(arm, micro):
        return f"{arm}_{micro}x{1024 // micro}"

    def measure(arm, micro, limit, reserve, profile=False):
        key = label(arm, micro)
        budget = min(limit, deadline - int(time.time()) - reserve)
        report["arms"][key] = launch(key, arm, micro, budget, profile=profile)
        save(dest, report)

    def compare(control, candidate, micro):
        ref, cand = label(control, micro), label(candidate, micro)
        if deadline - time.time() > 36 and all(
                report["arms"][key].get("status") in ("passed", "measured") for key in (ref, cand)):
            report["gates"][f"{cand}_vs_{ref}"] = gate(ref, cand)
            save(dest, report)

    def screen(micro, limit, reserve):
        budget = min(limit, deadline - int(time.time()) - reserve)
        report["screen_runs"][str(micro)] = launch(f"screen_{micro}", "screen", micro, budget)
        path = ROOT / f"screen-{micro}.json"
        if path.is_file():
            rows = read(path)["cases"]
            wanted = {"H3-56-0"} if mig and micro == 512 else EXPECTED_CASES
            report["screen_runs"][str(micro)]["missing_cases"] = sorted(wanted - rows.keys())
            report["screen_runs"][str(micro)]["incomplete_cases"] = sorted(
                k for k, v in rows.items() if v.get("status") != "complete"
            )
        save(dest, report)

    for index, arm in enumerate(ARMS):
        measure(arm, primary, (285 if not mig else 175) if not index else (220 if not mig else 150),
                (3 - index) * 115 + (450 if mig else 600), profile=arm in (ARMS[0], ARMS[-1]))
    for arm in ARMS[1:]:
        compare(ARMS[0], arm, primary)
    for arm in ARMS[2:]:
        compare(ARMS[1], arm, primary)
    compare(ARMS[2], ARMS[3], primary)

    screen(primary, 480 if mig else 1250, 225 if mig else 0)
    if mig:
        screen(extra, 370, 0)
        if deadline - time.time() > 215:
            for index, arm in enumerate(ARMS):
                measure(arm, extra, 95, (3 - index) * 35 + 25)
            for arm in ARMS[1:]:
                compare(ARMS[0], arm, extra)
        if deadline - time.time() > 75:
            screen(512, 135, 0)
    elif deadline - time.time() > 95:
        measure(ARMS[-1], extra, 180, 5)
    else:
        report["microbatch_check"] = "budget_skipped"
        save(dest, report)

    primary_arms = [report["arms"][label(arm, primary)] for arm in ARMS]
    required = all(row["status"] == "passed" for row in primary_arms)
    required &= all("trace" in row for row in (primary_arms[0], primary_arms[-1]))
    dispatch = primary_arms[-1].get("trace", {}).get("dispatch_counts", {})
    report["missing_optimized_dispatches"] = [name for name in (
        "r7_associative_forward", "r7_direct_backward", "coarse_associative_backward", "hback56"
    ) if not dispatch.get(name)]
    required &= not report["missing_optimized_dispatches"]
    required &= len(report["gates"]) >= 6
    for batch in (primary, extra) if mig else (primary,):
        entry = report["screen_runs"][str(batch)]
        required &= not (entry.get("missing_cases") or entry.get("incomplete_cases") or entry["status"] != "passed")
    report["status"] = "complete" if required else "partial"
    report["all_full_gates_passed"] = bool(report["gates"]) and all(row["passed"] for row in report["gates"].values())
    save(dest, report)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker()
    else:
        main()
