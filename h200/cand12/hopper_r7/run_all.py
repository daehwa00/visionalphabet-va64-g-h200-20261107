"""Run fixed-fixture full updates, live-operand replay, and winner smoke."""
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
DEADLINE = int(os.environ.get("HOPPER_R7_DEADLINE", "0"))
R6_FLAGS = ("LNET_R6_HBACK", "LNET_R6_CABWD")


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def worker():
    import torch
    import triton
    from lnet.kernel_launch_config import workload_geometries
    from plan import workload

    arm, micro = sys.argv[2], int(sys.argv[3])
    assert torch.__version__ == "2.9.1+cu128" and triton.__version__ == "3.5.1"
    if arm == "screen":
        from screen import install
        install()
    winners = read(ROOT / "winners.json") if arm == "hopper" else None
    records = workload(micro, "r4" if arm == "screen" else arm, winners)
    worker_path = EFFECTIVE / "matched_worker.py"
    sys.argv = [str(worker_path), *sys.argv[4:]]
    with workload_geometries(records):
        runpy.run_path(str(worker_path), run_name="__main__")


def trace_summary(path):
    sys.path.append(str(EFFECTIVE))
    from effective_trace_rank import rank

    ranked = rank(path)
    events = read(path)["traceEvents"]
    cpu = {e["args"]["External id"]: e["args"] for e in events
           if e.get("cat") == "cpu_op" and "kernel_file" in e.get("args", {})
           and "External id" in e["args"]}
    helpers = {}
    selected = ("_path_forward_kernel", "_path_backward_kernel", "hback56",
                "coarse_associative_backward", "_product_scan_coarse4_associative_backward_kernel")
    for event in events:
        name = event.get("name")
        if event.get("cat") != "kernel" or name not in selected:
            continue
        src = cpu.get(event["args"].get("External id"), {}).get("kernel_file")
        if not src or src in helpers or not Path(src).is_file():
            continue
        source = Path(src).read_text()
        helpers[src] = dict(symbol=name, sha256=hashlib.sha256(source.encode()).hexdigest(),
                            grid=event["args"].get("grid"), block=event["args"].get("block"),
                            markers=[m for m in ("_structural_fast_mlp", "_structural_half_hidden",
                                                 "_structural_fast_forward_2out", "_unpack_once_shape",
                                                 "_restore_parity", "_segment") if m in source])
    ranked["source_evidence"] = list(helpers.values())
    ranked["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    ranked["path"] = str(path)
    return ranked


def launch(label, arm, micro, timeout, extra=()):
    output = ROOT / "results" / (label + ".json")
    source = ROOT / ("r4" if arm in ("r4", "screen") else "current")
    cache = ROOT / "cache"
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(map(str, (source / "scripts", source / "src", HERE, EFFECTIVE))),
        LNET_LAUNCH_CACHE=str(cache / "launch"), TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
    )
    if arm not in ("r4", "screen"):
        env["LNET_R7_R5_ASSOC"] = "1"
    if arm in ("r6", "hopper"):
        env.update({name: "1" for name in R6_FLAGS})
    if arm == "hopper":
        env["LNET_R7_WINNERS"] = str(ROOT / "winners.json")
    fixture = Path(os.environ["COLD_INPUTS"])
    cmd = [sys.executable, str(HERE / "run_all.py"), "--worker", arm, str(micro),
           "--checkpoint", str(fixture / "checkpoint.pt"), "--fixture", str(fixture / "fixture.json"),
           "--contract", os.environ["CONTRACT"], "--output", str(output), "--microbatch", str(micro)]
    cmd.extend(["--warmup", "1", "--groups", "1", "--samples", "1"] if arm == "screen"
               else ["--capture", "--profile"])
    cmd.extend(extra)
    if timeout <= 0:
        return dict(status="budget_skipped", microbatch=micro)
    screen_end = int(time.time()) + max(0, timeout - 12)
    if arm == "screen":
        env["HOPPER_SCREEN_DEADLINE"] = str(screen_end)
    logpath = output.with_suffix(".log")
    error = None
    with logpath.open("w") as log:
        try:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout)
            if proc.returncode:
                error = f"worker exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            error = f"worker exceeded {timeout} s"
    row = read(output) if output.is_file() else {"microbatch": micro}
    # Export only numbers and bounded exception diagnostics.
    row.pop("traceback", None)
    if error:
        row.update(status="error", error=error,
                   detail="\n".join(logpath.read_text(errors="replace").splitlines()[-5:])[-650:])
    trace = output.with_suffix(".trace.json")
    if row.get("status") == "passed" and trace.is_file():
        row["trace"] = trace_summary(trace)
    print("FULL_ARM", label, row.get("status"), row.get("median_ms"), error or "", flush=True)
    return row


def gate(control, candidate):
    folder = ROOT / "results"
    output = folder / (candidate + "-gate.json")
    cmd = [sys.executable, str(EFFECTIVE / "matched_compare.py"), str(folder / (control + ".json")),
           str(folder / (candidate + ".json")), "--output", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if not output.is_file():
        return dict(passed=False, error=result.stderr[-500:])
    doc = read(output)
    expected = {"outputs": 5, "grads": 193, "updates": 193}
    if {k: len(v) for k, v in doc["groups"].items()} != expected:
        return dict(passed=False, error="Unexpected output, gradient or AdamW update count")
    groups = {}
    for group, tensors in doc["groups"].items():
        groups[group] = dict(count=len(tensors), failed=[n for n, v in tensors.items() if not v["passed"]],
                             max_abs=max(v["max_abs"] for v in tensors.values()),
                             max_rel_l2=max(v["relative_l2"] for v in tensors.values()))
    return dict(passed=doc["passed"], groups=groups, scalars=doc["scalar_checks"])


def eligible_winners(screen):
    chosen = {}
    for tag, case in screen.get("cases", {}).items():
        for row in case.get("trials", []):
            if not row.get("numeric_pass") or row.get("speedup", 0) <= 1.015:
                continue
            cfg = row["config"]
            plain = cfg["scheduler"] == "grid" and cfg["stages"] == 1 and cfg["lines"] == 1
            family = tag.split("-")[0]
            supported = (
                (family == "H1" and (cfg["variant"] == "r4" or
                                     (cfg["variant"] == "pack8" and tag.endswith("-0")))
                 and cfg["bm"] in (4, 8, 16, 32))
                or (family == "H2" and cfg["variant"] in ("r5", "r6") and cfg["group"] in (1, 2, 4, 8))
                or (family == "H3" and cfg["variant"] == "vfwd2out")
                or (family == "H5" and cfg["variant"] == "seg4" and cfg["bm"] in (16, 32))
            )
            if plain and supported and row["speedup"] > chosen.get(tag, {}).get("speedup", 0):
                chosen[tag] = dict(cfg, speedup=row["speedup"])
    return chosen


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    report = {"arms": {}, "gates": {}, "status": "running"}
    dest = ROOT / "run.json"
    save(dest, report)

    def arm(label, variant, micro, limit, reserve=0):
        seconds = min(limit, DEADLINE - int(time.time()) - reserve)
        report["arms"][label] = launch(label, variant, micro, seconds)
        save(dest, report)

    def compare(control, candidate):
        if all(report["arms"][label].get("status") == "passed" for label in (control, candidate)):
            report["gates"][candidate] = gate(control, candidate)
            save(dest, report)

    arm("r4_h200_control", "r4", 1024, 210, 750)
    arm("r5_combined_1024", "r5", 1024, 155, 750)
    compare("r4_h200_control", "r5_combined_1024")
    arm("r5_r6_1024", "r6", 1024, 155, 680)
    compare("r4_h200_control", "r5_r6_1024")
    for micro in (512, 256):
        arm(f"r4_{micro}", "r4", micro, 125, 620)
        arm(f"r5_r6_{micro}", "r6", micro, 125, 590)
        compare(f"r4_{micro}", f"r5_r6_{micro}")

    arm("live_screen", "screen", 1024, 840, 155)
    screen_path = ROOT / "screen.json"
    winners = eligible_winners(read(screen_path)) if screen_path.is_file() else {}
    report["integrable_winners"] = winners
    save(ROOT / "winners.json", winners)
    save(dest, report)
    if winners:
        arm("hopper_r7_candidate", "hopper", 1024, 145)
        compare("r4_h200_control", "hopper_r7_candidate")
    required = ("r5_combined_1024", "r5_r6_1024", "r5_r6_512", "r5_r6_256")
    report["status"] = ("complete" if all(k in report["gates"] for k in required)
                        and report["arms"]["live_screen"]["status"] == "passed" else "partial")
    save(dest, report)


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "--worker":
        worker()
    else:
        main()
