"""Qualify the fixed 1024-image workload and screen live SM90 operands."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EFFECTIVE = HERE.parent / "effective_cli"
ROOT = Path(os.environ.get("RUN_ROOT", "/tmp"))
R6_FLAGS = ("LNET_R6_HBACK", "LNET_R6_CABWD")
R7_FLAGS = ("LNET_R7_AFWD", "LNET_R7_DIRECT")
R89_FLAGS = ("LNET_R8_HFWD", "LNET_R8_HBACK", "LNET_R9_RMS_DW")
R10_FLAGS = ("LNET_R10_HFWD56_SEG4", "LNET_R10_RMS_DW56", "LNET_R10_TUNED_LAUNCH")
PACK_FLAG = "LNET_R10_SEG4PACK2"
SCAN = R6_FLAGS + R7_FLAGS
CURRENT = SCAN + R89_FLAGS
FULL = CURRENT + R10_FLAGS
CONTROL = "r4_control"
ARMS = {
    CONTROL: ("none", ()),
    "r5_r6_r7": ("both", SCAN),
    "r5_r6_r7_r8_r9": ("both", CURRENT),
    "r10_full_safe": ("both", FULL),
    "r10_full_seg4pack2": ("both", (*FULL, PACK_FLAG)),
    "r10_full_r5_off": ("none", FULL),
    "r10_full_r5_off_seg4pack2": ("none", (*FULL, PACK_FLAG)),
    "r5_combined": ("both", ()),
    "r5_vjp_only": ("vjp", ()),
    "r5_forward_only": ("forward", ()),
}
SCREEN_CASES = {"H3-56-0", "H1-56-0", "H1-56-1", "H5-56-0-forward"}
DISPATCH = ("r7_associative_forward", "r7_direct_backward", "coarse_associative_backward", "hback56",
            "r8_horizontal_forward", "r8_horizontal_backward", "r9_channel_weight_partial",
            "r10_hfwd56_seg4", "r10_rms56_partial", "r10_segmented_coarse_forward")



def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)
    persist = os.environ.get("PERSIST_ROOT")
    if persist and path.name in ("result.json", "screen-vertical.json"):
        try:
            dst = Path(persist) / path.name
            partial = dst.with_suffix(".tmp")
            shutil.copyfile(path, partial)
            partial.replace(dst)
        except OSError:
            pass


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


def observe_rms_graph(output):
    """Record the real 56² FakeTensor strides seen by Inductor before rewriting."""
    import torch
    from torch.fx import Node
    from lnet.r10_rms import RMS56ScaleGradientPass

    invoke = RMS56ScaleGradientPass.__call__
    path = output.with_suffix(".fx-rms56.json")

    def record(pass_, graph):
        rows = read(path) if path.is_file() else []

        def inspect(node, depth):
            if not isinstance(node, Node):
                return str(node)
            value = node.meta.get("val")
            info = dict(op=str(node.target), node=node.op)
            if isinstance(value, torch.Tensor):
                info.update(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
            if depth:
                info["args"] = [inspect(arg, depth - 1) for arg in node.args]
            return info

        for node in graph.nodes:
            if node.target == torch.ops.aten.sum.dim_IntList and node.args and isinstance(node.args[0], Node):
                tensor = node.args[0].meta.get("val")
                if isinstance(tensor, torch.Tensor) and tuple(tensor.shape[1:]) == (56, 56, 64):
                    entry = inspect(node, 3)
                    if entry not in rows:
                        rows.append(entry)
        if rows:
            save(path, rows)
        return invoke(pass_, graph)

    RMS56ScaleGradientPass.__call__ = record


def diagnose_nonfinite(output):
    from functools import wraps
    import effective_batch_update

    original = effective_batch_update.run_update

    @wraps(original)
    def update(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except FloatingPointError as exc:
            traceback = exc.__traceback__
            while traceback:
                tensors = traceback.tb_frame.f_locals.get("checks")
                if tensors:
                    report = {group: [name for name, t in values.items() if not bool(t.isfinite().all())]
                              for group, values in tensors.items()}
                    save(output.with_suffix(".nonfinite.json"), report)
                    break
                traceback = traceback.tb_next
            raise

    effective_batch_update.run_update = update


def worker():
    import torch
    import triton
    from lnet.kernel_launch_config import workload_geometries
    from plan import workload

    arm, micro = sys.argv[2], int(sys.argv[3])
    assert torch.__version__ == "2.9.1+cu128" and triton.__version__ == "3.5.1"
    if arm == "screen":
        from vertical_seg_screen import install

        install(micro)
    records = workload(micro, "none" if arm == "screen" else ARMS[arm][0])
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    diagnose_nonfinite(output)
    if os.environ.get("LNET_R10_RMS_DW56") == "1":
        observe_rms_graph(output)
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
        elif name == "hback56" or "horizontal_path_backward" in name or "bidirectional_backward" in name or "r8_horizontal_backward" in name:
            family = "hback"
        elif name == "_path_forward_kernel" or "associative_forward" in name:
            family = "vfwd"
        elif "bidirectional_forward" in name or "r8_horizontal_forward" in name or "r10_hfwd56_seg4" in name:
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
    dispatch = {name: sum(row["calls"] for row in rows if row["name"] == name
                          or row["name"].startswith(name + "_")) for name in DISPATCH}
    launches = {name: [row for row in rows if row["name"] == name or row["name"].startswith(name + "_")]
                for name, count in dispatch.items() if count}
    return dict(path=str(path), dispatch_counts=dispatch, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                total_kernel_ms=sum(totals.values()), kernel_event_count=sum(r["calls"] for r in rows), launches=launches,
                families_ms=dict(totals), top=rows[:35])


def launch(label, arm, micro, seconds, *, profile=False):
    output = ROOT / "results" / f"{label}.json"
    if seconds <= 0:
        return dict(status="budget_skipped", microbatch=micro)
    branch = "r4" if arm in (CONTROL, "screen") else "r7-reference" if arm == "r5_r6_r7" else "current"
    source = ROOT / branch
    cache = ROOT / "cache"
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(map(str, (source / "scripts", source / "src", HERE))),
        LNET_LAUNCH_CACHE=str(cache / "launch"), TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), PYTHONUNBUFFERED="1",
    )
    if arm != "screen":
        env.update(dict.fromkeys(ARMS[arm][1], "1"))
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
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if proc.wait(timeout=seconds):
                    error = f"worker exit {proc.returncode}"
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
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
    row["flags"] = {name: env.get(name, "0") for name in (*FULL, PACK_FLAG)}
    row["r5"] = ARMS[arm][0]
    fx = output.with_suffix(".fx-rms56.json")
    if fx.is_file():
        row["rms56_fx"] = read(fx)
    bad = output.with_suffix(".nonfinite.json")
    if bad.is_file():
        nonfinite = read(bad)
        row["nonfinite"] = {k: dict(count=len(v), examples=v[:10]) for k, v in nonfinite.items()}
    trace = output.with_suffix(".trace.json")
    if row.get("status") in ("passed", "measured") and trace.is_file():
        try:
            row["trace"] = trace_summary(trace)
            persist = os.environ.get("PERSIST_ROOT")
            if persist:
                import gzip

                target = Path(persist) / (trace.name + ".gz")
                try:
                    with trace.open("rb") as src, gzip.open(target, "wb", compresslevel=3) as dst:
                        shutil.copyfileobj(src, dst)
                    row["trace"]["persistent_gzip"] = str(target)
                except OSError:
                    target.unlink(missing_ok=True)
        except Exception as exc:
            row["trace_error"] = str(exc)[-400:]
    print("FULL_ARM", label, row.get("status"), row.get("median_ms"), error or "", flush=True)
    return row


def gate(control, candidate):
    folder = ROOT / "results"
    output = folder / f"{candidate}-vs-{control}.json"
    source = ROOT / "current"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source / "scripts", source / "src", HERE))))
    cmd = [sys.executable, str(HERE / "numerical_gate.py"), str(folder / f"{control}.json"),
           str(folder / f"{candidate}.json"), str(output)]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=35)
    except subprocess.TimeoutExpired:
        return dict(passed=False, error="numerical comparison timeout")
    if proc.returncode or not output.is_file():
        return dict(passed=False, error=proc.stderr[-850:])
    doc = read(output)
    from_keys = ("official", "passed", "requested_thresholds", "scalars")
    report = {key: doc[key] for key in from_keys}
    report["raw_report_path"] = str(output)
    report["raw_report_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    persist = os.environ.get("PERSIST_ROOT")
    if persist:
        try:
            dst = Path(persist) / output.name
            shutil.copyfile(output, dst)
            report["raw_report_path"] = str(dst)
        except OSError:
            pass
    report["raw"] = {}
    for group, values in doc["raw"].items():
        finite = [key for key in values if values[key]["finite"]]
        worst = {max(finite, key=lambda k: values[k][metric]) for metric in ("max_abs", "rel_l2")} if finite else set()
        keys = worst | set(doc["official"].get(group, {}).get("failed", ()))
        keys.update(n for n, v in values.items() if not v["requested_pass"])
        if group == "outputs":
            keys.update(values)
        keys.update(n for n in values if n.endswith("pole_input_projection.point_weight"))
        report["raw"][group] = {n: values[n] for n in sorted(keys)}
    return report


def strict_pair(control, candidate):
    try:
        a = [sample for group in control["wall_samples_ms"] for sample in group]
        b = [sample for group in candidate["wall_samples_ms"] for sample in group]
        return dict(dominance=candidate["median_ms"] < control["median_ms"],
                    raw_strict_separation=max(b) < min(a), candidate_max_ms=max(b), control_min_ms=min(a),
                    delta_ms=candidate["median_ms"] - control["median_ms"])
    except (KeyError, ValueError):
        return dict(dominance=False, raw_strict_separation=False)


def main():
    start = int(os.environ["HOPPER_R10_START"])
    topology = gpu_topology()
    deadline = int(os.environ["HOPPER_R10_DEADLINE"]) - 15
    dest = ROOT / "result.json"
    report = dict(gpu_topology=topology, arms={}, gates={}, pairs=[], status="running",
                  start_epoch_s=start, deadline_epoch_s=deadline + 15)
    save(dest, report)
    if (topology["capability"] != [9, 0] or topology["pool_mode"] != "whole_h200"
            or topology["multi_processor_count"] != 132 or not 139 <= topology["total_memory_gib"] <= 140
            or topology["device_name"] != "NVIDIA H200 NVL"):
        raise RuntimeError("Visible device must match the Whole H200 request")
    print("GPU_TOPOLOGY", json.dumps(topology), flush=True)
    (ROOT / "results").mkdir(exist_ok=True)

    def measure(key, arm, limit, reserve, profile=False):
        budget = min(limit, deadline - int(time.time()) - reserve)
        report["arms"][key] = launch(key, arm, 1024, budget, profile=profile)
        save(dest, report)

    def compare(control, candidate):
        key = f"{candidate}_vs_{control}"
        if deadline - time.time() > 36 and all(
                report["arms"][label].get("status") in ("passed", "measured")
                for label in (control, candidate)):
            report["gates"][key] = gate(control, candidate)
            save(dest, report)
            return report["gates"][key]
        return dict(passed=False, error="Missing measurements or exhausted deadline")

    def release_capture(label):
        (ROOT / "results" / f"{label}.capture.pt").unlink(missing_ok=True)

    labels = {arm: f"{arm}_1024x1" for arm in ARMS}
    r4 = labels[CONTROL]
    ablations = list(ARMS)[:7]
    isolation = list(ARMS)[7:]
    for arm in ablations:
        measure(labels[arm], arm, 300 if arm == CONTROL else 180, 425, profile=arm == CONTROL)
        if arm != CONTROL:
            compare(r4, labels[arm])
            release_capture(labels[arm])
    passing = {arm: report["arms"][labels[arm]]["median_ms"] for arm in ablations[3:]
               if report["gates"].get(f"{labels[arm]}_vs_{r4}", {}).get("passed")}
    if passing:
        winner = min(passing, key=passing.get)
        report["winner"] = winner
        report["winner_label"] = labels[winner]
        save(dest, report)
        for index, order in enumerate(((CONTROL, winner), (winner, CONTROL), (CONTROL, winner)), 1):
            pair_labels = {arm: f"pair{index}_{'control' if arm == CONTROL else 'candidate'}_1024x1" for arm in order}
            for arm in order:
                measure(pair_labels[arm], arm, 150, 210, profile=index == 1 and arm == winner)
            ref, cand = pair_labels[CONTROL], pair_labels[winner]
            numerical = compare(ref, cand)
            release_capture(ref)
            release_capture(cand)
            pair = dict(index=index, order=list(order), control=ref, candidate=cand,
                        numerical_pass=numerical["passed"],
                        **strict_pair(report["arms"][ref], report["arms"][cand]))
            report["pairs"].append(pair)
            save(dest, report)
        controls = [v["control_min_ms"] for v in report["pairs"] if "control_min_ms" in v]
        candidates = [v["candidate_max_ms"] for v in report["pairs"] if "candidate_max_ms" in v]
        report["raw_strict_separation_all_pairs"] = len(controls) == len(candidates) == 3 and max(candidates) < min(controls)
        trace = report["arms"].get("pair1_candidate_1024x1", {}).get("trace", {})
        counts = trace.get("dispatch_counts", {})
        wanted = set(DISPATCH) - {"r10_segmented_coarse_forward"}
        if PACK_FLAG in ARMS[winner][1]:
            wanted.add("r10_segmented_coarse_forward")
        report["missing_optimized_dispatches"] = sorted(name for name in wanted if not counts.get(name))
        report["qualified"] = (len(report["pairs"]) == 3 and all(
            p["dominance"] and p["raw_strict_separation"] and p["numerical_pass"] for p in report["pairs"])
            and report["raw_strict_separation_all_pairs"] and not report["missing_optimized_dispatches"]
            and "trace" in report["arms"][r4])
        save(dest, report)

    budget = min(320, deadline - int(time.time()) - 140)
    report["vertical_screen"] = launch("vertical_screen", "screen", 1024, budget)
    screen_path = ROOT / "screen-vertical.json"
    screen = read(screen_path)["cases"] if screen_path.is_file() else {}
    report["vertical_screen"]["missing_cases"] = sorted(SCREEN_CASES - screen.keys())
    report["vertical_screen"]["incomplete_cases"] = sorted(k for k, v in screen.items() if v["status"] != "complete")
    save(dest, report)

    for arm in isolation:
        measure(labels[arm], arm, 125, 36)
        compare(r4, labels[arm])
        release_capture(labels[arm])
    release_capture(r4)
    candidates = [labels[arm] for arm in ablations[1:]+isolation]
    required = all(f"{label}_vs_{r4}" in report["gates"] for label in candidates)
    required &= not (report["vertical_screen"]["missing_cases"] or report["vertical_screen"]["incomplete_cases"])
    report["status"] = "complete" if required else "partial"
    save(dest, report)


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    if sys.argv[1:2] == ["--worker"]:
        worker()
    else:
        main()
