"""Qualify R11 custom NVRTC CUDA C++ scan kernels on Whole H200 (1024x1, BF16 + fused AdamW)."""
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
R10_FLAGS = ("LNET_R10_HFWD56_SEG4", "LNET_R10_RMS_DW56", "LNET_R10_TUNED_LAUNCH", "LNET_R10_SEG4PACK2")
R10_WINNER = R6_FLAGS + R7_FLAGS + R89_FLAGS + R10_FLAGS
R11_FLAGS = ("LNET_R11_CUDA_VFWD", "LNET_R11_CUDA_VBACK", "LNET_R11_CUDA_HFWD", "LNET_R11_CUDA_HBACK")
ALL_FLAGS = R10_WINNER + R11_FLAGS

CONTROL = "r4_control"
ARMS = {
    CONTROL: ("none", ()),
    "r10_incumbent": ("both", R10_WINNER),
    "r11_vback": ("both", (*R10_WINNER, "LNET_R11_CUDA_VBACK")),
    "r11_vscan": ("both", (*R10_WINNER, "LNET_R11_CUDA_VFWD", "LNET_R11_CUDA_VBACK")),
    "r11_hscan": ("both", (*R10_WINNER, "LNET_R11_CUDA_HFWD", "LNET_R11_CUDA_HBACK")),
    "r11_full": ("none", (*R10_WINNER, *R11_FLAGS)),
    "r11_full_r5": ("both", (*R10_WINNER, *R11_FLAGS)),
}

R11_VFWD_KERNELS = (
    "vscan_fwd_56_d", "vscan_fwd_56_c", "vscan_fwd_28_d",
    "vscan_fwd_28_c", "vscan_fwd_14_d", "vscan_fwd_14_c",
)
R11_VBACK_KERNELS = (
    "vscan_bwd_56_d", "vscan_bwd_56_c", "vscan_bwd_28_d",
    "vscan_bwd_28_c", "vscan_bwd_14_d", "vscan_bwd_14_c",
)
R11_HFWD_KERNELS = ("hscan_fwd_56", "hscan_fwd_28", "hscan_fwd_14")
R11_HBACK_KERNELS = (
    "hscan_bwd_56_d", "hscan_bwd_56_c", "hscan_bwd_28_d",
    "hscan_bwd_28_c", "hscan_bwd_14_d", "hscan_bwd_14_c",
)
R10_FALLBACK_KERNELS = (
    "r7_associative_forward", "r7_direct_backward", "coarse_associative_backward",
    "hback56", "r8_horizontal_forward", "r8_horizontal_backward",
    "r9_channel_weight_partial", "r10_hfwd56_seg4", "r10_rms56_partial",
    "r10_segmented_coarse_forward",
)
DISPATCH = (
    *R11_VFWD_KERNELS, *R11_VBACK_KERNELS, *R11_HFWD_KERNELS, *R11_HBACK_KERNELS,
    *R10_FALLBACK_KERNELS,
)


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)
    persist = os.environ.get("PERSIST_ROOT")
    if persist and path.name == "result.json":
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


def nvrtc_preflight():
    """Verify R11 SM90a NVRTC cubin compilation and module loading before running arms."""
    source = ROOT / "current"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source / "scripts", source / "src", HERE))))
    code = (
        "import json, time, torch; "
        "from lnet.r11_cuda_vscan import _get_module as gv; "
        "from lnet.r11_cuda_hscan import _get_module as gh; "
        "t0 = time.monotonic(); mv = gv(0); t1 = time.monotonic(); mh = gh(0); t2 = time.monotonic(); "
        "print(json.dumps({'ok': True, 'cap': list(torch.cuda.get_device_capability(0)), "
        "'vscan_funcs': len(mv._funcs), 'hscan_funcs': len(mh._funcs), "
        "'vscan_compile_s': round(t1 - t0, 3), 'hscan_compile_s': round(t2 - t1, 3)}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    if proc.returncode:
        return dict(ok=False, returncode=proc.returncode, stderr=proc.stderr[-900:], stdout=proc.stdout[-400:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


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
    records = workload(micro, ARMS[arm][0])
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    diagnose_nonfinite(output)
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
        if (name.startswith("vscan_bwd_") or name in ("_path_backward_kernel", "r7_direct_backward", "coarse_associative_backward")
                or "associative_backward" in name):
            family = "vback"
        elif (name.startswith("hscan_bwd_") or name == "hback56"
              or "horizontal_path_backward" in name or "bidirectional_backward" in name or "r8_horizontal_backward" in name):
            family = "hback"
        elif (name.startswith("vscan_fwd_") or name == "_path_forward_kernel"
              or "associative_forward" in name or "segmented_coarse_forward" in name):
            family = "vfwd"
        elif (name.startswith("hscan_fwd_") or "bidirectional_forward" in name
              or "r8_horizontal_forward" in name or "r10_hfwd56_seg4" in name):
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
                families_ms=dict(totals), top=rows[:40])


def launch(label, arm, micro, seconds, *, profile=False):
    output = ROOT / "results" / f"{label}.json"
    if seconds <= 0:
        return dict(status="budget_skipped", microbatch=micro)
    branch = "r4" if arm == CONTROL else "current"
    source = ROOT / branch
    cache = ROOT / "cache"
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join(map(str, (source / "scripts", source / "src", HERE))),
        LNET_LAUNCH_CACHE=str(cache / "launch"), TRITON_CACHE_DIR=str(cache / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), PYTHONUNBUFFERED="1",
    )
    env.update(dict.fromkeys(ARMS[arm][1], "1"))
    fixture = Path(os.environ["COLD_INPUTS"])
    cmd = [sys.executable, str(HERE / "run_all.py"), "--worker", arm, str(micro),
           "--checkpoint", str(fixture / "checkpoint.pt"), "--fixture", str(fixture / "fixture.json"),
           "--contract", os.environ["CONTRACT"], "--output", str(output), "--microbatch", str(micro),
           "--capture"]
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
                   detail="\n".join(logpath.read_text(errors="replace").splitlines()[-8:])[-900:])
    row["flags"] = {name: env.get(name, "0") for name in ALL_FLAGS}
    row["r5"] = ARMS[arm][0]
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


def expected_r11_kernels(flags):
    wanted = set()
    if "LNET_R11_CUDA_VFWD" in flags:
        wanted.update(R11_VFWD_KERNELS)
    if "LNET_R11_CUDA_VBACK" in flags:
        wanted.update(R11_VBACK_KERNELS)
    if "LNET_R11_CUDA_HFWD" in flags:
        wanted.update(R11_HFWD_KERNELS)
    if "LNET_R11_CUDA_HBACK" in flags:
        wanted.update(R11_HBACK_KERNELS)
    return wanted


def main():
    start = int(os.environ["HOPPER_R11_START"])
    topology = gpu_topology()
    deadline = int(os.environ["HOPPER_R11_DEADLINE"]) - 15
    dest = ROOT / "result.json"
    report = dict(gpu_topology=topology, arms={}, gates={}, pairs=[], r10_pairs=[], status="running",
                  start_epoch_s=start, deadline_epoch_s=deadline + 15)
    save(dest, report)
    if (topology["capability"] != [9, 0] or topology["pool_mode"] != "whole_h200"
            or topology["multi_processor_count"] != 132 or not 139 <= topology["total_memory_gib"] <= 140
            or topology["device_name"] != "NVIDIA H200 NVL"):
        raise RuntimeError("Visible device must match the Whole H200 request")
    print("GPU_TOPOLOGY", json.dumps(topology), flush=True)
    report["nvrtc_preflight"] = nvrtc_preflight()
    print("NVRTC_PREFLIGHT", json.dumps(report["nvrtc_preflight"]), flush=True)
    save(dest, report)
    if not report["nvrtc_preflight"].get("ok"):
        report["status"] = "nvrtc_preflight_failed"
        save(dest, report)
        raise RuntimeError("R11 SM90a NVRTC preflight failed")

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
    arm_list = list(ARMS)
    for arm in arm_list:
        do_profile = arm in (CONTROL, "r10_incumbent", "r11_full")
        measure(labels[arm], arm, 300 if arm == CONTROL else 210, 400, profile=do_profile)
        if arm != CONTROL:
            compare(r4, labels[arm])
            release_capture(labels[arm])

    r11_arms = [arm for arm in arm_list if arm.startswith("r11_")]
    passing = {
        arm: report["arms"][labels[arm]]["median_ms"]
        for arm in r11_arms
        if report["gates"].get(f"{labels[arm]}_vs_{r4}", {}).get("passed")
        and isinstance(report["arms"][labels[arm]].get("median_ms"), (int, float))
    }
    if passing:
        winner = min(passing, key=passing.get)
        report["winner"] = winner
        report["winner_label"] = labels[winner]
        save(dest, report)
        for index, order in enumerate(((CONTROL, winner), (winner, CONTROL), (CONTROL, winner)), 1):
            pair_labels = {arm: f"pair{index}_{'control' if arm == CONTROL else 'candidate'}_1024x1" for arm in order}
            for arm in order:
                measure(pair_labels[arm], arm, 150, 120, profile=index == 1 and arm == winner)
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
        wanted = expected_r11_kernels(ARMS[winner][1])
        report["missing_optimized_dispatches"] = sorted(name for name in wanted if not counts.get(name))
        report["qualified"] = (
            len(report["pairs"]) == 3
            and all(p["dominance"] and p["raw_strict_separation"] and p["numerical_pass"] for p in report["pairs"])
            and report["raw_strict_separation_all_pairs"]
            and not report["missing_optimized_dispatches"]
            and "trace" in report["arms"][r4]
        )
        save(dest, report)

        # Optional head-to-head alternating pair against R10 incumbent if budget permits.
        if deadline - int(time.time()) > 260:
            for index, order in enumerate((("r10_incumbent", winner), (winner, "r10_incumbent")), 1):
                pl = {arm: f"r10pair{index}_{'r10' if arm == 'r10_incumbent' else 'r11'}_1024x1" for arm in order}
                for arm in order:
                    measure(pl[arm], arm, 120, 40)
                ref, cand = pl["r10_incumbent"], pl[winner]
                release_capture(ref)
                release_capture(cand)
                report["r10_pairs"].append(dict(
                    index=index, order=list(order), control=ref, candidate=cand,
                    **strict_pair(report["arms"][ref], report["arms"][cand]),
                ))
                save(dest, report)

    release_capture(r4)
    candidates = [labels[arm] for arm in arm_list[1:]]
    required = all(f"{label}_vs_{r4}" in report["gates"] for label in candidates)
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
