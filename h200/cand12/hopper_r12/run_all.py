"""Qualify R12 packed and specialized NVRTC scan kernels on Whole H200 (1024x1, BF16 + fused AdamW)."""
from __future__ import annotations

from functools import partial
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
R12_PACKED = "LNET_R12_PACKED"
R12_INDEX32 = "LNET_R12_INDEX32"
R12_PRECISE = "LNET_R12_PRECISE"
R11_VERTICAL = ("LNET_R11_CUDA_VFWD", "LNET_R11_CUDA_VBACK")
ALL_FLAGS = R10_WINNER + R11_FLAGS + (R12_PACKED, R12_INDEX32, R12_PRECISE)

CONTROL = "r4_control"
INCUMBENT = "r11_vscan"
ARMS = {
    CONTROL: ("none", ()),
    INCUMBENT: ("both", (*R10_WINNER, *R11_VERTICAL)),
    "r12_vscan_packed": ("both", (*R10_WINNER, *R11_VERTICAL, R12_PACKED)),
    "r12_full_packed": ("none", (*R10_WINNER, *R11_FLAGS, R12_PACKED)),
    "r12_full_packed_index32": ("none", (*R10_WINNER, *R11_FLAGS, R12_PACKED, R12_INDEX32)),
    "r12_full": ("none", (*R10_WINNER, *R11_FLAGS, R12_PACKED, R12_INDEX32, R12_PRECISE)),
    "r12_vscan_all": ("both", (*R10_WINNER, *R11_VERTICAL, R12_PACKED, R12_INDEX32, R12_PRECISE)),
    "r12_r10_forward": ("none", (*R10_WINNER, "LNET_R11_CUDA_VBACK", "LNET_R11_CUDA_HBACK",
                                R12_PACKED, R12_INDEX32)),
}
PROFILE_ARMS = (CONTROL, INCUMBENT, "r12_vscan_packed", "r12_full_packed", "r12_full")


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
    *R10_FALLBACK_KERNELS, "_reduce_parameter_grads",
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


def load_sm90a_modules():
    import torch
    from lnet.r11_cuda_hscan import _get_module as hscan_module
    from lnet.r11_cuda_vscan import _get_module as vscan_module
    from lnet.r12_cuda_scans import _vertical_module

    capability = list(torch.cuda.get_device_capability(0))
    if capability != [9, 0]:
        raise RuntimeError(f"Expected SM90a preflight device, got {capability}")
    modules = {
        "r11_vscan": (vscan_module, R11_VFWD_KERNELS + R11_VBACK_KERNELS),
        "r11_hscan": (hscan_module, R11_HFWD_KERNELS + R11_HBACK_KERNELS),
        "r12_index32": (partial(_vertical_module, index32=True, bias_last=False),
                        R11_VFWD_KERNELS + R11_VBACK_KERNELS),
        "r12_index32_bias_last": (partial(_vertical_module, index32=True, bias_last=True), R11_VFWD_KERNELS),
    }
    result = dict(ok=True, cap=capability, architecture="sm_90a", modules={})
    for name, (load, expected) in modules.items():
        start = time.monotonic()
        module = load(0)
        if module._funcs.keys() != set(expected):
            raise RuntimeError(f"Unexpected NVRTC symbols for {name}")
        result["modules"][name] = dict(functions=len(expected), compile_load_s=round(time.monotonic() - start, 3))
    print(json.dumps(result))


def nvrtc_preflight():
    source = ROOT / "current"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source / "scripts", source / "src", HERE))))
    try:
        proc = subprocess.run([sys.executable, str(__file__), "--nvrtc-preflight"],
                              env=env, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return dict(ok=False, error="SM90a NVRTC preflight exceeded 90 s")
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
    counts = defaultdict(int)
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
        counts[family] += 1
    rows = [
        dict(family=key[0], name=key[1], grid=key[2], block=key[3], calls=value[0], ms_update=value[1])
        for key, value in sorted(bins.items(), key=lambda item: -item[1][1])
    ]
    other = defaultdict(lambda: [0, 0.0])
    for row in rows:
        if row["family"] == "other":
            aggregate = other[row["name"]]
            aggregate[0] += row["calls"]
            aggregate[1] += row["ms_update"]
        row["ms_update"] = round(row["ms_update"], 6)
    other_by_kernel = [(name, calls, round(ms, 6))
                       for name, (calls, ms) in sorted(other.items(), key=lambda item: -item[1][1])]
    dispatch = {name: sum(row["calls"] for row in rows if row["name"] == name
                          or row["name"].startswith(name + "_")) for name in DISPATCH}
    launches = {name: [row for row in rows if row["name"] == name or row["name"].startswith(name + "_")]
                for name, count in dispatch.items() if count}
    return dict(path=str(path), dispatch_counts=dispatch, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                total_kernel_ms=sum(totals.values()), kernel_event_count=sum(r["calls"] for r in rows), launches=launches,
                families_ms=dict(totals), families_events=dict(counts), kernel_bins=len(rows),
                top=rows[:100], other_by_kernel=other_by_kernel)


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
    env.update(dict.fromkeys(ALL_FLAGS, "0"))
    env.update(dict.fromkeys(ARMS[arm][1], "1"))
    fixture = Path(os.environ["COLD_INPUTS"])
    cmd = [sys.executable, str(HERE / "run_all.py"), "--worker", arm, str(micro),
           "--checkpoint", str(fixture / "checkpoint.pt"), "--fixture", str(fixture / "fixture.json"),
           "--contract", os.environ["CONTRACT"], "--output", str(output), "--microbatch", str(micro),
           "--capture", "--groups", "3", "--samples", "2"]
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


def expected_kernels(arm):
    flags = ARMS[arm][1]
    wanted = set()
    if "LNET_R11_CUDA_VFWD" in flags:
        wanted.update(R11_VFWD_KERNELS)
    if "LNET_R11_CUDA_VBACK" in flags:
        wanted.update(R11_VBACK_KERNELS)
    if "LNET_R11_CUDA_HFWD" in flags:
        wanted.update(R11_HFWD_KERNELS)
    if "LNET_R11_CUDA_HBACK" in flags:
        wanted.update(R11_HBACK_KERNELS)
    if R12_PACKED in flags:
        wanted.add("_reduce_parameter_grads")
    if arm != CONTROL:
        wanted.add("r10_rms56_partial")
        if "LNET_R11_CUDA_HFWD" not in flags:
            wanted.update(("r10_hfwd56_seg4", "r8_horizontal_forward"))
        if "LNET_R11_CUDA_HBACK" not in flags:
            wanted.update(("hback56", "r8_horizontal_backward"))
        if "LNET_R11_CUDA_VFWD" not in flags:
            wanted.update(("r7_associative_forward", "r10_segmented_coarse_forward"))
    return wanted


def missing_dispatches(arm, measurement):
    counts = measurement.get("trace", {}).get("dispatch_counts", {})
    return sorted(name for name in expected_kernels(arm) if not counts.get(name))


def numerical_pass(gate_result):
    return gate_result.get("passed", False) and gate_result.get("requested_thresholds", {}).get("passed", False)


def strict_separation(pairs, expected):
    return (len(pairs) == expected
            and all("control_min_ms" in pair and "candidate_max_ms" in pair for pair in pairs)
            and max(pair["candidate_max_ms"] for pair in pairs) < min(pair["control_min_ms"] for pair in pairs))


def main():
    start = int(os.environ["HOPPER_R12_START"])
    topology = gpu_topology()
    deadline = int(os.environ["HOPPER_R12_DEADLINE"]) - 15
    dest = ROOT / "result.json"
    report = dict(gpu_topology=topology, arms={}, gates={}, pairs=[], r11_pairs=[], status="running",
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
        raise RuntimeError("R12 SM90a NVRTC preflight failed")

    (ROOT / "results").mkdir(exist_ok=True)

    def measure(key, arm, limit, reserve, profile=False):
        budget = min(limit, deadline - int(time.time()) - reserve)
        row = launch(key, arm, 1024, budget, profile=profile)
        row["arm"] = arm
        row["expected_dispatches"] = sorted(expected_kernels(arm))
        if profile:
            row["missing_optimized_dispatches"] = missing_dispatches(arm, row)
        report["arms"][key] = row
        save(dest, report)

    def compare(control, candidate):
        key = f"{candidate}_vs_{control}"
        if deadline - time.time() > 36 and all(
                report["arms"][label].get("status") == "passed"
                for label in (control, candidate)):
            result = gate(control, candidate)
        else:
            result = dict(passed=False, error="Missing measurements or exhausted deadline")
        report["gates"][key] = result
        save(dest, report)
        return result

    def release_capture(label):
        (ROOT / "results" / f"{label}.capture.pt").unlink(missing_ok=True)

    def interleave(reference, winner, pair_key, prefix, order_by_pair):
        for index, control_first in enumerate(order_by_pair, 1):
            ref = f"{prefix}{index}_control_1024x1"
            cand = f"{prefix}{index}_candidate_1024x1"
            order = (reference, winner) if control_first else (winner, reference)
            for arm in order:
                label = ref if arm == reference else cand
                measure(label, arm, 150, 40, profile=(reference == CONTROL and index == 1 and arm == winner))
            numerical = compare(ref, cand)
            release_capture(ref)
            release_capture(cand)
            report[pair_key].append(dict(
                index=index, order=list(order), control=ref, candidate=cand,
                numerical_pass=numerical_pass(numerical),
                **strict_pair(report["arms"][ref], report["arms"][cand]),
            ))
            save(dest, report)

    labels = {arm: f"{arm}_1024x1" for arm in ARMS}
    r4 = labels[CONTROL]
    for arm, label in labels.items():
        measure(label, arm, 300 if arm == CONTROL else 210, 400, profile=arm in PROFILE_ARMS)
        if arm != CONTROL:
            compare(r4, label)
            release_capture(label)
    release_capture(r4)

    passing = {
        arm: report["arms"][label]["median_ms"]
        for arm, label in labels.items()
        if arm.startswith("r12_")
        and numerical_pass(report["gates"][f"{label}_vs_{r4}"])
    }
    report.update(winner=None, qualified=False)
    if passing:
        winner = min(passing, key=passing.get)
        report.update(winner=winner, winner_label=labels[winner])
        save(dest, report)
        interleave(CONTROL, winner, "pairs", "pair", (True, False, True))
        interleave(INCUMBENT, winner, "r11_pairs", "r11pair", (True, False))
        report["raw_strict_separation_all_pairs"] = strict_separation(report["pairs"], 3)
        report["r11_raw_strict_separation_all_pairs"] = strict_separation(report["r11_pairs"], 2)
        report["r11_dominance"] = len(report["r11_pairs"]) == 2 and all(p["dominance"] for p in report["r11_pairs"])
        winner_trace = report["arms"]["pair1_candidate_1024x1"]
        report["missing_optimized_dispatches"] = missing_dispatches(winner, winner_trace)
        report["qualified"] = (
            report["raw_strict_separation_all_pairs"]
            and all(p["dominance"] and p["raw_strict_separation"] and p["numerical_pass"] for p in report["pairs"])
            and "trace" in winner_trace
            and not report["missing_optimized_dispatches"]
            and "trace" in report["arms"][r4]
        )

    required = (all(row.get("status") == "passed" for row in report["arms"].values())
                and all("official" in result for result in report["gates"].values())
                and all("trace" in report["arms"][labels[arm]]
                        and not report["arms"][labels[arm]]["missing_optimized_dispatches"] for arm in PROFILE_ARMS)
                and (not passing or (
                    len(report["pairs"]) == 3 and len(report["r11_pairs"]) == 2
                    and "trace" in report["arms"]["pair1_candidate_1024x1"]
                )))
    report["status"] = "complete" if required else "partial"
    save(dest, report)
    if not required:
        raise RuntimeError("Required Whole H200 measurements, comparisons, or trace evidence are missing")


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    if sys.argv[1:2] == ["--worker"]:
        worker()
    elif sys.argv[1:2] == ["--nvrtc-preflight"]:
        load_sm90a_modules()
    else:
        main()
