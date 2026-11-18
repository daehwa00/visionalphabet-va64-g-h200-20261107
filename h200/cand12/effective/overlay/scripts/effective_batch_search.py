"""Isolated, finite full-update calibration for an effective batch of 1024."""

from __future__ import annotations

import contextlib
import copy
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch._inductor import config as inductor

from automatic_training_layout import (
    BUNDLE,
    BUNDLED_LAYOUTS,
    DOT_BUNDLE,
    DOT_NAME,
    LAYOUT_NAME,
    PROFILE_NAME,
    SEGMENT_BUNDLE,
    SEGMENT_NAME,
    PROJECT,
    atomic_json,
    candidates,
    capture_launch_scopes,
    check_compiled_plan,
    compare_tensors,
    digest,
    read_json,
    resolve,
    restore_rng,
    stored_geometry,
    workload_geometries,
)

MODULE = Path(__file__).resolve()
VERSION = 2
MICROBATCHES = (1024, 512, 256, 128)


def _positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _proof(value, counts=None):
    return (
        isinstance(value, dict)
        and value.get("passed") is True
        and all(
            isinstance(value.get(group), dict)
            and isinstance(value[group].get("count"), int)
            and value[group]["count"] > 0
            and (counts is None or value[group]["count"] == counts[group])
            and value[group].get("failed") == []
            and value[group].get("max_relative_l2", math.inf)
            <= (0.005 if group == "outputs" else 0.02)
            and (
                group == "grads"
                or value[group].get("max_abs", math.inf) <= (0.05 if group == "outputs" else 3e-5)
            )
            and all(
                isinstance(value[group].get(k), (int, float))
                and math.isfinite(value[group][k])
                and value[group][k] >= 0
                for k in ("max_abs", "max_relative_l2")
            )
            for group in ("outputs", "grads", "updates")
        )
    )


def _applied(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(row, dict)
            and isinstance(row.get("launches"), list)
            and bool(row["launches"])
            for row in value
        )
    )


OUTPUT_NAMES = {"joint", "affine", "fusion", "lrq", "descriptor"}


def _capture_schema(checks, parameter_names):
    expected = {
        "outputs": OUTPUT_NAMES,
        "grads": set(parameter_names),
        "updates": set(parameter_names),
    }
    return {
        "trainable_names": sorted(parameter_names),
        "counts": {group: len(checks.get(group, {})) for group in expected},
        "missing": {
            group: sorted(names - checks.get(group, {}).keys()) for group, names in expected.items()
        },
        "extra": {
            group: sorted(checks.get(group, {}).keys() - names) for group, names in expected.items()
        },
        "finite": all(
            isinstance(t, torch.Tensor) and bool(torch.isfinite(t).all())
            for group in expected
            for t in checks.get(group, {}).values()
        ),
    }


def _valid_schema(schema, expected=None):
    if not isinstance(schema, dict) or schema.get("finite") is not True:
        return False
    names = schema.get("trainable_names")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) for n in names)
        or len(set(names)) != len(names)
    ):
        return False
    counts = {"outputs": 5, "grads": len(names), "updates": len(names)}
    return (
        schema.get("counts") == counts
        and schema.get("missing") == {g: [] for g in counts}
        and schema.get("extra") == {g: [] for g in counts}
        and (expected is None or schema == expected)
    )


def _is_cuda_oom(exc):
    """Require a typed CUDA OOM in the real exception chain, never string matching."""
    pending, visited = [exc], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        for nested in (current.__cause__, current.__context__):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def _valid_pairs(trial, control, counts):
    pairs = trial["pairs"]
    if len(pairs) != 3:
        return False
    micro, records_digest = trial["microbatch"], trial["records_digest"]
    for i, pair in enumerate(pairs):
        order = ["candidate", "control"] if i == 1 else ["control", "candidate"]
        if (
            pair["pair"] != i + 1
            or pair.get("order") != order
            or pair["numerical_pass"] is not True
        ):
            return False
        if (pair["candidate_microbatch"], pair["candidate_records_digest"]) != (
            micro,
            records_digest,
        ):
            return False
        if (pair["control_microbatch"], pair["control_records_digest"]) != control:
            return False
        if not all(
            _proof(pair.get(k), counts)
            for k in ("candidate_reference_checks", "control_reference_checks")
        ):
            return False
        if not all(
            _positive(pair.get(k))
            for k in ("candidate_ms", "control_ms", "candidate_peak_mib", "control_peak_mib")
        ):
            return False
        if records_digest != digest({}) and not _applied(pair.get("applied")):
            return False
        if control[0] == micro and not _proof(pair.get("checks"), counts):
            return False
    return max(p["candidate_ms"] for p in pairs) < min(p["control_ms"] for p in pairs)


def valid_receipt(doc, key):
    """Validate the entire approved baseline-to-winner chain, including counts."""
    try:
        if (
            not isinstance(doc, dict)
            or doc.get("version") != VERSION
            or doc.get("key") != key
            or digest(doc.get("workload")) != key
        ):
            return False
        if (
            doc.get("search_complete") is not True
            or not isinstance(doc.get("trials"), list)
            or not doc["trials"]
        ):
            return False
        baseline, micro = doc["baseline_microbatch"], doc["microbatch"]
        if baseline not in (128, 64, 32) or micro not in (*MICROBATCHES, 64, 32):
            return False
        schema = doc["capture_schema"]
        if not _valid_schema(schema):
            return False
        counts = schema["counts"]
        assessed = doc["assessed_microbatches"]
        if not all(
            str(m) in assessed and assessed[str(m)]["status"] in ("ok", "oom")
            for m in (*MICROBATCHES, baseline)
        ):
            return False
        if assessed[str(baseline)]["status"] != "ok":
            return False
        for assessment in assessed.values():
            if assessment["status"] == "ok" and (
                not _positive(assessment.get("ms"))
                or not _positive(assessment.get("peak_mib"))
                or not _proof(assessment.get("reference_checks"), counts)
                or not _valid_schema(assessment.get("capture_schema"), schema)
            ):
                return False
        if not isinstance(doc.get("portfolio"), list) or len(set(doc["portfolio"])) != len(
            doc["portfolio"]
        ):
            return False
        screened_names = {t.get("plan") for t in doc["trials"]}
        if not set(doc["portfolio"]).issubset(screened_names):
            return False
        current, confirmed = (baseline, digest({})), []
        for trial in doc["trials"]:
            if trial.get("status") not in (
                "screened",
                "numerical_failure",
                "oom",
                "confirmed",
                "rejected",
            ):
                return False
            if trial.get("status") == "screened" and trial.get("plan") != "default-only":
                screen = trial.get("screen", {})
                if (
                    not _proof(trial.get("reference_checks"), counts)
                    or not _positive(screen.get("ms"))
                    or not _positive(screen.get("peak_mib"))
                    or not _valid_schema(screen.get("capture_schema"), schema)
                ):
                    return False
                if screen["ms"] < assessed[str(baseline)]["ms"] and not any(
                    t.get("plan", "").startswith(trial["plan"] + "-round-") for t in doc["trials"]
                ):
                    return False
            if trial.get("status") == "confirmed":
                if not _valid_pairs(trial, current, counts):
                    return False
                current = trial["microbatch"], trial["records_digest"]
                confirmed.append(trial)
        if doc["status"] == "baseline":
            return (
                doc["records"] == {}
                and micro == baseline
                and doc["winning_pairs"] == []
                and not confirmed
            )
        if doc["status"] != "winner" or not isinstance(doc["records"], dict) or not confirmed:
            return False
        return (
            current == (micro, digest(doc["records"]))
            and confirmed[-1]["plan"] == doc["winner"]
            and confirmed[-1]["pairs"] == doc["winning_pairs"]
        )
    except (KeyError, TypeError, ValueError, StopIteration, AttributeError):
        return False


def worker(jobfile):
    from effective_batch_update import run_update
    from va_imagenet100_runtime import _build_runtime

    job = read_json(jobfile)
    snapshot = torch.load(jobfile.parent / "snapshot.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda", snapshot["index"])
    torch.cuda.set_device(device)
    cfg = snapshot["settings"]
    torch.set_num_threads(cfg["threads"])
    torch.set_float32_matmul_precision(cfg["matmul"])
    torch.backends.cudnn.benchmark = cfg["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = cfg["cudnn_tf32"]
    torch.backends.cudnn.deterministic = cfg["cudnn_deterministic"]
    torch.use_deterministic_algorithms(cfg["deterministic"], warn_only=cfg["warn_only"])
    inductor.emulate_precision_casts = cfg["emulate_precision_casts"]
    factories, recipe = snapshot["factories"], snapshot["recipe"]
    model = resolve(factories["build"])(
        snapshot["variant"], resolve(factories["config"])(**snapshot["model_config"])
    ).to(device)
    model = resolve(factories["prepare"])(model, recipe)
    model.train()
    opt = resolve(factories["optimizer"])(model, recipe)
    batch = tuple(v.to(device) if isinstance(v, torch.Tensor) else v for v in snapshot["batch"])

    def reset():
        model.load_state_dict(snapshot["model"], strict=True)
        opt.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        opt.zero_grad(set_to_none=True)
        restore_rng(snapshot["rng"], device)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            workload_geometries(
                job["records"], discovery=str(jobfile.parent) if job["discover"] else ""
            )
        )
        names = (LAYOUT_NAME, PROFILE_NAME, SEGMENT_NAME, DOT_NAME)
        found = {n: stack.enter_context(capture_launch_scopes(n)) for n in names}
        reset()
        runtime = _build_runtime(model, recipe)

        def step(capture=False):
            return run_update(
                model,
                runtime,
                opt,
                batch,
                job["microbatch"],
                recipe["precision"],
                capture=capture,
                metrics=True,
            )

        checks = step(True)["checks"]
        schema = _capture_schema(
            checks, [name for name, param in model.named_parameters() if param.requires_grad]
        )
        if not _valid_schema(schema):
            raise ValueError(f"Incomplete or nonfinite full-update capture: {schema}")
        torch.save(checks, jobfile.with_suffix(".pt"))
        regime = (
            "compiled"
            if os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode")
            else "eager"
        )
        observed = {
            n: sorted({s.key for s in scopes if s.execution_regime == regime})
            for n, scopes in found.items()
        }
        defaults = {}
        for n, scopes in found.items():
            defaults[n] = {}
            for scope in scopes:
                if scope.key not in observed[n]:
                    continue
                geom = stored_geometry(n, scope=scope) or stored_geometry(
                    n,
                    scope=scope,
                    path={
                        LAYOUT_NAME: BUNDLED_LAYOUTS,
                        PROFILE_NAME: BUNDLE,
                        SEGMENT_NAME: SEGMENT_BUNDLE,
                        DOT_NAME: DOT_BUNDLE,
                    }[n],
                )
                defaults[n][scope.key] = (
                    dict(blocks=geom.blocks, num_warps=geom.num_warps, num_stages=geom.num_stages)
                    if geom
                    else dict(blocks={}, num_warps=4, num_stages=1)
                )
        for _ in range(2):
            reset()
            step()
        samples, peaks = [], []
        groups, updates = (3, 2) if job["confirm"] else (1, 3)
        for _ in range(groups):
            wall = []
            for _ in range(updates):
                reset()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                step()
                torch.cuda.synchronize(device)
                wall.append(1000 * (time.perf_counter() - start))
                peaks.append(torch.cuda.max_memory_allocated(device) / 2**20)
            samples.append(statistics.mean(wall))
        result = dict(
            status="ok",
            ms=statistics.median(samples),
            groups_ms=samples,
            peak_mib=max(peaks),
            observed=observed,
            defaults=defaults,
            capture_schema=schema,
        )
        result["images_per_second"] = 1024000 / result["ms"]
        if job["profile"] and job["records"]:
            reset()
            torch.cuda.synchronize(device)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                step()
                torch.cuda.synchronize(device)
            trace = jobfile.with_suffix(".trace.json")
            prof.export_chrome_trace(str(trace))
            result["applied"] = check_compiled_plan(trace, job["records"])
        atomic_json(jobfile.with_suffix(".result.json"), result)


def _run_arm(work, label, microbatch, records, *, discover=False, confirm=False, profile=False):
    job = work / (label + ".job")
    atomic_json(
        job,
        dict(
            records=records,
            microbatch=microbatch,
            discover=discover,
            confirm=confirm,
            profile=profile,
        ),
    )
    env = dict(os.environ)
    for name in ("VA_IN100_PROGRESS_PATH", "WANDB_PROJECT"):
        env.pop(name, None)
    env["WANDB_MODE"] = "disabled"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(MODULE.parent), str(PROJECT / "src"), env.get("PYTHONPATH", ""))
    )
    with job.with_suffix(".log").open("w") as log:
        child = subprocess.run(
            [sys.executable, str(MODULE), "--worker", str(job)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    result = read_json(job.with_suffix(".result.json"))
    if child.returncode or result.get("status") not in ("ok", "oom"):
        raise RuntimeError(
            f"{label}: child execution failed: " + job.with_suffix(".log").read_text()[-1400:]
        )
    if result["status"] == "oom":
        return result, None
    path = job.with_suffix(".pt")
    checks = torch.load(path, weights_only=True)
    path.unlink()
    return result, checks


def measure(snapshot, key, info, work):
    work.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, work / "snapshot.pt")
    assessed, refs, discoveries, trials = {}, {}, {}, []
    counter = 0

    def arm(micro, records, **options):
        nonlocal counter
        counter += 1
        print(
            f"[effective batch] arm {counter}: micro={micro} records={digest(records)[:10]} confirm={options.get('confirm', False)}",
            flush=True,
        )
        result, checks = _run_arm(work, f"arm-{counter:04d}", micro, records, **options)
        if result["status"] == "ok":
            expected = next((v["capture_schema"] for v in discoveries.values()), None)
            if not _valid_schema(result.get("capture_schema"), expected):
                raise RuntimeError("Incomplete/mismatched full-update capture schema")
        return result, checks

    baseline = None
    for micro in (128, 64, 32):
        result, checks = arm(micro, {}, discover=True)
        assessed[str(micro)] = result
        if result["status"] == "ok":
            baseline = micro
            result["reference_checks"] = compare_tensors(checks, checks)
            refs[micro], discoveries[micro] = checks, result
            break
    if baseline is None:
        raise RuntimeError("No feasible baseline microbatch (128, 64, 32)")
    for micro in MICROBATCHES:
        if str(micro) in assessed:
            continue
        result, checks = arm(micro, {}, discover=True)
        assessed[str(micro)] = result
        if result["status"] == "ok":
            result["reference_checks"] = compare_tensors(checks, checks)
            refs[micro], discoveries[micro] = checks, result
    best = dict(microbatch=baseline, records={}, ms=discoveries[baseline]["ms"], plan="baseline")
    winner, winning_pairs = None, []
    portfolio = []
    for micro, discovery in discoveries.items():
        if micro != baseline:
            portfolio.append((f"micro-{micro}-default", micro, {}))
        for name, records in candidates(discovery["observed"]).items():
            if all(
                discovery["defaults"].get(n, {}).get(k) == g
                for n, table in records.items()
                for k, g in table.items()
            ):
                continue
            portfolio.append((f"micro-{micro}-{name}", micro, records))
    promising = []
    for name, micro, records in portfolio:
        entry = dict(plan=name, microbatch=micro, records_digest=digest(records), pairs=[])
        trials.append(entry)
        result, tensors = arm(micro, records)
        if result["status"] == "oom":
            entry["status"] = "oom"
            continue
        proof = compare_tensors(refs[micro], tensors)
        entry.update(
            status="screened" if proof["passed"] else "numerical_failure",
            screen=result,
            reference_checks=proof,
        )
        if proof["passed"] and result["ms"] < best["ms"]:
            promising.append((name, micro, records, entry, result["ms"]))
    # Revisit a finite promising portfolio against the current winner, including
    # a second pass after the control geometry changes.
    for round_index in range(2):
        changed = False
        for name, micro, records, screened, _ in sorted(promising, key=lambda row: row[4]):
            if name == best["plan"]:
                continue
            entry = dict(
                plan=f"{name}-round-{round_index}",
                microbatch=micro,
                records_digest=digest(records),
                pairs=[],
                status="rejected",
            )
            trials.append(entry)
            for i in range(3):
                order = ["candidate", "control"] if i == 1 else ["control", "candidate"]
                pair = dict(
                    pair=i + 1,
                    order=order,
                    candidate_microbatch=micro,
                    control_microbatch=best["microbatch"],
                    candidate_records_digest=digest(records),
                    control_records_digest=digest(best["records"]),
                )
                tensors = {}
                for side in order:
                    m, plan = (
                        (micro, records)
                        if side == "candidate"
                        else (best["microbatch"], best["records"])
                    )
                    result, tensors[side] = arm(
                        m, plan, confirm=True, profile=side == "candidate" and bool(records)
                    )
                    if result["status"] == "oom":
                        entry["status"] = "oom"
                        break
                    pair[side + "_ms"] = result["ms"]
                    pair[side + "_peak_mib"] = result["peak_mib"]
                    pair[side + "_reference_checks"] = compare_tensors(refs[m], tensors[side])
                    if side == "candidate":
                        pair["applied"] = result.get("applied", [])
                if entry["status"] == "oom":
                    break
                pair["checks"] = (
                    compare_tensors(tensors["control"], tensors["candidate"])
                    if micro == best["microbatch"]
                    else {
                        "passed": True,
                        "reason": "different microbatch BN semantics; independently compared with same-micro defaults",
                    }
                )
                pair["numerical_pass"] = all(
                    pair[k]["passed"]
                    for k in ("checks", "candidate_reference_checks", "control_reference_checks")
                )
                entry["pairs"].append(pair)
                if (
                    not pair["numerical_pass"]
                    or pair["candidate_ms"] >= pair["control_ms"]
                    or (records and not _applied(pair["applied"]))
                ):
                    break
            pairs = entry["pairs"]
            if (
                len(pairs) == 3
                and all(p["numerical_pass"] for p in pairs)
                and (not records or all(_applied(p.get("applied")) for p in pairs))
                and max(p["candidate_ms"] for p in pairs) < min(p["control_ms"] for p in pairs)
            ):
                entry["status"] = "confirmed"
                best = dict(
                    plan=name,
                    microbatch=micro,
                    records=records,
                    ms=statistics.median(p["candidate_ms"] for p in pairs),
                )
                winner, winning_pairs = entry["plan"], pairs
                changed = True
        if not changed:
            break
    if not trials:
        trials.append(
            dict(
                plan="default-only",
                status="screened",
                microbatch=baseline,
                records_digest=digest({}),
                pairs=[],
            )
        )
    return dict(
        version=VERSION,
        key=key,
        workload=info,
        status="winner" if winner else "baseline",
        winner=winner,
        microbatch=best["microbatch"],
        records=best["records"],
        baseline_microbatch=baseline,
        capture_schema=discoveries[baseline]["capture_schema"],
        trials=trials,
        winning_pairs=winning_pairs,
        assessed_microbatches=assessed,
        search_complete=True,
        screening=dict(warmup=2, groups=1, updates=3),
        confirmation=dict(warmup=2, groups=3, updates=2),
        portfolio=[name for name, _, _ in portfolio],
    )


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This worker is invoked by the training entry point.")
    job_path = Path(sys.argv[2])
    try:
        worker(job_path)
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        atomic_json(
            job_path.with_suffix(".result.json"),
            dict(status="oom", error=str(exc), exception_type=type(exc).__name__),
        )
