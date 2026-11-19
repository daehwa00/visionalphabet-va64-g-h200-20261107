"""Isolated, finite full-update calibration for an effective batch of 1024."""

from __future__ import annotations

import contextlib
import copy
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch._inductor import config as inductor

from effective_batch_proposals import OFF, proposal_candidates, related_geometry, same_launch_scope

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
    published_candidates,
    resolve,
    restore_rng,
    stored_geometry,
    workload_geometries,
)

MODULE = Path(__file__).resolve()
VERSION = 2
MICROBATCHES = (1024, 512, 256, 128, 64, 32)
VJP_NAME = "effective_output_vjp"
STRUCT_NAME = "effective_structural_scan"
LEGACY_NAMES = (LAYOUT_NAME, PROFILE_NAME, SEGMENT_NAME, DOT_NAME)


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


def _scope_fields(key):
    return json.loads(key)


def _direct_scope(key):
    fields = _scope_fields(key)
    shape = fields["shape"]
    return (
        fields.get("execution_regime") == "compiled"
        and shape.get("height") == 56
        and shape.get("operation") == 1
        and shape.get("coarse") == 0
    )


def _parent_keys(observed, key):
    target = _scope_fields(key)
    return {
        name: next(
            (
                k
                for k in observed.get(name, [])
                if _scope_fields(k)["shape"] == target["shape"]
                and _scope_fields(k).get("dtype") == target.get("dtype")
                and _direct_scope(k)
            ),
            None,
        )
        for name in (PROFILE_NAME, SEGMENT_NAME, DOT_NAME)
    }


def _vjp_contract(records):
    contracts = []
    for key, marker in records.get(VJP_NAME, {}).items():
        if not marker.get("blocks"):
            if marker != {"blocks": {}, "num_warps": 4, "num_stages": 1}:
                raise ValueError("Invalid disabled VJP sentinel")
            if _direct_scope(key):
                keys = _parent_keys(records, key)
                parents = [records.get(name, {}).get(k) for name, k in keys.items()]
                if (
                    parents[0]
                    and all(g == parents[0] for g in parents)
                    and parents[0].get("blocks", {}).get("DOUBLING") == 1
                ):
                    shape = _scope_fields(key)["shape"]
                    contracts.append(
                        (tuple(shape[n] for n in ("batch", "height", "width", "modes")), 0)
                    )
            continue
        variant = marker["blocks"].get("VARIANT")
        if (
            not _direct_scope(key)
            or variant not in (1, 2)
            or marker != {"blocks": {"VARIANT": variant}, "num_warps": 4, "num_stages": 1}
        ):
            raise ValueError("Invalid effective output VJP marker scope/geometry")
        structural = related_geometry(records, STRUCT_NAME, key)
        if structural and structural.get("blocks"):
            raise ValueError("a structural backward implements its own output VJP")
        keys = _parent_keys(records, key)
        if any(k is None for k in keys.values()):
            raise ValueError("VJP marker requires explicit retained/segment/dot parents")
        geometries = [records[name][k] for name, k in keys.items()]
        if (
            not all(g == geometries[0] for g in geometries)
            or geometries[0].get("blocks", {}).get("DOUBLING") != 1
        ):
            raise ValueError("VJP parents must have identical four-row direct dot geometry")
        shape = _scope_fields(key)["shape"]
        contracts.append((tuple(shape[n] for n in ("batch", "height", "width", "modes")), variant))
    return contracts


def _dispatch_error(records, applied):
    """Describe the first missing or incorrect compiled leaf, including nested structural VJPs."""
    expected = []
    try:
        structural = _structural_contract(records)
        owns_output = {shape for phase, shape, v in structural if phase == "backward" and v}
        for shape, variant in _vjp_contract(records):
            if shape in owns_output:
                if variant:
                    return f"backward {shape}: a structural backward implements its own output VJP"
                continue
            expected.append(("backward", shape, dict(
                output_vjp_variant=variant, segment_rows=4, output_weight_bias_dot=True
            )))
        for phase, shape, variant in structural:
            fields = {"structural_variant": variant}
            if variant and phase in ("backward", "coarse-backward"):
                # The structural adjoint replaces the parent JIT. Its unpack_rows
                # form calls the rows dot directly; the other forms use the canonical dot.
                fields.update(
                    output_vjp_variant=2 if variant == 5 else 0,
                    segment_rows=4,
                    output_weight_bias_dot=True,
                )
            expected.append((phase, shape, fields))
    except (KeyError, TypeError, ValueError) as exc:
        return f"invalid marker contract: {exc}"
    if expected and (not isinstance(applied, list) or any(not isinstance(r, dict) for r in applied)):
        return "missing compiled launch evidence"
    for phase, shape, fields in expected:
        rows = [r for r in applied if r.get("phase") == phase and tuple(r.get("shape", ())) == shape]
        if not rows or any(not r.get("launches") for r in rows):
            return f"{phase} {shape}: expected {fields}, missing profiled launches"
        for row in rows:
            for launch in row["launches"]:
                if not isinstance(launch, dict):
                    return f"{phase} {shape}: invalid compiled launch evidence"
                found = {k: launch.get(k) for k in fields}
                if found != fields:
                    return f"{phase} {shape}: expected {fields}, observed {found}, grid={launch.get('grid')}"
    return None


def _vjp_applied(records, applied):
    return _dispatch_error(records, applied) is None


def _structural_contract(records):
    contracts = []
    for key, marker in records.get(STRUCT_NAME, {}).items():
        fields = _scope_fields(key)
        shape = fields["shape"]
        op, coarse = shape.get("operation"), shape.get("coarse")
        phase = (
            "backward"
            if (op, coarse) == (1, 0)
            else "coarse-backward"
            if (op, coarse) == (1, 1)
            else "forward"
            if (op, coarse) == (0, 0)
            else None
        )
        if not marker.get("blocks"):
            if marker != {"blocks": {}, "num_warps": 4, "num_stages": 1}:
                raise ValueError("Invalid disabled structural sentinel")
            if phase is not None and fields.get("execution_regime") == "compiled" and shape.get("height") == 56:
                contracts.append(
                    (phase, tuple(shape[n] for n in ("batch", "height", "width", "modes")), 0)
                )
            continue
        variant = marker["blocks"].get("VARIANT")
        valid_op = (
            (phase == "backward" and variant in (1, 2, 3, 4, 5, 6))
            or (phase == "coarse-backward" and variant in (1, 2, 3, 4, 5))
            or (phase == "forward" and variant == 1)
        )
        if (
            fields.get("execution_regime") != "compiled"
            or shape.get("height") != 56
            or not valid_op
            or marker != {"blocks": {"VARIANT": variant}, "num_warps": 4, "num_stages": 1}
        ):
            raise ValueError("Invalid structural scan marker scope/geometry")
        contracts.append(
            (phase, tuple(shape[n] for n in ("batch", "height", "width", "modes")), variant)
        )
    return contracts


def vjp_candidates(observed, defaults, parents):
    """At most two real direct-dot parents per variant, scoped to this microbatch."""
    targets = [key for key in observed.get(VJP_NAME, []) if _direct_scope(key)]
    if not targets:
        return {}
    selected, seen = [], set()
    # Explicitly reassert the existing parent while leaving every other default alone.
    sources = [("default-direct", {})] + sorted(
        parents.items(),
        key=lambda item: ("output-dot-direct" not in item[0], "published" not in item[0], item[0]),
    )
    for label, source in sources:
        plan, usable = copy.deepcopy(source), []
        for key in targets:
            structural = related_geometry(source, STRUCT_NAME, key)
            if structural and structural.get("blocks"):
                continue
            keys = _parent_keys(observed, key)
            if any(k is None for k in keys.values()):
                continue
            values = [
                source.get(name, {}).get(k, defaults.get(name, {}).get(k))
                for name, k in keys.items()
            ]
            if (
                any(g is None for g in values)
                or not all(g == values[0] for g in values)
                or values[0].get("blocks", {}).get("DOUBLING") != 1
            ):
                continue
            for name, parent_key in keys.items():
                plan.setdefault(name, {})[parent_key] = copy.deepcopy(values[0])
            for struct_key in observed.get(STRUCT_NAME, []):
                inherited = defaults.get(STRUCT_NAME, {}).get(struct_key) or {}
                if same_launch_scope(struct_key, key) and inherited.get("blocks"):
                    plan.setdefault(STRUCT_NAME, {})[struct_key] = copy.deepcopy(OFF)
            usable.append(key)
        if not usable:
            continue
        identity = digest(plan)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append((label, plan, usable))
        if len(selected) == 2:
            break
    proposals = {}
    for label, parent, keys in selected:
        for variant in (1, 2):
            plan = copy.deepcopy(parent)
            for key in keys:
                plan.setdefault(VJP_NAME, {})[key] = {
                    "blocks": {"VARIANT": variant},
                    "num_warps": 4,
                    "num_stages": 1,
                }
            _vjp_contract(plan)
            proposals[f"vjp-{variant}-{label}"] = plan
    return proposals


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
    if digest(trial["records"]) != records_digest:
        return False
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
        if records_digest != digest({}) and (
            not _applied(pair.get("applied")) or not _vjp_applied(trial["records"], pair["applied"])
        ):
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
                if digest(trial["records"]) != trial["records_digest"] or not _vjp_applied(
                    trial["records"], screen.get("applied", [])
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
    _vjp_contract(job["records"])
    _structural_contract(job["records"])
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
        names = (*LEGACY_NAMES, VJP_NAME, STRUCT_NAME)
        vjp_bundle = importlib.import_module("lnet.effective_output_vjp").BUNDLE
        struct_bundle = importlib.import_module("lnet.effective_structural_scan").BUNDLE
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
                        VJP_NAME: vjp_bundle,
                        STRUCT_NAME: struct_bundle,
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
            mismatch = _dispatch_error(job["records"], result["applied"])
            if mismatch:
                raise RuntimeError(f"{jobfile.stem}: compiled trial dispatch mismatch: {mismatch}")
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


def _plan_digest(records, defaults):
    """Ignore only explicit overrides exactly equal to observed defaults.

    An off marker over an active default remains an override. Unknown defaults
    also remain explicit, so this never assumes a missing dispatch is disabled.
    """
    changed = {
        name: {
            key: value
            for key, value in table.items()
            if key not in defaults.get(name, {}) or defaults[name][key] != value
        }
        for name, table in records.items()
    }
    return digest({name: table for name, table in changed.items() if table})


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
    proposal_rejections = {}
    proposal_provenance = {}
    for micro, discovery in discoveries.items():
        if micro != baseline:
            portfolio.append((f"micro-{micro}-default", micro, {}))
        original = {name: discovery["observed"].get(name, []) for name in LEGACY_NAMES}
        parent_plans = candidates(original)
        plans = published_candidates(original) if micro == 32 else dict(parent_plans)
        plans.update(vjp_candidates(discovery["observed"], discovery["defaults"], parent_plans))
        rejected = []
        provenance = {}
        plans.update(
            proposal_candidates(discovery["observed"], diagnostics=rejected, provenance=provenance)
        )
        if provenance:
            proposal_provenance[str(micro)] = provenance
        if rejected:
            proposal_rejections[str(micro)] = rejected
            print(f"[effective batch] proposal rejections micro={micro}: {rejected}", flush=True)
        seen_records = set()
        for name, records in plans.items():
            records_key = _plan_digest(records, discovery["defaults"])
            if records_key in seen_records:
                continue
            seen_records.add(records_key)
            if all(
                discovery["defaults"].get(n, {}).get(k) == g
                for n, table in records.items()
                for k, g in table.items()
            ):
                continue
            portfolio.append((f"micro-{micro}-{name}", micro, records))
    promising = []
    for name, micro, records in portfolio:
        entry = dict(
            plan=name, microbatch=micro, records=records, records_digest=digest(records), pairs=[]
        )
        trials.append(entry)
        result, tensors = arm(
            micro,
            records,
            profile=bool(records.get(VJP_NAME)) or bool(records.get(STRUCT_NAME)),
        )
        if result["status"] == "oom":
            entry["status"] = "oom"
            continue
        proof = compare_tensors(refs[micro], tensors)
        entry.update(
            status="screened" if proof["passed"] else "numerical_failure",
            screen=result,
            reference_checks=proof,
        )
        if (records.get(VJP_NAME) or records.get(STRUCT_NAME)) and not _vjp_applied(
            records, result.get("applied", [])
        ):
            entry["status"] = "numerical_failure"
            entry["dispatch_failure"] = "VJP or structural marker was not executed"
            continue
        if proof["passed"] and result["ms"] < best["ms"]:
            promising.append((name, micro, records, entry, result["ms"]))
    # Each approved control opens unseen edges to the remaining finite portfolio.
    # Previously approved geometries never re-enter, so noisy timings cannot cycle.
    compared_edges, approved = set(), {(baseline, digest({}))}
    rankings = {name: elapsed for name, _, _, _, elapsed in promising}
    while True:
        control_id = best["microbatch"], digest(best["records"])
        pending = [
            row
            for row in promising
            if (row[1], digest(row[2])) not in approved
            and (control_id, (row[1], digest(row[2]))) not in compared_edges
        ]
        if not pending:
            break
        name, micro, records, screened, _ = min(pending, key=lambda row: rankings[row[0]])
        candidate_id = micro, digest(records)
        compared_edges.add((control_id, candidate_id))
        entry = dict(
            plan=f"{name}-round-{len(compared_edges)}",
            microbatch=micro,
            records_digest=digest(records),
            records=records,
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
                    m,
                    plan,
                    confirm=True,
                    profile=bool(plan.get(VJP_NAME))
                    or bool(plan.get(STRUCT_NAME))
                    or (side == "candidate" and bool(records)),
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
                or (
                    records
                    and (
                        not _applied(pair["applied"]) or not _vjp_applied(records, pair["applied"])
                    )
                )
            ):
                break
        pairs = entry["pairs"]
        if pairs:
            rankings[name] = statistics.median(p["candidate_ms"] for p in pairs)
        if (
            len(pairs) == 3
            and all(p["numerical_pass"] for p in pairs)
            and (
                not records
                or all(
                    _applied(p.get("applied")) and _vjp_applied(records, p["applied"])
                    for p in pairs
                )
            )
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
            approved.add(candidate_id)
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
        proposal_rejections=proposal_rejections,
        proposal_provenance=proposal_provenance,
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
