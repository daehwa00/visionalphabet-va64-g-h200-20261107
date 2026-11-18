"""Cold calibration of actual optimizer steps. Trial processes never own the live optimizer."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import functools
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import statistics
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import triton
from torch import nn
from torch._inductor import config as inductor
from torch.utils._pytree import tree_map

from lnet.kernel_launch_config import (
    capture_launch_scopes,
    device_key,
    geometry_store_signature,
    workload_geometries,
    stored_geometry,
)
from lnet.path_backward_layout import (
    BUNDLED_LAYOUTS,
    LAYOUT_NAME,
    _driver_api_version,
    _layout_revision,
    candidate_geometry,
)
from lnet.retained_path_dispatch import BUNDLE, PROFILE_NAME, step_geometry, step_revision
from imagenet100_checkpoint_runtime import atomic_json

from lnet.segmented_path_scan import BUNDLE as SEGMENT_BUNDLE, SEGMENT_NAME, segment_revision
from lnet.output_dot_dispatch import BUNDLE as DOT_BUNDLE, DOT_NAME, dot_revision

MODULE = Path(__file__).resolve()
PROJECT = MODULE.parents[1]
PACKAGED = PROJECT / "src/lnet/validated_training_workloads.json"


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cpu_tree(obj):
    return tree_map(
        lambda v: (
            v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        ),
        obj,
    )


def ident(fn):
    module = fn.__module__
    if module == "__main__":
        module = Path(inspect.getfile(fn)).stem
    return [module, fn.__qualname__]


def resolve(spec):
    obj = importlib.import_module(spec[0])
    for name in spec[1].split("."):
        obj = getattr(obj, name)
    return obj


def rng(device):
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def state_hash(obj):
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode())
            h.update(x.tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=repr):
                visit(k)
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode())
            for v in x:
                visit(v)
        else:
            h.update(repr(x).encode())

    visit(obj)
    return h.hexdigest()


def read_json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def receipt_valid(doc, key):
    if not isinstance(doc, dict) or doc.get("key") != key or doc.get("version") != 1:
        return False
    if digest(doc.get("workload")) != key:
        return False
    if doc.get("status") == "baseline":
        return doc.get("records") == {} and isinstance(doc.get("trials"), list)
    if doc.get("status") != "winner" or not isinstance(doc.get("records"), dict):
        return False
    try:
        evidence = next(t for t in doc["trials"] if t["plan"] == doc["winner"])
        plan = doc["records"]
        pairs = doc["winning_pairs"]
        return (
            bool(plan)
            and evidence["records_digest"] == digest(plan)
            and evidence["pairs"] == pairs
            and len(pairs) == 3
            and all(
                p["numerical_pass"]
                and p["checks"]["passed"]
                and all(math.isfinite(p[k]) and p[k] > 0 for k in ("candidate_ms", "control_ms"))
                for p in pairs
            )
            and bool(pairs[0]["applied"])
            and max(p["candidate_ms"] for p in pairs) < min(p["control_ms"] for p in pairs)
        )
    except (KeyError, ValueError, TypeError, StopIteration):
        return False


def published_candidates(observed):
    """Propose published layouts on other hardware, retaining only their geometry."""
    remapped = {}
    for name, filename in (
        (LAYOUT_NAME, BUNDLED_LAYOUTS),
        (PROFILE_NAME, BUNDLE),
        (SEGMENT_NAME, SEGMENT_BUNDLE),
        (DOT_NAME, DOT_BUNDLE),
    ):
        for device, entries in read_json(filename).get("records", {}).items():
            for original, geometry in entries.get(name, {}).items():
                source = json.loads(original)
                old = source["shape"]
                blocks = geometry["blocks"]
                for key in observed.get(name, []):
                    target = json.loads(key)
                    new = target["shape"]
                    if target["dtype"] != source["dtype"] or any(
                        new.get(k) != old.get(k)
                        for k in ("height", "width", "modes", "operation", "coarse")
                    ):
                        continue
                    if name in (PROFILE_NAME, SEGMENT_NAME, DOT_NAME):
                        tile, group = blocks.get("BM", 0), blocks.get("GROUP", 0)
                        checked = step_geometry(
                            new["operation"],
                            group,
                            tile,
                            geometry["num_warps"],
                            blocks.get("DOUBLING", 0),
                        )
                        width = (
                            new["width"] // 2
                            if new["operation"] != 2 and new["coarse"]
                            else new["width"]
                        )
                    else:
                        tile, group = blocks["BLOCK_MODES"], blocks["LINES_PER_GROUP"]
                        checked = candidate_geometry(
                            group, tile, geometry["num_warps"], blocks["COMPACT_CHANNELS"]
                        )
                        width = new["width"]
                    if (
                        new["modes"] % tile
                        or (group and new["batch"] * width % group)
                        or checked.blocks != blocks
                    ):
                        continue
                    label = f"published-{device}-b{old['batch']}"
                    plan = remapped.setdefault(label, {})
                    if name in (SEGMENT_NAME, DOT_NAME):
                        parents = (
                            (PROFILE_NAME, SEGMENT_NAME) if name == DOT_NAME else (PROFILE_NAME,)
                        )
                        keys = [
                            next(
                                (
                                    k
                                    for k in observed.get(parent, [])
                                    if json.loads(k)["shape"] == new
                                ),
                                None,
                            )
                            for parent in parents
                        ]
                        if any(k is None for k in keys):
                            continue
                        for parent, parent_key in zip(parents, keys, strict=True):
                            plan.setdefault(parent, {})[parent_key] = geometry
                    plan.setdefault(name, {})[key] = geometry
    return remapped


def candidates(observed):
    """Portable portfolios. All overrides use scopes captured from this actual training graph."""
    plans = {}
    for label in ("retained-small", "adjoints", "retained-wide", "associative", "order-only"):
        records = {}
        for name, keys in observed.items():
            if name in (SEGMENT_NAME, DOT_NAME):
                continue
            table = {}
            for key in keys:
                s = json.loads(key)
                shape = s["shape"]
                h = shape["height"]
                m = shape["modes"]
                b = shape["batch"]
                w = shape["width"]
                if name == LAYOUT_NAME:
                    tile = (
                        4
                        if h >= 56 and label in ("retained-small", "adjoints", "associative")
                        else 8
                    )
                    warps = 2 if tile == 4 else 4
                    group = 4 if b * w % 4 == 0 else 1
                    geom = candidate_geometry(group, tile, warps, 0 if label == "order-only" else 3)
                else:
                    op = shape["operation"]
                    coarse = shape["coarse"]
                    if label in ("associative", "order-only") or (op == 0 and label == "adjoints"):
                        table[key] = {"blocks": {}, "num_warps": 4, "num_stages": 1}
                        continue
                    tile = 32 if op == 2 else 8 if (op == 0 or label == "retained-wide") else 4
                    group = 1 if op == 2 else 4 if b * (w // 2 if coarse else w) % 4 == 0 else 1
                    warps = 2 if tile == 4 else 4
                    geom = step_geometry(
                        op,
                        group,
                        tile,
                        warps,
                        int(op == 1 and label in ("retained-small", "adjoints")),
                    )
                if m % tile:
                    continue
                table[key] = {
                    "blocks": geom.blocks,
                    "num_warps": geom.num_warps,
                    "num_stages": geom.num_stages,
                }
            if table:
                records[name] = table
        if records and records not in plans.values():
            plans[label] = records
    for label, records in published_candidates(observed).items():
        if records not in plans.values():
            plans[label] = records
    segment_scopes = observed.get(SEGMENT_NAME, [])
    if segment_scopes:
        baseline = {"blocks": {}, "num_warps": 4, "num_stages": 1}
        for plan in plans.values():
            segment_choices = plan.get(SEGMENT_NAME, {})
            plan[SEGMENT_NAME] = {k: segment_choices.get(k, baseline) for k in segment_scopes}
        if "retained-small" in plans:
            four = copy.deepcopy(plans["retained-small"])
            by_shape = {
                json.dumps(json.loads(k)["shape"], sort_keys=True): v
                for k, v in four.get(PROFILE_NAME, {}).items()
            }
            for k in segment_scopes:
                value = by_shape.get(json.dumps(json.loads(k)["shape"], sort_keys=True))
                if value and value["blocks"].get("DOUBLING") == 1:
                    four[SEGMENT_NAME][k] = value
            if four not in plans.values():
                plans["four-row"] = four
    dot_scopes = observed.get(DOT_NAME, [])
    if dot_scopes:
        disabled = {"blocks": {}, "num_warps": 4, "num_stages": 1}
        for plan in plans.values():
            selected = plan.get(DOT_NAME, {})
            plan[DOT_NAME] = {k: selected.get(k, disabled) for k in dot_scopes}
        if segment_scopes and "retained-small" in plans:
            # The dot uses the four-row prefix. Enable both choices on exactly
            # the matching backward scope and test the complete training step.
            for coarse, label in ((0, "output-dot-direct"), (1, "output-dot-coarse")):
                trial = copy.deepcopy(plans["retained-small"])
                geometries = {
                    json.dumps(json.loads(k)["shape"], sort_keys=True): v
                    for k, v in trial.get(PROFILE_NAME, {}).items()
                }
                segments = {
                    json.dumps(json.loads(k)["shape"], sort_keys=True): k for k in segment_scopes
                }
                for k in dot_scopes:
                    shape = json.loads(k)["shape"]
                    key = json.dumps(shape, sort_keys=True)
                    g = geometries.get(key)
                    if (
                        shape["coarse"] == coarse
                        and key in segments
                        and g
                        and g["blocks"].get("DOUBLING") == 1
                    ):
                        trial[SEGMENT_NAME][segments[key]] = g
                        trial[DOT_NAME][k] = g
                if trial not in plans.values():
                    plans[label] = trial
    return plans


def compare_tensors(reference, actual):
    result = {}
    passed = True
    for group in ("outputs", "grads", "updates"):
        a, b = reference[group], actual[group]
        if not a or a.keys() != b.keys():
            return {"passed": False, "keys_failed": group}
        max_abs = max_l2 = 0.0
        fail = []
        for name, x in a.items():
            y = b[name]
            if x.shape != y.shape or not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                fail.append(name)
                continue
            x, y = x.float(), y.float()
            delta = y - x
            absolute = delta.abs().max().item()
            rel = (delta.norm() / x.norm().clamp_min(1e-12)).item()
            limit = {"outputs": 0.05, "updates": 3e-5}.get(group)
            if limit is None:
                limit = 0.03 * x.abs().max().item() + 1e-6
            if absolute > limit or rel > (0.005 if group == "outputs" else 0.02):
                fail.append(name)
            max_abs = max(max_abs, absolute)
            max_l2 = max(max_l2, rel)
        result[group] = dict(count=len(a), max_abs=max_abs, max_relative_l2=max_l2, failed=fail)
        passed &= not fail
    result["passed"] = passed
    return result


def check_compiled_plan(trace, records):
    """Check the actual leaf dispatch in an untimed step before accepting a plan."""
    from tune_path_backward import PHASES, launch_contracts, retained_launches

    actual = []
    specialized = set()
    for key, geometry in records.get(PROFILE_NAME, {}).items():
        fields = json.loads(key)["shape"]
        shape = tuple(fields[k] for k in ("batch", "height", "width", "modes"))
        op = (fields["operation"], fields["coarse"])
        phase = next(name for name, v in PHASES.items() if v == op)
        blocks = geometry["blocks"]
        launches = retained_launches(trace, shape, phase, bool(blocks))
        if blocks:
            expected = (
                blocks["GROUP"],
                blocks["BM"],
                geometry["num_warps"],
                blocks.get("DOUBLING", 0),
            )
            if any(
                (v["group"], v["modes"], v["warps"], v["compact"]) != expected for v in launches
            ):
                raise RuntimeError(f"Compiled {phase} differs from trial plan")
            if op == (1, 0):
                specialized.add(shape)
        actual.append(dict(phase=phase, shape=shape, launches=launches))
    for name in (SEGMENT_NAME, DOT_NAME):
        for key, geometry in records.get(name, {}).items():
            if not geometry["blocks"]:
                continue
            fields = json.loads(key)["shape"]
            shape = tuple(fields[k] for k in ("batch", "height", "width", "modes"))
            phase = "coarse-backward" if fields["coarse"] else "backward"
            matched = next((q for q in actual if q["phase"] == phase and q["shape"] == shape), None)
            if matched is None or not all(
                v.get("segment_rows") == 4 and (name != DOT_NAME or v.get("output_weight_bias_dot"))
                for v in matched["launches"]
            ):
                raise RuntimeError(f"{name} was not executed for {phase} {shape}")
    for key, geometry in records.get(LAYOUT_NAME, {}).items():
        fields = json.loads(key)["shape"]
        shape = tuple(fields[k] for k in ("batch", "height", "width", "modes"))
        if shape in specialized:
            continue
        launches = launch_contracts(trace, shape)
        b = geometry["blocks"]
        expected = (
            b["LINES_PER_GROUP"],
            b["BLOCK_MODES"],
            geometry["num_warps"],
            b["COMPACT_CHANNELS"],
        )
        if any((v["group"], v["modes"], v["warps"], v["compact"]) != expected for v in launches):
            raise RuntimeError("Compiled path layout differs from trial plan")
        actual.append(dict(phase="legacy", shape=shape, launches=launches))
    return actual


def worker(jobfile):
    from va_imagenet100_runtime import _build_runtime, training_objective
    import imagenet100_data_runtime as data

    job = read_json(jobfile)
    out = jobfile.parent
    snapshot = torch.load(out / "snapshot.pt", map_location="cpu", weights_only=False)
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
    f = snapshot["factories"]
    recipe = snapshot["recipe"]
    model = resolve(f["build"])(
        snapshot["variant"], resolve(f["config"])(**snapshot["model_config"])
    ).to(device)
    model = resolve(f["prepare"])(model, recipe)
    model.train()
    opt = resolve(f["optimizer"])(model, recipe)
    group = [
        (x.to(device), y.to(device), z.to(device), lam, sample_rng)
        for x, y, z, lam, sample_rng in snapshot["batches"]
    ]
    params = {name: v for name, v in model.named_parameters() if v.requires_grad}

    def reset():
        model.load_state_dict(snapshot["model"], strict=True)
        opt.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        opt.zero_grad(set_to_none=True)
        restore_rng(group[0][4], device)

    def step(capture=False):
        opt.zero_grad(set_to_none=True)
        outputs = {}
        for i, (x, y, z, lam, sample_rng) in enumerate(group):
            if capture:
                restore_rng(sample_rng, device)
            data.begin_cudagraph_step(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=recipe["precision"] == "bfloat16"
            ):
                values = runtime(x)
                _, loss, _ = training_objective(model, values, y, z, lam)
            (loss / len(group)).backward()
            if capture:
                for j, v in enumerate(values):
                    outputs[f"{i}/{j}"] = v.detach().cpu().clone()
        if capture:
            missing = [k for k, v in params.items() if v.grad is None]
            if missing:
                raise RuntimeError(f"Missing gradients: {missing}")
            grads = {k: v.grad.detach().cpu().clone() for k, v in params.items()}
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if capture:
            checks = dict(
                outputs=outputs,
                grads=grads,
                updates={k: v.detach().cpu() - snapshot["model"][k] for k, v in params.items()},
            )
            if not all(
                torch.isfinite(t).all() for values in checks.values() for t in values.values()
            ):
                raise RuntimeError("Nonfinite training-step tensors")
            return checks

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            workload_geometries(job["records"], discovery=str(out) if job["discover"] else "")
        )
        found = {
            n: stack.enter_context(capture_launch_scopes(n))
            for n in (LAYOUT_NAME, PROFILE_NAME, SEGMENT_NAME, DOT_NAME)
        }
        reset()
        runtime = _build_runtime(model, recipe)
        checks = step(True)
        torch.save(checks, jobfile.with_suffix(".pt"))
        regime = (
            "compiled"
            if os.environ.get("LNET_COMPILE_MODE") or recipe.get("compile_mode")
            else "eager"
        )
        observed = {
            n: sorted(s.key for s in ss if s.execution_regime == regime) for n, ss in found.items()
        }
        defaults = {}
        if job["discover"]:
            for n, scopes in found.items():
                default = {
                    LAYOUT_NAME: BUNDLED_LAYOUTS,
                    PROFILE_NAME: BUNDLE,
                    SEGMENT_NAME: SEGMENT_BUNDLE,
                    DOT_NAME: DOT_BUNDLE,
                }[n]
                defaults[n] = {}
                for scope in scopes:
                    if scope.key not in observed[n]:
                        continue
                    geom = stored_geometry(n, scope=scope) or stored_geometry(
                        n, scope=scope, path=default
                    )
                    defaults[n][scope.key] = (
                        dict(
                            blocks=geom.blocks, num_warps=geom.num_warps, num_stages=geom.num_stages
                        )
                        if geom
                        else dict(blocks={}, num_warps=4, num_stages=1)
                    )
        result = {"observed": observed, "defaults": defaults, "count": len(params)}
        if not job["discover"]:
            # Restore model buffers, moments and RNG outside each serialized training step.
            for _ in range(5):
                reset()
                step()
            samples = []
            peaks = []
            for _ in range(5):
                wall = []
                for _ in range(10):
                    reset()
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    t = time.perf_counter()
                    step()
                    torch.cuda.synchronize(device)
                    wall.append((time.perf_counter() - t) * 1000)
                    peaks.append(torch.cuda.max_memory_allocated(device) / 2**20)
                samples.append(statistics.mean(wall))
            result.update(ms=statistics.median(samples), groups_ms=samples, peak_mib=max(peaks))
        if job.get("profile"):
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


def measure(snapshot, key, info, work):
    torch.save(snapshot, work / "snapshot.pt")

    def arm(label, records, discover=False):
        job = work / (label + ".job")
        atomic_json(
            job,
            {"records": records, "discover": discover, "profile": label.endswith("-0-candidate")},
        )
        env = dict(os.environ)
        for k in ("VA_IN100_PROGRESS_PATH", "WANDB_PROJECT"):
            env.pop(k, None)
        env["WANDB_MODE"] = "disabled"
        env["PYTHONPATH"] = os.pathsep.join(
            (str(MODULE.parent), str(PROJECT / "src"), env.get("PYTHONPATH", ""))
        )
        with job.with_suffix(".log").open("w") as log:
            completed = subprocess.run(
                [sys.executable, str(MODULE), "--worker", str(job)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode:
            raise RuntimeError(f"{label}: " + job.with_suffix(".log").read_text()[-1400:])
        return read_json(job.with_suffix(".result.json")), job.with_suffix(".pt")

    discovery, ref_file = arm("discovery", {}, True)
    reference = torch.load(ref_file, weights_only=True)
    trials = []
    best = None
    score = 1.0
    for name, plan in candidates(discovery["observed"]).items():
        if all(
            discovery["defaults"].get(n, {}).get(k) == g
            for n, table in plan.items()
            for k, g in table.items()
        ):
            continue
        print(f"[training layout] comparing {name}", flush=True)
        entry = {"plan": name, "pairs": [], "records_digest": digest(plan)}
        trials.append(entry)
        try:
            for i in range(3):
                order = ["candidate", "control"] if i == 1 else ["control", "candidate"]
                pair = {"order": order, "pair": i + 1}
                tensors = {}
                for side in order:
                    try:
                        result, path = arm(
                            f"{name}-{i}-{side}", plan if side == "candidate" else {}
                        )
                    except RuntimeError as exc:
                        if side == "control":
                            raise ValueError("Control execution failed") from exc
                        raise
                    if "applied" in result:
                        pair["applied"] = result["applied"]
                    tensors[side] = torch.load(path, weights_only=True)
                    pair[side + "_ms"] = result["ms"]
                    pair[side + "_peak_mib"] = result["peak_mib"]
                    independent = compare_tensors(reference, tensors[side])
                    if not independent["passed"]:
                        if side == "control":
                            raise ValueError(f"Unstable control: {independent}")
                        entry["numerical_failure"] = independent
                        break
                if "numerical_failure" in entry:
                    break
                detail = compare_tensors(tensors["control"], tensors["candidate"])
                pair.update(checks=detail, numerical_pass=detail["passed"])
                entry["pairs"].append(pair)
                if not detail["passed"]:
                    break
        except RuntimeError as exc:
            entry["execution_error"] = str(exc)
            continue
        rows = entry["pairs"]
        if (
            len(rows) == 3
            and rows[0].get("applied")
            and all(p["numerical_pass"] for p in rows)
            and max(p["candidate_ms"] for p in rows) < min(p["control_ms"] for p in rows)
        ):
            gain = statistics.median(p["control_ms"] / p["candidate_ms"] for p in rows)
            if gain > score:
                best = (name, plan, rows)
                score = gain
    # An unavailable compiler/device is retryable, not evidence that baseline won.
    if trials and all("execution_error" in t for t in trials):
        raise RuntimeError(str(trials[0]["execution_error"]))
    return dict(
        version=1,
        key=key,
        workload=info,
        status="winner" if best else "baseline",
        winner=best[0] if best else None,
        records=best[1] if best else {},
        winning_pairs=best[2] if best else [],
        trials=trials,
        observed=discovery["observed"],
    )


class AutomaticTrainingRuntime(nn.Module):
    """Use actual, already-mixed microbatches from one ordinary optimizer group on a cache miss."""

    def __init__(
        self,
        model,
        optimizer,
        recipe,
        model_config,
        variant,
        bindings,
        root,
        training_generator,
        mixup_generator,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.recipe = recipe
        self.model_config = model_config
        self.variant = variant
        self.training_generator = training_generator
        self.mixup_generator = mixup_generator
        self.factories = {
            k: ident(v)
            for k, v in dict(
                build=bindings.build_model,
                config=bindings.model_config,
                optimizer=bindings.build_optimizer,
                prepare=bindings.prepare_model,
            ).items()
        }
        self.base = (
            Path(os.environ.get("LNET_LAUNCH_CACHE", Path.home() / ".cache/lnet"))
            / "training-workloads"
        )
        self.audit = root / "layout-calibration"
        self.decisions = {}
        self.runtimes = {}
        self.batches = []
        self.current = None
        self.epoch_index = 0
        self.keys = {}
        # Read the schema once. Do not tie a reusable performance decision to weight values or epoch LR.
        names = {id(v): n for n, v in model.named_parameters()}
        self.schema = {
            "model": [(n, list(v.shape), str(v.dtype)) for n, v in model.state_dict().items()],
            "optimizer": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
            "groups": [
                {
                    "names": [names[id(v)] for v in g["params"]],
                    "options": {
                        k: g.get(k)
                        for k in (
                            "fused",
                            "foreach",
                            "capturable",
                            "amsgrad",
                            "maximize",
                            "differentiable",
                            "betas",
                            "eps",
                            "weight_decay",
                        )
                    },
                }
                for g in optimizer.param_groups
            ],
            "factories": self.factories,
            "model_config": model_config,
            "variant": variant,
        }
        self.settings = dict(
            threads=torch.get_num_threads(),
            matmul=torch.get_float32_matmul_precision(),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            emulate_precision_casts=inductor.emulate_precision_casts,
        )
        revision = hashlib.sha256()
        for f in (
            MODULE,
            MODULE.with_name("va_imagenet100_runtime.py"),
            MODULE.with_name("va_ablation_augment.py"),
            MODULE.with_name("va_model_builder.py"),
            MODULE.with_name("va_packed_backbone.py"),
            *(sorted((PROJECT / "src/lnet").glob("*.py"))),
            *(
                Path(inspect.getfile(v))
                for v in (bindings.build_model, bindings.build_optimizer, bindings.prepare_model)
            ),
        ):
            revision.update(f.read_bytes())
        train_fn = bindings.train_epoch
        train_kwargs = train_fn.keywords if isinstance(train_fn, functools.partial) else {}
        train_fn = train_fn.func if isinstance(train_fn, functools.partial) else train_fn
        self.identity = dict(
            train_function=ident(train_fn),
            train_options=train_kwargs,
            objective=recipe.get("head_objective"),
            device=device_key(),
            driver_api=_driver_api_version(),
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            model=digest(self.schema),
            training_code=revision.hexdigest(),
            legacy_revision=_layout_revision(),
            step_revisions=[step_revision(i) for i in range(3)],
            segment_revision=segment_revision(step_revision(1)),
            output_dot_revision=dot_revision(step_revision(1)),
            output_dot_records=geometry_store_signature(DOT_NAME, DOT_BUNDLE, omit_empty=True),
            profile=os.environ.get("LNET_COMPILE_PROFILE"),
            precision=recipe["precision"],
            channels_last=recipe["channels_last"],
            settings=self.settings,
            legacy_records=geometry_store_signature(LAYOUT_NAME, BUNDLED_LAYOUTS),
            retained_records=geometry_store_signature(PROFILE_NAME, BUNDLE),
            segment_records=geometry_store_signature(SEGMENT_NAME, SEGMENT_BUNDLE, omit_empty=True),
            grad_accum=recipe["gradient_accumulation_steps"],
            cudnn_version=torch.backends.cudnn.version(),
            cuda_arch=torch.cuda.get_device_capability(),
            cuda_bf16_reduced=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            cuda_fp16_reduced=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            mixup_alpha=recipe["mixup_alpha"],
            scan_env={
                k: os.environ.get(k)
                for k in (
                    "VA_REAL_MEMORY_SCAN_BATCH_PARTIALS",
                    "VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS",
                    "LNET_DISABLE_LAUNCH_AUTOTUNE",
                )
            },
        )

    def runtime_for(self, key, records):
        from va_imagenet100_runtime import _build_runtime

        tag = (key, digest(records))
        if tag not in self.runtimes:
            self.runtimes[tag] = _build_runtime(self.model, self.recipe)
            if key != "evaluation":
                atomic_json(
                    self.audit / (key + ".active.json"),
                    {
                        "workload": key,
                        "records_digest": tag[1],
                        "overrides": sum(len(v) for v in records.values()),
                        "legacy_graph_records": geometry_store_signature(
                            LAYOUT_NAME, BUNDLED_LAYOUTS
                        ),
                        "retained_graph_records": geometry_store_signature(PROFILE_NAME, BUNDLE),
                    },
                )
        return self.runtimes[tag]

    def forward(self, x):
        # Validation uses the existing inference path. Its shapes and objective have no training receipt.
        return self.runtime_for("evaluation", {})(x)

    def event(self, key, status, **details):
        print(f"[training layout] {status} {key[:12]}", flush=True)
        atomic_json(self.audit / (key + ".json"), dict(key=key, status=status, **details))

    def initial(self, key):
        for path, label in ((self.base / (key + ".json"), "local"), (PACKAGED, "bundled")):
            d = read_json(path)
            if label == "bundled":
                d = d.get(key, {})
            if receipt_valid(d, key):
                self.event(
                    key, "reused", source=label, decision=d["status"], winner=d.get("winner")
                )
                return d
        return None

    def live_state(self, device):
        return dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            gradients={n: v.grad for n, v in self.model.named_parameters()},
            rng=rng(device),
            train_generator=self.training_generator.get_state(),
            mixup=copy.deepcopy(self.mixup_generator.bit_generator.state),
        )

    def finish(self, key, info, device):
        cache = self.base / (key + ".json")
        try:
            self.base.mkdir(parents=True, exist_ok=True)
            # Competing first users share one decision; workers never write production records.
            gpu_id = str(getattr(torch.cuda.get_device_properties(device), "uuid", device.index))
            device_lock = self.base / ("gpu-" + digest((socket.gethostname(), gpu_id)) + ".lock")
            with (self.base / (key + ".lock")).open("a+") as lock, device_lock.open("a+") as card:
                fcntl.flock(lock, fcntl.LOCK_EX)
                fcntl.flock(card, fcntl.LOCK_EX)
                other = self.initial(key)
                if other is not None:
                    return other
                self.event(key, "calibrating", workload=info)
                live = self.live_state(device)
                state_before = {k: state_hash(v) for k, v in live.items()}
                before = digest(state_before)
                t = time.perf_counter()
                snapshot = dict(
                    variant=self.variant,
                    model_config=self.model_config,
                    factories=self.factories,
                    recipe=self.recipe,
                    settings=self.settings,
                    model=cpu_tree(live["model"]),
                    optimizer=cpu_tree(live["optimizer"]),
                    rng=cpu_tree(live["rng"]),
                    batches=self.batches,
                    index=device.index if device.index is not None else torch.cuda.current_device(),
                )
                # Return unused training allocations to the device before a separate CUDA process starts.
                with torch.cuda.device(device):
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
                with tempfile.TemporaryDirectory(prefix=".layout-", dir=self.base) as tmp:
                    try:
                        decision = measure(snapshot, key, info, Path(tmp))
                    finally:
                        state_after = {k: state_hash(v) for k, v in self.live_state(device).items()}
                        after = digest(state_after)
                        if before != after:
                            raise AssertionError(
                                "Live training state changed during isolated calibration"
                            )
                decision.update(
                    live_before=before,
                    live_after=after,
                    state_before=state_before,
                    state_after=state_after,
                    elapsed_s=time.perf_counter() - t,
                )
                decision["fixed_inputs_digest"] = state_hash(self.batches)
                decision["optimizer_tensor_count"] = len(snapshot["optimizer"]["state"])
                if not receipt_valid(decision, key):
                    raise RuntimeError("Incomplete calibration evidence")
                atomic_json(cache, decision)
                self.event(
                    key,
                    "calibrated",
                    decision=decision["status"],
                    winner=decision.get("winner"),
                    live_before=before,
                    live_after=after,
                    receipt=str(cache),
                    elapsed_s=decision["elapsed_s"],
                )
                return decision
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.warn(
                f"Training-layout calibration unavailable; using baseline for this run: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self.event(key, "fallback", error=str(exc))
            return {"records": {}}

    @contextlib.contextmanager
    def microbatch(self, x, y, other, mixing, group_size, offset):
        if offset == 0:
            info = {
                **self.identity,
                "input_shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "input_stride": list(x.stride()),
                "microbatches": group_size,
                "schedule_phase": "warmup"
                if self.epoch_index < self.recipe.get("warmup_epochs", 0)
                else "scheduled",
                "zero_lr_groups": tuple(g["lr"] == 0 for g in self.optimizer.param_groups),
            }
            shape_key = (
                tuple(x.shape),
                str(x.dtype),
                tuple(x.stride()),
                group_size,
                info["schedule_phase"],
                info["zero_lr_groups"],
            )
            if shape_key not in self.keys:
                self.keys[shape_key] = digest(info)
            key = self.keys[shape_key]
            self.current = (key, info)
            if key not in self.decisions:
                self.decisions[key] = self.initial(key)
            self.batches = []
        key, info = self.current
        decision = self.decisions[key]
        if decision is None:
            self.batches.append(
                (x.detach().cpu(), y.detach().cpu(), other.detach().cpu(), mixing, rng(x.device))
            )
        records = decision["records"] if decision else {}
        with workload_geometries(records):
            yield self.runtime_for(key, records)
        if offset + 1 == group_size and decision is None:
            try:
                self.decisions[key] = self.finish(key, info, x.device)
            finally:
                self.batches = []


def training_microbatch(runtime, x, y, other, mixing, size, offset):
    if isinstance(runtime, AutomaticTrainingRuntime):
        return runtime.microbatch(x, y, other, mixing, size, offset)
    return contextlib.nullcontext(runtime)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This worker is invoked by the training entry point.")
    worker(Path(sys.argv[2]))
