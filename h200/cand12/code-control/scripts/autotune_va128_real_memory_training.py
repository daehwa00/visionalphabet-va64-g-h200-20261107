"""Measure full-model training plans and save an explicit, workload-guarded choice.

Selection never changes global settings or launches real training. Numerical
comparisons use a baseline with the same scan policy. A plan is reusable only
when its identity (including source/checkpoint/software/input/batch) matches.
Use --cpu-affinity with an available CPU list to stabilize host scheduling;
explicit pinning is recommended when CPU cores have different performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

# Standalone experiment orchestration follows the benchmark script conventions.
# ruff: noqa: ANN001, ANN201

IDENTITY_FIELDS = (
    "checkpoint_sha256", "variant", "epoch", "parameters", "batch", "input_sha256",
    "precision", "tf32", "memory_format", "optimizer", "learning_rates", "torch",
    "cuda", "cudnn", "triton", "gpu", "mode", "deterministic", "reset_each_step",
    "cudnn_benchmark", "pythonhashseed", "cublas_workspace_config", "emulate_precision_casts",
    "gradient_clip_norm", "scope", "warmup", "steps_per_group", "repeats", "cpu_affinity",
    "torch_num_threads",
)


def select_plan(trials, identity, repetitions=3, improvement=0.02):
    """Pure selection over recorded trials; invalid/mismatched plans cannot win."""
    if repetitions < 1 or not 0 <= improvement < 1:
        raise ValueError("Invalid selection counts or improvement")
    measured = None
    legacy = identity.get("diagnostic_legacy_records") is True
    summaries = []
    for policy in ("default", "autotune"):
        baseline_valid = False
        baseline_trials = []
        for reader in (False, True):
            selected = [t for t in trials if t["policy"] == policy and t["reader"] == reader]
            valid = len(selected) == repetitions and {t["repeat"] for t in selected} == set(
                range(1, repetitions + 1)
            )
            for trial in selected:
                result = trial.get("result", {})
                trial_valid = (
                    trial["identity"] == identity
                    and trial["returncode"] == 0
                    and result.get("status") == "passed"
                    and result.get("validation_finite") is True
                    and result.get("final_finite") is True
                    and result.get("missing_gradients") == []
                    and result.get("deterministic") is True
                    and result.get("reset_each_step") is True
                    and all(result.get(key) == identity[key]
                            for key in ("checkpoint_sha256", "batch", "precision", "cpu_affinity")
                            if key in identity and not (legacy and key not in result))
                    and ("cpu_threads" not in identity
                         or result.get("torch_num_threads") == identity["cpu_threads"]
                         or (legacy and "torch_num_threads" not in result))
                    and not result.get("norm_candidate", False)
                    and not result.get("dense_reader", False)
                    and result.get("spatial") == reader
                    and result.get("vector") == reader
                    and result.get("scan_autotune_disabled") == (policy == "default")
                    and math.isfinite(result.get("median_ms", math.inf))
                    and result.get("median_ms", 0) > 0
                )
                fields = [key for key in IDENTITY_FIELDS
                          if not (legacy and key in ("cpu_affinity", "torch_num_threads")
                                  and key not in result)]
                if all(key in result for key in fields):
                    current = {key: result[key] for key in fields}
                else:
                    trial_valid = False
                if reader or trial["repeat"] > 1:
                    checks = result.get("gradient_checks", {})
                    trial_valid &= (
                        result.get("output_check", {}).get("passed") is True
                        and set(checks) == {"grads", "updates"}
                        and all(checks.values())
                        and all(
                            item.get("passed") is True
                            for group in checks.values()
                            for item in group.values()
                        )
                    )
                trial_valid &= baseline_valid if reader else True
                if trial_valid:
                    measured = current if measured is None else measured
                    trial_valid = current == measured
                valid &= trial_valid
            paired_speedups = []
            if reader:
                valid &= baseline_valid
                if valid:
                    baseline_by_repeat = {t["repeat"]: t["result"]["median_ms"]
                                          for t in baseline_trials}
                    for trial in sorted(selected, key=lambda t: t["repeat"]):
                        baseline_ms = baseline_by_repeat[trial["repeat"]]
                        reader_ms = trial["result"]["median_ms"]
                        paired_speedups.append({
                            "repeat": trial["repeat"], "baseline_ms": baseline_ms,
                            "reader_ms": reader_ms, "speedup": baseline_ms / reader_ms,
                            "improvement_fraction": 1 - reader_ms / baseline_ms,
                            "confirmed_improvement": reader_ms <= baseline_ms * (1 - improvement),
                        })
            else:
                baseline_valid = valid
                baseline_trials = selected
            confirmed = bool(reader and valid and paired_speedups
                             and all(pair["confirmed_improvement"] for pair in paired_speedups))
            summaries.append(
                {
                    "policy": policy,
                    "reader": reader,
                    "validated": bool(valid),
                    "paired_speedups": paired_speedups,
                    "confirmed_improvement": confirmed,
                    "median_ms": statistics.median(t["result"]["median_ms"] for t in selected)
                    if valid
                    else None,
                    "trials": selected,
                }
            )
    baselines = [s for s in summaries if s["validated"] and not s["reader"]]
    if not baselines:
        raise ValueError("No fully validated baseline plan")
    baseline = min(baselines, key=lambda s: s["median_ms"])
    eligible = [
        s
        for s in summaries
        if s["validated"]
        and s["reader"]
        and s["confirmed_improvement"]
        and s["median_ms"] <= baseline["median_ms"] * (1 - improvement)
    ]
    winner = min(eligible, key=lambda s: s["median_ms"]) if eligible else baseline
    matched = next(s for s in baselines if s["policy"] == winner["policy"])
    return {
        "identity": {**identity, "measured": measured},
        "selected": {
            "policy": winner["policy"],
            "reader": winner["reader"],
            "median_ms": winner["median_ms"],
            "baseline_median_ms": matched["median_ms"],
            "fastest_baseline_median_ms": baseline["median_ms"],
            "minimum_improvement": improvement,
            "independent_repetitions": repetitions,
            "paired_speedups": winner["paired_speedups"],
            "confirmed_improvement": winner["confirmed_improvement"],
            "cpu_affinity": identity.get("cpu_affinity"),
            "cpu_threads": identity.get("cpu_threads"),
            "diagnostic_legacy_records": legacy,
            "environment": {
                "VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS": str(int(winner["reader"])),
                "LNET_DISABLE_LAUNCH_AUTOTUNE": str(int(winner["policy"] == "default")),
            },
        },
        "plans": summaries,
        "reuse_guard": "Require exact identity match; numerical parity is matched-policy only.",
    }


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_manifest(repo):
    return {str(path.relative_to(repo)): sha256(path)
            for folder in (repo / "src", repo / "scripts")
            for path in sorted(folder.rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "contract", "images", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-affinity", help="Comma-separated available CPU IDs; default: inherit")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output in [getattr(args, key).resolve() for key in ("checkpoint", "contract", "images")]:
        parser.error("Output plan must differ from checkpoint, contract and images")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    if min(args.batch, args.repetitions, args.warmup, args.repeats, args.steps, args.cpu_threads) < 1:
        parser.error("Batch and measurement counts must be positive")
    available = sorted(os.sched_getaffinity(0))
    affinity = available
    if args.cpu_affinity is not None:
        try:
            affinity = sorted({int(value) for value in args.cpu_affinity.split(",")})
        except ValueError:
            parser.error("--cpu-affinity requires a comma-separated list of CPU IDs")
        if not affinity or not set(affinity) <= set(available):
            parser.error("--cpu-affinity must select available CPU IDs")
    repo = Path(__file__).resolve().parents[1]
    sources = source_manifest(repo)
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    identity = {
        "source_sha256": source_hash,
        "python": sys.version,
        "executable": sys.executable,
        "batch": args.batch,
        "cpu_threads": args.cpu_threads,
        "cpu_affinity": affinity,
        "requested_cpu_affinity": args.cpu_affinity,
        "shape": [args.batch, 3, 224, 224],
        "precision": "bf16 autocast",
        **{
            f"{name}_sha256": sha256(getattr(args, name))
            for name in ("checkpoint", "contract", "images")
        },
        "inherited_environment": {
            key: value for key, value in os.environ.items()
            if key.startswith(("LNET_", "VA_", "TORCH", "TRITON", "CUDA",
                               "NVIDIA", "CUBLAS", "CUDNN", "OMP_", "MKL_", "OPENBLAS_", "KMP_"))
        },
    }
    work = Path(tempfile.mkdtemp(
        prefix=f"autotune-{source_hash}-b{args.batch}-", dir=args.output.parent))
    cache_root = args.output.parent / f"autotune-cache-{source_hash}"
    trials = []
    for repeat in range(1, args.repetitions + 1):
        for policy in ("default", "autotune") if repeat % 2 else ("autotune", "default"):
            for reader in (False, True):
                name = f"r{repeat}-{policy}-{'reader' if reader else 'baseline'}"
                output = work / f"{name}.json"
                reference = work / (
                    f"r{repeat}-{policy}-baseline.json" if reader else f"r1-{policy}-baseline.json"
                )
                benchmark = repo / "scripts/benchmark_va128_real_memory_training.py"
                command = [sys.executable, str(benchmark)]
                if args.cpu_affinity is not None:
                    command = ["taskset", "-c", ",".join(map(str, affinity)), *command]
                for option in ("checkpoint", "contract", "images"):
                    command += [f"--{option}", str(getattr(args, option).resolve())]
                command += ["--output", str(output), "--deterministic", "--reset-each-step"]
                for option in ("batch", "warmup", "repeats", "steps", "cpu_threads"):
                    command += ["--" + option.replace("_", "-"), str(getattr(args, option))]
                if reader:
                    command += ["--spatial", "--vector"]
                if reader or repeat > 1:
                    command += ["--reference", str(reference)]
                cache = cache_root / policy
                env = {
                    **os.environ,
                    "PYTHONPATH": f"{repo / 'src'}:{repo / 'scripts'}",
                    "VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS": "0",
                    "PYTHONHASHSEED": "0",
                    "LNET_DISABLE_LAUNCH_AUTOTUNE": str(int(policy == "default")),
                    "TORCHINDUCTOR_FREEZING": "0",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
                    "TRITON_CACHE_DIR": str(cache / "triton"),
                    "CUDA_CACHE_PATH": str(cache / "cuda"),
                }
                if source_manifest(repo) != sources:
                    raise RuntimeError("Source changed during tuning; start a new source namespace")
                active = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                ).strip()
                if active:
                    raise RuntimeError(f"GPU occupied; no process was stopped: {active}")
                print(f"Running {name}", flush=True)
                with output.with_suffix(".log").open("w") as log:
                    child = subprocess.run(
                        command,
                        cwd=repo,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                trials.append(
                    {
                        "policy": policy,
                        "reader": reader,
                        "repeat": repeat,
                        "identity": identity,
                        "returncode": child.returncode,
                        "path": str(output),
                        "command": command,
                        "cpu_affinity": affinity,
                        "environment": {key: env[key] for key in (
                            "PYTHONPATH", "VA_REAL_MEMORY_TRAINING_OPTIMIZATIONS",
                            "PYTHONHASHSEED", "LNET_DISABLE_LAUNCH_AUTOTUNE",
                            "TORCHINDUCTOR_FREEZING", "CUBLAS_WORKSPACE_CONFIG",
                            "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                            "CUDA_CACHE_PATH")},
                        "result": json.loads(output.read_text()) if output.exists() else {},
                    }
                )
                (work / "trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    plan = select_plan(trials, identity, args.repetitions)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan["selected"], indent=2))


if __name__ == "__main__":
    main()
