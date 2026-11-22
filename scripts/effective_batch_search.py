"""Calibrate activation microbatches in disposable training processes."""

from __future__ import annotations

import copy
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from automatic_training_layout import (
    PROJECT,
    compare_tensors,
    digest,
    read_json,
    resolve,
    restore_rng,
)
from imagenet100_checkpoint_runtime import atomic_json

if TYPE_CHECKING:
    from automatic_training_layout import TensorCapture

MODULE = Path(__file__).resolve()
VERSION = 1
PAIR_ORDERS = (("control", "candidate"), ("candidate", "control"), ("control", "candidate"))


def candidate_microbatches(effective_samples: int) -> tuple[int, ...]:
    if effective_samples < 1:
        raise ValueError("Effective sample count must be positive")
    minimum = min(32, effective_samples)
    if effective_samples < 32 and effective_samples & (effective_samples - 1):
        minimum = max(1, effective_samples // 4)
    return (
        effective_samples,
        *tuple(
            1 << power
            for power in range((effective_samples - 1).bit_length() - 1, -1, -1)
            if 1 << power >= minimum
        ),
    )


def baseline_microbatches(effective_samples: int) -> tuple[int, ...]:
    return tuple(size for size in candidate_microbatches(effective_samples) if size <= 128)


def _is_cuda_oom(exc: BaseException) -> bool:
    """Identify typed CUDA OOMs, including ones wrapped by compilation."""
    pending, visited = [exc], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, torch.cuda.OutOfMemoryError):
            return True
        for nested in (
            current.__cause__,
            current.__context__,
            getattr(current, "inner_exception", None),
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def worker(jobfile: Path) -> None:
    """Run one benchmark arm and write its result, including typed OOMs."""
    job = read_json(jobfile)
    try:
        result, checks = _benchmark(jobfile.parent, job["microbatch"])
    except Exception as exc:
        if _is_cuda_oom(exc):
            result = {"status": "oom"}
        elif isinstance(exc, FloatingPointError):
            result = {"status": "nonfinite"}
        else:
            raise
    else:
        torch.save(checks, jobfile.with_suffix(".pt"))
    atomic_json(jobfile.with_suffix(".result.json"), result)


def _benchmark(work: Path, microbatch: int) -> tuple[dict[str, Any], TensorCapture]:
    from effective_batch_update import run_update
    from va_imagenet100_runtime import _build_runtime

    snapshot = torch.load(work / "snapshot.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda", snapshot["index"])
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(snapshot["memory_fraction"], device)
    cfg = snapshot["settings"]
    torch.set_num_threads(cfg["threads"])
    torch.set_float32_matmul_precision(cfg["matmul"])
    torch.backends.cudnn.benchmark = cfg["cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = cfg["cudnn_tf32"]
    torch.backends.cudnn.deterministic = cfg["cudnn_deterministic"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = cfg["cuda_bf16_reduced"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = cfg["cuda_fp16_reduced"]
    torch.use_deterministic_algorithms(cfg["deterministic"], warn_only=cfg["warn_only"])
    factories = {key: resolve(spec) for key, spec in snapshot["factories"].items()}
    recipe = snapshot["recipe"]
    model = factories["build"](
        snapshot["variant"], factories["config"](**snapshot["model_config"])
    ).to(device)
    model = factories["prepare"](model, recipe)
    model.train()
    optimizer = factories["optimizer"](model, recipe)
    batch = tuple(v.to(device) if isinstance(v, torch.Tensor) else v for v in snapshot["batch"])
    runtime = _build_runtime(model, recipe)

    def step(*, capture: bool = False) -> dict[str, Any]:
        return run_update(
            model,
            runtime,
            optimizer,
            batch,
            microbatch,
            recipe["precision"],
            capture=capture,
            metrics=True,
        )

    def reset() -> None:
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        optimizer.zero_grad(set_to_none=True)
        restore_rng(snapshot["rng"], device)

    reset()
    captured = step(capture=True)
    if not torch.isfinite(captured["loss"]) or not torch.isfinite(captured["clip_norm"]):
        raise FloatingPointError("Nonfinite loss or gradient norm")
    for _ in range(2):
        reset()
        step()

    samples = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        reset()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        samples.append(1000 * (time.perf_counter() - start))
    ms = statistics.median(samples)
    return (
        {
            "status": "ok",
            "ms": ms,
            "images_per_second": len(batch[1]) * 1000 / ms,
            "peak_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        },
        captured["checks"],
    )


def _run_arm(
    work: Path, label: str, microbatch: int
) -> tuple[dict[str, Any], TensorCapture | None]:
    job = work / (label + ".job")
    atomic_json(job, {"microbatch": microbatch})
    env = dict(os.environ, WANDB_MODE="disabled")
    env.pop("VA_IN100_PROGRESS_PATH", None)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(MODULE.parent), str(PROJECT / "src"), env.get("PYTHONPATH", ""))
    )
    log_path = job.with_suffix(".log")
    with log_path.open("w") as log:
        child = subprocess.run(
            [sys.executable, str(MODULE), "--worker", str(job)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=1200,
        )
    result = read_json(job.with_suffix(".result.json"))
    if child.returncode or result.get("status") not in {"ok", "oom", "nonfinite"}:
        tail = log_path.read_text(errors="replace")[-1400:]
        raise RuntimeError(f"{label}: calibration worker failed: {tail}")
    if result["status"] != "ok":
        return result, None
    path = job.with_suffix(".pt")
    checks = torch.load(path, weights_only=True)
    path.unlink()
    return result, checks


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _passes(proof: dict[str, Any]) -> bool:
    return proof.get("passed") is True and all(
        type(proof[group]["count"]) is int
        and proof[group]["count"] > 0
        and proof[group]["failed"] == []
        for group in ("outputs", "grads", "updates")
    )


def _confirmed(pairs: list[dict[str, Any]]) -> bool:
    return (
        len(pairs) == len(PAIR_ORDERS)
        and all(
            tuple(pair["order"]) == order
            and all(
                pair[role]["status"] == "ok"
                and _positive(pair[role]["ms"])
                and _passes(pair[role]["parity"])
                for role in order
            )
            for pair, order in zip(pairs, PAIR_ORDERS, strict=True)
        )
        and max(pair["candidate"]["ms"] for pair in pairs)
        < min(pair["control"]["ms"] for pair in pairs)
    )


def _contenders(assessed: dict[str, Any], baseline: int) -> list[int]:
    return sorted(
        (
            int(micro)
            for micro, result in assessed.items()
            if int(micro) != baseline and result["status"] == "ok" and _passes(result["parity"])
        ),
        key=lambda micro: (assessed[str(micro)]["ms"], -micro),
    )


def _valid_benchmark(result: dict[str, Any]) -> bool:
    if result["status"] == "ok":
        return all(_positive(result[name]) for name in ("ms", "images_per_second", "peak_mib"))
    return result["status"] in {"oom", "nonfinite"}


def _replay(doc: dict[str, Any], assessed: dict[str, Any], baseline: int) -> bool:
    best = baseline
    confirmations = iter(doc["confirmations"])
    for candidate in _contenders(assessed, baseline):
        if assessed[str(candidate)]["ms"] >= assessed[str(best)]["ms"]:
            continue
        trial = next(confirmations)
        if trial["candidate"] != candidate or trial["control"] != best:
            return False
        pairs = trial["pairs"]
        if len(pairs) != len(PAIR_ORDERS) or not all(
            tuple(pair["order"]) == order and all(_valid_benchmark(pair[role]) for role in order)
            for pair, order in zip(pairs, PAIR_ORDERS, strict=True)
        ):
            return False
        if _confirmed(pairs):
            best = candidate
    return (
        next(confirmations, None) is None
        and doc["microbatch"] == best
        and doc["status"] == ("baseline" if best == baseline else "confirmed")
    )


def valid_receipt(doc: dict[str, Any], key: str) -> bool:
    """Check benchmark coverage and replay every confirmed change of winner."""
    try:
        workload = doc["workload"]
        if doc["version"] != VERSION or doc["key"] != key or digest(workload) != key:
            return False
        if ("state_before" in doc or "state_after" in doc) and (
            not doc["state_before"] or doc["state_before"] != doc["state_after"]
        ):
            return False
        count = workload["effective_samples"]
        assessed = doc["assessed"]
        if set(assessed) != {str(micro) for micro in candidate_microbatches(count)} or not all(
            _valid_benchmark(result) for result in assessed.values()
        ):
            return False
        baseline = next(
            size for size in baseline_microbatches(count) if assessed[str(size)]["status"] == "ok"
        )
        return (
            doc["baseline"] == baseline
            and _passes(assessed[str(baseline)]["parity"])
            and _replay(doc, assessed, baseline)
        )
    except (KeyError, ValueError, TypeError, AttributeError, StopIteration, OverflowError):
        return False


def measure(
    snapshot: dict[str, Any],
    key: str,
    info: dict[str, Any],
    workdir: Path,
) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, workdir / "snapshot.pt")
    count = info["effective_samples"]
    assessed = {}
    reference = None
    counter = 0

    def arm(microbatch: int) -> tuple[dict[str, Any], TensorCapture | None]:
        nonlocal counter
        counter += 1
        result, checks = _run_arm(workdir, f"arm-{counter:04d}", microbatch)
        if result["status"] == "ok" and reference is not None:
            result["parity"] = compare_tensors(reference, checks)
        return result, checks

    for baseline in baseline_microbatches(count):
        result, checks = arm(baseline)
        assessed[str(baseline)] = result
        if result["status"] == "ok":
            reference = checks
            result["parity"] = compare_tensors(reference, reference)
            break
    else:
        raise RuntimeError("No feasible baseline microbatch")

    for microbatch in candidate_microbatches(count):
        if str(microbatch) not in assessed:
            assessed[str(microbatch)], _ = arm(microbatch)

    best = baseline
    confirmations = []
    for candidate in _contenders(assessed, baseline):
        if assessed[str(candidate)]["ms"] >= assessed[str(best)]["ms"]:
            continue
        arms = {"control": best, "candidate": candidate}
        pairs = [
            {"order": list(order), **{role: arm(arms[role])[0] for role in order}}
            for order in PAIR_ORDERS
        ]
        confirmations.append({"control": best, "candidate": candidate, "pairs": pairs})
        if _confirmed(pairs):
            best = candidate

    receipt = {
        "version": VERSION,
        "key": key,
        "workload": info,
        "baseline": baseline,
        "assessed": assessed,
        "confirmations": confirmations,
        "microbatch": best,
        "status": "baseline" if best == baseline else "confirmed",
    }
    if not valid_receipt(receipt, key):
        raise RuntimeError("Incomplete effective-update calibration evidence")
    return receipt


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This worker is invoked by the training entry point.")
    worker(Path(sys.argv[2]))
