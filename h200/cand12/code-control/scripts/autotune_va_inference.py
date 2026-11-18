"""Joint measured fusion search for original VA checkpoints on an idle CUDA GPU.

Each model trial is isolated in a subprocess so compiler/CUDA Graph state cannot
leak between plans. Only the best actually measured, numerically valid plan is
cached. A bounded beam search is not a proof of a global optimum.
"""
from __future__ import annotations

# Script-level canonical builders and private CUDA preparation are intentional.
# pyright: reportImplicitRelativeImport=false
import argparse
import gc
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Protocol, cast

import torch
import triton
from va128_inference import make_inference_snapshot, prepare_va_inference
from va_family_inference import build_va_family, load_va_checkpoint
from va_inference_hints import measured_component_hints
from va_inference_identity import default_cache_dir, inference_signature
from va_inference_plan import (
    ExecutionPlan,
    plan_from_json,
    plan_id,
    plan_to_dict,
    plan_to_json,
    signature_sha,
    write_plan_cache,
)
from va_inference_search import search_execution_plans
from va_inference_unified import collect_catalog


class _DeviceUtilities(Protocol):
    def get_device_properties(self, device: int) -> dict[str, int]: ...


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return cast("dict[str, object]", value)


def _configure() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(809)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _model_and_input(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.Tensor]:
    _configure()
    model = build_va_family(args.width).eval()
    load_va_checkpoint(model, args.checkpoint)
    example = torch.randn(args.batch, 3, 224, 224, device=args.device).contiguous(
        memory_format=torch.channels_last,
    )
    return model, example


def _inspect(args: argparse.Namespace) -> None:
    model, example = _model_and_input(args)
    signature = inference_signature(model, example)
    frozen = make_inference_snapshot(model, example.device)
    catalog = collect_catalog(frozen, example)
    index = example.device.index
    if index is None:
        raise RuntimeError("the allocated CUDA tensor must have a device ordinal")
    # CUDA exposes this extension; the abstract DriverBase stub omits it.
    utilities = cast("_DeviceUtilities", triton.runtime.driver.active.utils)  # pyright: ignore[reportAttributeAccessIssue]
    properties = utilities.get_device_properties(index)
    _write(args.output, {
        "signature": signature, "catalog": catalog,
        "shared_memory_limit": properties["max_shared_mem"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    })


def _trial(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    model, example = _model_and_input(args)
    signature = inference_signature(model, example)
    inspection = _read(args.inspection)
    if signature != inspection["signature"]:
        raise RuntimeError("source, model, hardware or backend settings changed during search")
    plan = plan_from_json(args.plan.read_text())
    torch.cuda.reset_peak_memory_stats(example.device)
    engine = prepare_va_inference(model, example, plan=plan)
    del model
    gc.collect()
    output = engine(example)
    for _ in range(29):
        output = engine(example)
    torch.cuda.synchronize(example.device)
    logits = output.detach().float().cpu()
    finite = bool(torch.isfinite(logits).all())
    if logits.shape != (args.batch, 1000):
        raise ValueError("candidate changed the logits contract")
    relative = 0.0
    if plan == ExecutionPlan():
        if not finite:
            raise ValueError("the baseline produced nonfinite logits")
        torch.save(logits, args.reference)
    else:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        if reference.shape != logits.shape or not torch.isfinite(reference).all():
            raise ValueError("the saved baseline logits are invalid")
        relative = float((logits - reference).norm() / reference.norm().clamp_min(1e-12))
    valid = finite and relative <= args.relative_l2_limit
    samples = []
    if valid:
        for _ in range(5):
            torch.cuda.synchronize(example.device)
            begin = time.perf_counter()
            for _ in range(100):
                engine(example)
            torch.cuda.synchronize(example.device)
            samples.append((time.perf_counter() - begin) * 10.0)
    result = {
        "status": "ok" if valid else "invalid", "plan_id": plan_id(plan),
        "plan": plan_to_dict(plan), "signature": signature,
        "median_ms": statistics.median(samples) if samples else None, "samples_ms": samples,
        "relative_l2": relative if finite else None, "finite": finite,
        "relative_l2_limit": args.relative_l2_limit,
        "images_s": 1000 * args.batch / statistics.median(samples) if samples else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(example.device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(example.device) / 1024**2,
        "preparation_and_measurement_seconds": time.perf_counter() - started,
        "applied_fusions": engine.applied_fusions,
    }
    _write(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "plan_id", "median_ms", "relative_l2")}), flush=True)


def _terminate(child: subprocess.Popen[bytes]) -> None:
    # The child can finish between wait timeout and signal delivery.
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def _run(command: list[str], log: Path, environment: dict[str, str], timeout: float) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=environment, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(child)
            return 124
        except BaseException:
            _terminate(child)
            raise


def _search(args: argparse.Namespace) -> None:
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join((str(root / "src"), str(root / "scripts"), environment.get("PYTHONPATH", ""))),
        OMP_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="4", LNET_DISABLE_LAUNCH_AUTOTUNE="1",
    )
    base = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint.resolve()),
            "--width", str(args.width), "--batch", str(args.batch), "--device", args.device,
            "--relative-l2-limit", str(args.relative_l2_limit)]
    inspection_path = work / "inspection.json"
    code = _run([*base, "--action", "inspect", "--output", str(inspection_path)],
                work / "inspect.log", environment, args.timeout)
    if code:
        raise RuntimeError(f"catalog inspection failed ({code}); see {work / 'inspect.log'}")
    inspection = _read(inspection_path)
    signature = cast("dict[str, object]", inspection["signature"])
    catalog = cast("list[dict[str, object]]", inspection["catalog"])
    run_root = work / signature_sha(signature)
    run_root.mkdir(exist_ok=True)
    reference = run_root / "reference.pt"
    hints = _read(args.hints) if args.hints else None
    if args.component_results is not None and hints is None:
        hardware = cast("dict[str, object]", signature["hardware"])
        hints = measured_component_hints(catalog, args.component_results,
                                         device_name=cast("str", hardware["name"]))
        _write(work / "component-hints.json", hints)
    calls = 0

    def measure(plan: ExecutionPlan) -> dict[str, object]:
        nonlocal calls
        identifier = plan_id(plan)
        output = run_root / f"{identifier}.json"
        if output.is_file():
            cached = _read(output)
            if (cached.get("signature") == signature and cached.get("plan_id") == identifier
                    and cached.get("relative_l2_limit") == args.relative_l2_limit
                    and (plan != ExecutionPlan() or reference.is_file())):
                return cached
        plan_path = run_root / f"{identifier}.plan.json"
        plan_path.write_text(plan_to_json(plan), encoding="utf-8")
        trial_environment = dict(environment)
        trial_environment["TORCHINDUCTOR_CACHE_DIR"] = str(run_root / "compiler" / identifier)
        trial_environment["TRITON_CACHE_DIR"] = str(work / "triton")
        command = [*base, "--action", "trial", "--output", str(output),
                   "--plan", str(plan_path), "--inspection", str(inspection_path),
                   "--reference", str(reference)]
        output.unlink(missing_ok=True)
        code = _run(command, run_root / f"{identifier}.log", trial_environment, args.timeout)
        calls += 1
        if code or not output.is_file():
            result: dict[str, object] = {
                "status": "error", "plan_id": identifier, "plan": plan_to_dict(plan),
                "signature": signature, "returncode": code,
                "error": f"trial failed; see {identifier}.log",
            }
            if output.is_file():
                result.update(_read(output))
            result["status"] = "error"
            result["returncode"] = code
            if "changed during search" in str(result.get("error", "")):
                result["fatal"] = True
            _write(output, result)
        else:
            result = _read(output)
        print(json.dumps({"trial": calls, "plan_id": identifier,
                          "status": result["status"], "median_ms": result.get("median_ms"),
                          "relative_l2": result.get("relative_l2")}), flush=True)
        return result

    found = search_execution_plans(
        catalog, measure, hints=hints, beam_width=args.beam_width, max_trials=args.max_trials,
        shared_memory_limit=cast("int", inspection["shared_memory_limit"]),
    )
    evidence = {"algorithm": "bounded_beam", "beam_width": args.beam_width,
                "max_trials": args.max_trials, "stopped_reason": found.stopped_reason,
                "catalog": catalog, "history": list(found.history), "component_hints": hints,
                "relative_l2_limit": args.relative_l2_limit,
                "accuracy_scope": "fixed Gaussian input only; no real-image accuracy claim"}
    chosen = dict(found.best_measurement)
    summary = {"signature": signature, "best_plan": plan_to_dict(found.best_plan),
               "best_measurement": chosen, "search": evidence}
    _write(work / "summary.json", summary)
    if args.write_cache:
        write_plan_cache(args.cache_dir, signature, found.best_plan, chosen, evidence)
    print(json.dumps({"best_plan_id": plan_id(found.best_plan), "median_ms": chosen["median_ms"],
                      "stopped_reason": found.stopped_reason}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("search", "inspect", "trial"), default="search")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(32, 48, 64, 96, 128), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--inspection", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--hints", type=Path)
    parser.add_argument("--component-results", type=Path)
    parser.add_argument("--relative-l2-limit", type=float, default=0.05)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--max-trials", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--write-cache", action="store_true")
    args = parser.parse_args()
    if (args.batch < 1 or not 0 < args.relative_l2_limit <= 1
            or args.timeout <= 0 or not math.isfinite(args.timeout)
            or args.max_trials < 1 or args.beam_width < 1):
        parser.error("batch, timeout and numerical limit must be positive")
    if args.action == "trial" and any(value is None for value in (args.plan, args.inspection, args.reference)):
        parser.error("trial requires --plan, --inspection and --reference")
    try:
        {"inspect": _inspect, "trial": _trial, "search": _search}[args.action](args)
    except Exception as error:
        if args.action != "search":
            _write(args.output, {"status": "error", "error": str(error), "exception": type(error).__name__})
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
