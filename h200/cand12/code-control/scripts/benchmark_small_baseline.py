"""Select native or compiled official baselines using bounded real ImageNet updates.

Candidate weights are diagnostic only and are never reused by full training.
"""

# Match retained runtime internals and script-style imports.
# pyright: reportPrivateUsage=false, reportExplicitAny=false
# pyright: reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false
# ruff: noqa: NPY002
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from benchmark_rmt_training import (
    BoundedBatches,
    atomic_json,
    install_stop_handlers,
    is_oom,
    state_digest,
    wait_for_trial,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor
    from torch.utils.data import DataLoader


class ParityValidationError(RuntimeError):
    """A finite eager reference disagreed numerically with an optimization candidate."""


def assert_numerical_parity(
    actual: Any, expected: Any, *, label: str, rtol: float, atol: float
) -> None:
    import torch

    if not bool(torch.isfinite(expected).all()):
        raise RuntimeError(f"nonfinite eager reference for {label}; model failure")
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        raise ParityValidationError(f"{label} numerical parity failed: {error}") from error


def compile_unsupported(error: BaseException) -> bool:
    """Only an explicit Dynamo unsupported operation permits native fallback."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if type(current).__name__ == "Unsupported" and type(current).__module__.startswith(
            "torch._dynamo"
        ):
            return True
        pending.extend(
            value
            for value in (
                current.__cause__,
                current.__context__,
                getattr(current, "inner_exception", None),
            )
            if isinstance(value, BaseException)
        )
    return False


def trial_command(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    for name in (
        "model",
        "models_root",
        "source_manifest",
        "data_root",
        "data_manifest",
        "workers",
        "seed",
        "warmup_updates",
        "measure_updates",
        "stop_file",
    ):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    return [*command, "--output-dir", str(output), "--candidate", mode]


def aggregate_relative_l2(
    pairs: Iterable[tuple[Tensor, Tensor]], *, label: str, maximum: float
) -> dict[str, float | int]:
    """Check total relative error without an absolute tolerance hiding small gradients."""
    import torch

    difference_square = reference_square = None
    tensor_count = 0
    for actual, expected in pairs:
        actual_fp32 = actual.detach().float()
        expected_fp32 = expected.detach().float()
        if not bool(torch.isfinite(expected_fp32).all()):
            raise RuntimeError(f"nonfinite {label} in eager reference; model failure")
        if not bool(torch.isfinite(actual_fp32).all()):
            raise ParityValidationError(
                f"nonfinite {label} in candidate with finite eager reference"
            )
        difference = (actual_fp32 - expected_fp32).square().sum(dtype=torch.float64)
        reference = expected_fp32.square().sum(dtype=torch.float64)
        difference_square = (
            difference if difference_square is None else difference_square + difference
        )
        reference_square = reference if reference_square is None else reference_square + reference
        tensor_count += 1
    if difference_square is None or reference_square is None:
        raise RuntimeError(f"no tensors available for {label} parity")
    numerator, denominator = float(difference_square.item()), float(reference_square.item())
    relative = (
        math.sqrt(numerator / denominator) if denominator else (0.0 if not numerator else math.inf)
    )
    if not math.isfinite(relative) or relative > maximum:
        raise ParityValidationError(
            f"{label} aggregate relative L2 {relative:.6g} exceeds {maximum}"
        )
    return {
        "difference_squared_l2": numerator,
        "reference_squared_l2": denominator,
        "relative_l2": relative,
        "maximum_relative_l2": maximum,
        "tensor_count": tensor_count,
    }


def verify_forward_backward(model: Any, execution: Any, device: Any) -> dict[str, Any]:
    import imagenet1k_runtime as runtime
    import torch

    reference = copy.deepcopy(model).eval()
    runtime._compile_model_in_place(model, execution)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(90210)
    inputs = torch.randn((2, 3, 224, 224), generator=generator, device=device).contiguous(
        memory_format=torch.channels_last
    )
    # Compare the same BF16 computation, including parameter gradients, before any updates.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference(inputs)
        actual = model(inputs)
    logits_parity = aggregate_relative_l2([(actual, expected)], label="logits", maximum=0.02)
    assert_numerical_parity(actual.float(), expected.float(), label="logits", rtol=0.08, atol=0.025)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    gradient_pairs: list[tuple[Tensor, Tensor]] = []
    for (name, parameter), (reference_name, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if name != reference_name:
            raise RuntimeError("compiled model parameter names changed")
        if (parameter.grad is None) != (reference_parameter.grad is None):
            raise RuntimeError(f"gradient presence differs: {name}")
        if parameter.grad is not None:
            assert_numerical_parity(
                parameter.grad,
                reference_parameter.grad,
                label=f"gradient {name}",
                rtol=0.12,
                atol=0.025,
            )
            gradient_pairs.append((parameter.grad, reference_parameter.grad))
    gradient_parity = aggregate_relative_l2(gradient_pairs, label="gradients", maximum=0.05)
    model.zero_grad(set_to_none=True)
    del reference, expected, actual, inputs
    torch.cuda.empty_cache()
    return {
        "status": "passed",
        "compiled": execution.compile_model,
        "gradient_tensors": len(gradient_pairs),
        "logits_aggregate": logits_parity,
        "gradients_aggregate": gradient_parity,
        "precision": "bfloat16",
        "logits_rtol": 0.08,
        "logits_atol": 0.025,
        "gradient_rtol": 0.12,
        "gradient_atol": 0.025,
    }


def run_candidate(args: argparse.Namespace) -> dict[str, Any]:
    import imagenet1k_runtime as runtime
    import numpy as np
    import run_small_baseline_imagenet1k as runner
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one assigned CUDA GPU")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_root = args.output_dir / "diagnostic"
    args.batch_size, args.no_compile, args.resume = 128, args.candidate != "compiled", False
    args.smoke_updates = 0
    task = runner.make_task(args)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = runner.make_builder(args, task)(task.model_key, task.source_root, 1000).to(device)
    model.to(memory_format=torch.channels_last)  # pyright: ignore[reportCallIssue]
    optimizer = runtime._build_optimizer(model, task.learning_rate, device)
    bundle = runtime._build_loaders(task, device)
    initial_rng = runtime._capture_rng(bundle, device)
    initial_state = {
        "model": state_digest(model.state_dict()),
        "optimizer": state_digest(optimizer.state_dict()),
        "rng": state_digest(initial_rng),
        "source": task.source_sha256,
    }
    parity = verify_forward_backward(model, task.execution, device)
    runtime._restore_rng(initial_rng, bundle, device)
    accumulation = task.gradient_accumulation_steps
    if (args.warmup_updates + args.measure_updates) * accumulation > len(bundle.train):
        raise ValueError("insufficient real training batches")
    iterator = iter(bundle.train)
    mixup = runtime._make_mixup(task)
    full_updates = math.ceil(len(bundle.train) / accumulation)
    offset = 0
    original_lr = runtime._learning_rate
    runtime._learning_rate = lambda task, step, _: original_lr(task, offset + step, full_updates)
    measurements = {}
    try:
        for phase, updates in (
            ("warmup", args.warmup_updates),
            ("measurement", args.measure_updates),
        ):
            print(
                "SMALL_BASELINE_BENCHMARK_STAGE="
                + json.dumps({"phase": phase, "mode": args.candidate, "updates": updates}),
                flush=True,
            )
            torch.cuda.synchronize()
            if phase == "measurement":
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            metrics, _ = runtime._train_epoch(
                model,
                cast(
                    "DataLoader[Any]",
                    cast("object", BoundedBatches(iterator, updates * accumulation)),
                ),
                optimizer,
                mixup,
                task,
                device,
                offset,
                stop_requested=args.stop_file.is_file,
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            measurements[phase] = {
                "seconds": elapsed,
                "optimizer_updates": int(metrics["optimizer_steps"]),
                "examples": int(metrics["examples"]),
                "images_per_second": metrics["examples"] / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "loss": metrics["loss"],
            }
            offset += updates
    finally:
        runtime._learning_rate = original_lr
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise RuntimeError("nonfinite model parameters after benchmark")
    return {
        "status": "passed",
        "model": args.model,
        "mode": args.candidate,
        "compile": task.execution.compile_model,
        "batch_size": 128,
        "workers": args.workers,
        "global_batch_size": 256,
        "gradient_accumulation_steps": accumulation,
        "initial_state": initial_state,
        "validation": parity,
        "warmup_updates": args.warmup_updates,
        "measure_updates": args.measure_updates,
        "real_data": True,
        "loader_wait_and_augmentation_included": True,
        "warmup_excluded_from_measurement": True,
        "device": torch.cuda.get_device_name(),
        **measurements,
    }


def select_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != 2 or {result.get("mode") for result in results} != {"eager", "compiled"}:
        raise ValueError("both native and compiled candidates must be recorded")
    for result in results:
        if result.get("status") not in (
            "passed",
            "oom",
            "compile_unsupported",
            "parity_failed",
            "compile_disabled",
        ):
            raise ValueError("runtime/algorithm failures cannot be silently bypassed")
        if (
            result.get("status") in ("compile_unsupported", "parity_failed", "compile_disabled")
            and result["mode"] != "compiled"
        ):
            raise ValueError("native parity or execution failure cannot be a compile fallback")
        if result.get("status") == "compile_disabled" and not result.get("reason"):
            raise ValueError("disabled compilation requires an explicit correctness policy reason")
    passed = [r for r in results if r["status"] == "passed"]
    if not passed:
        raise RuntimeError("no candidate passed; physical batch 128 cannot be selected")
    for result in passed:
        measurement = result["measurement"]
        if (
            result["batch_size"] != 128
            or result["gradient_accumulation_steps"] != 2
            or result["global_batch_size"] != 256
            or measurement["optimizer_updates"] != result["measure_updates"]
            or measurement["examples"] != 256 * result["measure_updates"]
            or not math.isfinite(measurement["images_per_second"])
            or measurement["images_per_second"] <= 0
            or not math.isfinite(measurement["loss"])
            or result["validation"]["status"] != "passed"
        ):
            raise ValueError("incomplete or invalid training measurement")
    if len({json.dumps(r["initial_state"], sort_keys=True) for r in passed}) != 1:
        raise ValueError("candidate initial weights, optimizer, RNG or sources differ")
    winner = max(passed, key=lambda r: r["measurement"]["images_per_second"])
    return {
        "status": "selected",
        "model": winner["model"],
        "compile": winner["compile"],
        "batch_size": 128,
        "workers": winner["workers"],
        "global_batch_size": 256,
        "images_per_second": winner["measurement"]["images_per_second"],
        "results": results,
        "note": "Single steady-state measurement per mode; compile warmup excluded. "
        "Diagnostic weights are discarded; full training starts from the requested seed.",
    }


def compare_candidates(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    for mode in ("eager", "compiled"):
        if args.stop_file.is_file():
            raise SystemExit(130)
        output = args.output_dir / mode
        output.mkdir(parents=True, exist_ok=False)
        if mode == "compiled" and args.native_only:
            disabled = {
                "status": "compile_disabled",
                "mode": "compiled",
                "model": args.model,
                "reason": "Explicit native-only correctness policy: default torch.compile BF16 "
                "previously failed numerical parity or runtime stability validation; do not retry at "
                "another initialization or reuse the invalid optimization.",
                "executed": False,
            }
            atomic_json(output / "result.json", disabled)
            results.append(disabled)
            continue
        environment = os.environ.copy()
        environment.update(
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            TORCHINDUCTOR_COMPILE_THREADS="4",
            TRITON_CACHE_DIR=str(args.output_dir / "cache" / "triton"),
            TORCHINDUCTOR_CACHE_DIR=str(args.output_dir / "cache" / "inductor"),
        )
        code = wait_for_trial(
            trial_command(args, mode, output), environment, args.trial_timeout_seconds
        )
        if args.stop_file.is_file():
            raise SystemExit(130)
        report = output / "result.json"
        if not report.is_file():
            raise RuntimeError(f"{mode} exited {code} without report; inspect trial log")
        result = json.loads(report.read_text())
        if result.get("status") == "stopped":
            raise SystemExit(130)
        if result.get("status") not in (
            "passed",
            "oom",
            "compile_unsupported",
            "parity_failed",
        ) or (result["status"] == "passed" and code != 0):
            raise RuntimeError(f"{mode} failed; inspect {report}")
        results.append(result)
    selected = select_candidate(results)
    atomic_json(args.output_dir / "selection.json", selected)
    print("SMALL_BASELINE_BENCHMARK_SELECTION=" + json.dumps(selected), flush=True)
    return selected


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=("efficientvmamba_t", "mobilenetv4_conv_small", "repvit_m0_9"),
    )
    for name in ("models-root", "source-manifest", "data-root", "data-manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Measure native execution only; record compilation disabled by correctness policy.",
    )
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--measure-updates", type=int, default=50)
    parser.add_argument("--trial-timeout-seconds", type=float, default=1800)
    parser.add_argument("--candidate", choices=("eager", "compiled"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.workers < 0
        or min(args.warmup_updates, args.measure_updates) < 1
        or (not math.isfinite(args.trial_timeout_seconds) or args.trial_timeout_seconds <= 0)
    ):
        parser.error("workers must be nonnegative; update counts and timeout must be positive")
    if args.native_only and args.candidate == "compiled":
        parser.error("native-only correctness policy forbids a compiled candidate")
    args.output_dir = args.output_dir.resolve()
    args.stop_file = args.stop_file.resolve()
    return args


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    install_stop_handlers(args.stop_file)
    if args.candidate:
        try:
            result = run_candidate(args)
        except Exception as error:
            import imagenet1k_runtime as runtime

            stopped = isinstance(error, (InterruptedError, runtime.TrainingInterrupted))
            if stopped:
                status = "stopped"
            elif args.candidate == "compiled" and isinstance(error, ParityValidationError):
                status = "parity_failed"
            elif is_oom(error):
                status = "oom"
            elif args.candidate == "compiled" and compile_unsupported(error):
                status = "compile_unsupported"
            else:
                status = "failed"
            atomic_json(
                args.output_dir / "result.json",
                {
                    "status": status,
                    "mode": args.candidate,
                    "model": args.model,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise SystemExit(130 if stopped else 1) from error
        atomic_json(args.output_dir / "result.json", result)
    else:
        compare_candidates(args)


if __name__ == "__main__":
    main()
