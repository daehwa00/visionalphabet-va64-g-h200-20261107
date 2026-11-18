#!/usr/bin/env python3
"""Evaluate scan unification with parity, matrix, and fair CUDA evidence."""

from __future__ import annotations

# The evaluator intentionally retains the removed recurrence as a frozen oracle.
# It is not imported by production code.
# pyright: reportAny=false, reportExplicitAny=false
import argparse
import gc
import hashlib
import json
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import torch
from torch import Tensor, nn

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_product_scan_contracts import DEFAULT_EPSILON
from lnet.pac_product_scan_normalization import static_variance_tables
from lnet.pac_product_scan_pipeline import (
    ScanMemoryPolicy,
    run_product_scan_path_collapse_pipeline,
    run_product_scan_pipeline,
)
from lnet.pac_product_scan_reference import (
    bidirectional_product_scan_reference,
    product_scan_coarse4_reference,
    product_scan_descriptor4_reference,
    product_scan_full4_reference,
)
from lnet.pac_same_resolution_depth import SameResolutionPoleScanBlock
from lnet.pac_triton_bidirectional_product_scan import pac_triton_bidirectional_product_scan

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lnet.complex_scan_types import ComplexField


LENGTHS = (7, 14, 28, 56)
EPILOGUES = ("coarse", "same_resolution", "descriptor")
PARITY_LIMIT = 2.0e-2
TIME_LIMIT = 1.02
MEMORY_LIMIT = 1.0
MatrixEpilogue = Literal["coarse", "same_resolution", "descriptor"]
Pole = tuple[Tensor, Tensor, Tensor, Tensor]
MatrixOutput = Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]


class BenchmarkResult(TypedDict):
    median_ms: float
    samples_ms: list[float]
    peak_bytes: int


class PreparedBenchmark(TypedDict):
    module: nn.Module
    real: Tensor
    imag: Tensor


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provenance-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--modes", type=int, default=128)
    parser.add_argument("--poles", type=int, default=160)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--matrix-batch-size", type=int, default=1)
    parser.add_argument("--matrix-modes", type=int, default=8)
    parser.add_argument("--require-parity", action="store_true")
    parser.add_argument("--require-no-regression", action="store_true")
    parser.add_argument("--require-matrix", action="store_true")
    return parser.parse_args()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_value(root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _source_paths(root: Path) -> tuple[Path, ...]:
    production = sorted((root / "src/lnet").glob("pac_*product_scan*.py"))
    explicit = (
        root / "src/lnet/complex_scan_stage.py",
        root / "src/lnet/pac_same_resolution_depth.py",
        Path(__file__).resolve(),
        root / "scripts/run_r2k3_scan_unification_gate.sh",
        root / "tests/test_evaluate_r2k3_scan_unification.py",
        root / "tests/test_pac_product_scan_module_boundaries.py",
        root / "tests/test_pac_product_scan_pipeline.py",
        root / "tests/test_pac_product_scan_ops.py",
        root / "tests/test_product_scan_path_collapse.py",
        root / "tests/test_pac_same_resolution_depth.py",
        root / "tests/test_pac_kernel_launch_config.py",
        root / "tests/test_complex_scan_module_boundaries.py",
        root / "tests/test_complex_scan.py",
    )
    return tuple(dict.fromkeys(path for path in (*production, *explicit) if path.is_file()))


def _source_provenance(root: Path | None = None) -> dict[str, object]:
    root = _repository_root() if root is None else root.resolve()
    records: dict[str, str] = {}
    aggregate = hashlib.sha256()
    for path in _source_paths(root):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records[relative] = digest
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
    status = _git_value(root, "status", "--short", "--", *records)
    return {
        "bundle_sha256": aggregate.hexdigest(),
        "files": records,
        "git_commit": _git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status.splitlines() if status else [],
    }


def _write_payload(payload: dict[str, object], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def _vertical_recurrence(
    pole: Pole,
    source: ComplexField,
    source_variance: Tensor,
    *,
    reverse: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """The removed production recurrence, retained only as benchmark oracle."""
    source_real, source_imag = source
    storage_dtype = source_real.dtype
    active_real = source_real.float()
    active_imag = source_imag.float()
    active_variance = source_variance.float()
    decay_real, decay_imag, gamma_real, gamma_imag = (
        value.reshape(1, 1, -1).float() for value in pole
    )
    if reverse:
        active_real = active_real.flip(1)
        active_imag = active_imag.flip(1)
        active_variance = active_variance.flip(1)
        decay_imag = -decay_imag
        gamma_imag = -gamma_imag
    batch, height, width, modes = active_real.shape
    state_real = active_real.new_zeros((batch, width, modes))
    state_imag = active_imag.new_zeros((batch, width, modes))
    variance = active_variance.new_zeros((batch, width, modes))
    variance_decay = decay_real.detach().square() + decay_imag.detach().square()
    variance_gamma = gamma_real.detach().square() + gamma_imag.detach().square()
    real_rows = []
    imag_rows = []
    variance_rows = []
    for row in range(height):
        input_real = active_real[:, row]
        input_imag = active_imag[:, row]
        next_real = (
            decay_real * state_real
            - decay_imag * state_imag
            + gamma_real * input_real
            - gamma_imag * input_imag
        )
        next_imag = (
            decay_real * state_imag
            + decay_imag * state_real
            + gamma_real * input_imag
            + gamma_imag * input_real
        )
        variance = variance_decay * variance + variance_gamma * active_variance[:, row]
        state_real, state_imag = next_real, next_imag
        real_rows.append(state_real)
        imag_rows.append(state_imag)
        variance_rows.append(variance)
    output_real = torch.stack(real_rows, dim=1)
    output_imag = torch.stack(imag_rows, dim=1)
    output_variance = torch.stack(variance_rows, dim=1)
    if reverse:
        output_real = output_real.flip(1)
        output_imag = output_imag.flip(1)
        output_variance = output_variance.flip(1)
    return output_real.to(storage_dtype), output_imag.to(storage_dtype), output_variance


def _legacy_collapsed_memory(
    block: SameResolutionPoleScanBlock,
    real: Tensor,
    imag: Tensor,
) -> ComplexField:
    source = block.reader(real, imag)
    batch, height, width, modes = source[0].shape
    pole_x, pole_y = block.pole_scan.pole_coefficients(tuple(source[0].shape))
    horizontal = pac_triton_bidirectional_product_scan(pole_x, source)
    variance_x, _ = static_variance_tables(
        pole_x[0].detach(),
        pole_x[1].detach(),
        pole_x[2].detach(),
        pole_x[3].detach(),
        pole_y[0].detach(),
        pole_y[1].detach(),
        pole_y[2].detach(),
        pole_y[3].detach(),
        width,
        height,
    )
    positive_variance = variance_x[0].view(1, 1, width, modes).expand(
        batch, height, width, modes
    )
    negative_variance = variance_x[1].view(1, 1, width, modes).expand_as(positive_variance)
    scans = (
        _vertical_recurrence(pole_y, horizontal[:2], positive_variance, reverse=False),
        _vertical_recurrence(pole_y, horizontal[2:], negative_variance, reverse=False),
        _vertical_recurrence(pole_y, horizontal[:2], positive_variance, reverse=True),
        _vertical_recurrence(pole_y, horizontal[2:], negative_variance, reverse=True),
    )
    real_paths = []
    imag_paths = []
    for state_real, state_imag, variance in scans:
        active_variance = (
            variance.mean((1, 2), keepdim=True)
            if block.pole_scan.product_gain_normalization == "global"
            else variance
        )
        inverse = torch.rsqrt(active_variance.detach().clamp_min(DEFAULT_EPSILON)).to(
            state_real.dtype
        )
        real_paths.append(state_real * inverse)
        imag_paths.append(state_imag * inverse)
    return block.collapse_paths(
        torch.stack(real_paths, dim=-2),
        torch.stack(imag_paths, dim=-2),
    )


class _Candidate(nn.Module):
    def __init__(self, block: SameResolutionPoleScanBlock) -> None:
        super().__init__()
        self.block = block

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        return self.block.collapsed_memory(real, imag)


class _Legacy(nn.Module):
    def __init__(self, block: SameResolutionPoleScanBlock) -> None:
        super().__init__()
        self.block = block

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        return _legacy_collapsed_memory(self.block, real, imag)


def _block(modes: int, poles: int) -> SameResolutionPoleScanBlock:
    template = ComplexScanStage(poles, maximum_phase=1.0, output_modes=None)
    return SameResolutionPoleScanBlock(
        modes,
        pole_modes=poles,
        reader_rank=2,
        kernel_size=3,
        pole_template=template,
        post_hidden=max(1, round(1.5 * modes)),
    ).cuda().train()


def _relative_l2(value: Tensor, target: Tensor) -> float:
    difference = value.float() - target.float()
    denominator = target.float().square().sum().sqrt().clamp_min(1.0e-12)
    return float((difference.square().sum().sqrt() / denominator).detach())


def _tensor_error(value: Tensor, target: Tensor) -> dict[str, float]:
    difference = value.float() - target.float()
    target_float = target.float()
    denominator = target_float.square().sum().sqrt().clamp_min(1.0e-12)
    return {
        "relative_l2": float((difference.square().sum().sqrt() / denominator).detach()),
        "max_abs": float(difference.abs().max().detach()),
        "target_l2": float(target_float.square().sum().sqrt().detach()),
        "actual_l2": float(value.float().square().sum().sqrt().detach()),
    }


def _parity(
    legacy: _Legacy,
    candidate: _Candidate,
    modes: int,
) -> dict[str, float]:
    torch.manual_seed(107)
    legacy_input = tuple(
        torch.randn(2, 7, 7, modes, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    )
    candidate_input = tuple(value.detach().clone().requires_grad_() for value in legacy_input)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = legacy(*legacy_input)
        actual = candidate(*candidate_input)
    cotangent = tuple(torch.randn_like(value) for value in actual)
    expected_tensors = (*legacy_input, *legacy.parameters())
    actual_tensors = (*candidate_input, *candidate.parameters())
    expected_gradients = torch.autograd.grad(
        expected,
        expected_tensors,
        cotangent,
        allow_unused=True,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        actual_tensors,
        cotangent,
        allow_unused=True,
    )
    gradient_errors = []
    for value, target in zip(actual_gradients, expected_gradients, strict=True):
        if value is None or target is None:
            if value is not target:
                gradient_errors.append(float("inf"))
            continue
        gradient_errors.append(_relative_l2(value, target))
    return {
        "output_relative_l2": max(
            _relative_l2(value, target) for value, target in zip(actual, expected, strict=True)
        ),
        "gradient_relative_l2": max(gradient_errors, default=0.0),
    }


def _clear_gradients(module: nn.Module, real: Tensor, imag: Tensor) -> None:
    module.zero_grad(set_to_none=True)
    real.grad = None
    imag.grad = None


def _run_training_step(module: nn.Module, real: Tensor, imag: Tensor) -> None:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = cast("ComplexField", module(real, imag))
        loss = output[0].float().square().mean() + output[1].float().square().mean()
    loss.backward()


def _prepare_benchmark(
    module: nn.Module,
    real: Tensor,
    imag: Tensor,
    *,
    warmup: int,
) -> PreparedBenchmark:
    compiled = cast(
        "nn.Module",
        torch.compile(module, mode="default", dynamic=False, fullgraph=False),
    )
    for _ in range(warmup):
        _clear_gradients(module, real, imag)
        _run_training_step(compiled, real, imag)
    _clear_gradients(module, real, imag)
    torch.cuda.synchronize()
    return {"module": compiled, "real": real, "imag": imag}


def _benchmark(
    modules: dict[str, nn.Module],
    *,
    batch_size: int,
    modes: int,
    warmup: int,
    steps: int,
) -> tuple[dict[str, BenchmarkResult], dict[str, dict[str, str]]]:
    """Counterbalance frozen-input timings and compare incremental peak memory."""
    torch.manual_seed(109)
    frozen = tuple(
        torch.randn(
            batch_size,
            7,
            7,
            modes,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(2)
    )
    prepared: dict[str, PreparedBenchmark] = {}
    errors: dict[str, dict[str, str]] = {}
    for name, module in modules.items():
        try:
            prepared[name] = _prepare_benchmark(
                module,
                frozen[0].detach().clone().requires_grad_(),
                frozen[1].detach().clone().requires_grad_(),
                warmup=warmup,
            )
        except Exception as error:  # noqa: BLE001 - retain failed candidate evidence.
            errors[name] = {"type": type(error).__name__, "message": str(error)}
    if "legacy" not in prepared:
        return {}, errors
    samples: dict[str, list[float]] = {name: [] for name in prepared}
    names = tuple(prepared)
    for iteration in range(steps):
        order = names if iteration % 2 == 0 else tuple(reversed(names))
        for name in order:
            active = prepared[name]
            _clear_gradients(active["module"], active["real"], active["imag"])
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _run_training_step(active["module"], active["real"], active["imag"])
            end.record()
            end.synchronize()
            samples[name].append(float(start.elapsed_time(end)))

    peak_bytes: dict[str, int] = {}
    for name in (*names, *reversed(names)):
        active = prepared[name]
        _clear_gradients(active["module"], active["real"], active["imag"])
        gc.collect()
        torch.cuda.synchronize()
        starting_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _run_training_step(active["module"], active["real"], active["imag"])
        torch.cuda.synchronize()
        incremental_peak = max(0, torch.cuda.max_memory_allocated() - starting_bytes)
        peak_bytes[name] = max(peak_bytes.get(name, 0), incremental_peak)
    return (
        {
            name: {
                "median_ms": statistics.median(values),
                "samples_ms": values,
                "peak_bytes": peak_bytes[name],
            }
            for name, values in samples.items()
        },
        errors,
    )


def _matrix_specs() -> tuple[tuple[int, MatrixEpilogue, bool], ...]:
    return tuple(
        (length, epilogue, not (length == 7 and epilogue == "coarse"))
        for length in LENGTHS
        for epilogue in EPILOGUES
    )


class _MatrixCase(nn.Module):
    def __init__(self, modes: int, epilogue: MatrixEpilogue, policy: ScanMemoryPolicy) -> None:
        super().__init__()
        self.epilogue: MatrixEpilogue = epilogue
        self.policy: ScanMemoryPolicy = policy
        self.pole_scan = ComplexScanStage(
            modes,
            maximum_phase=1.0,
            output_modes=None,
            scan_memory_policy=policy,
        )
        self.collapse = PathOnlyCollapse(modes, path_hidden=8)

    def poles(self, shape: tuple[int, int, int, int]) -> tuple[Pole, Pole]:
        return self.pole_scan.pole_coefficients(shape)

    def forward(self, real: Tensor, imag: Tensor) -> MatrixOutput:
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        pole_x, pole_y = self.poles(shape)
        if self.epilogue == "same_resolution":
            return run_product_scan_path_collapse_pipeline(
                pole_x,
                pole_y,
                (real, imag),
                self.collapse.packed_parameters(),
                gain_normalization="global",
                memory_policy=self.policy,
                epsilon=DEFAULT_EPSILON,
            )
        return cast(
            "MatrixOutput",
            run_product_scan_pipeline(
                pole_x,
                pole_y,
                (real, imag),
                epilogue=self.epilogue,
                gain_normalization="global",
                memory_policy=self.policy,
                epsilon=DEFAULT_EPSILON,
            ),
        )


class _MatrixReference(nn.Module):
    def __init__(self, candidate: _MatrixCase) -> None:
        super().__init__()
        self.epilogue: MatrixEpilogue = candidate.epilogue
        self.pole_scan = deepcopy(candidate.pole_scan)
        self.collapse = deepcopy(candidate.collapse)

    def poles(self, shape: tuple[int, int, int, int]) -> tuple[Pole, Pole]:
        return self.pole_scan.pole_coefficients(shape)

    def forward(self, real: Tensor, imag: Tensor) -> MatrixOutput:
        source = real, imag
        shape = cast("tuple[int, int, int, int]", tuple(real.shape))
        pole_x, pole_y = self.poles(shape)
        if self.epilogue == "same_resolution":
            full = product_scan_full4_reference(
                pole_x,
                pole_y,
                source,
                gain_normalization="global",
                epsilon=DEFAULT_EPSILON,
            )
            return self.collapse.forward_packed(*full)
        horizontal = bidirectional_product_scan_reference(pole_x, source)
        source_a = horizontal[:2]
        source_b = horizontal[2:]
        if self.epilogue == "coarse":
            return product_scan_coarse4_reference(
                pole_x,
                pole_y,
                source_a,
                source_b,
                gain_normalization="global",
                epsilon=DEFAULT_EPSILON,
            )
        return product_scan_descriptor4_reference(
            pole_x,
            pole_y,
            source_a,
            source_b,
            gain_normalization="global",
            epsilon=DEFAULT_EPSILON,
        )


def _outputs(value: MatrixOutput) -> tuple[Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def _finite(values: Iterable[Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def _matrix_parameters(
    module: _MatrixCase | _MatrixReference,
) -> tuple[nn.Parameter, ...]:
    parameters = tuple(module.pole_scan.parameters())
    if module.epilogue == "same_resolution":
        return *parameters, *module.collapse.parameters()
    return parameters


def _matrix_gradient_names(module: _MatrixCase | _MatrixReference) -> tuple[str, ...]:
    names = (
        "input_real",
        "input_imag",
        *(f"pole_scan.{name}" for name, _ in module.pole_scan.named_parameters()),
    )
    if module.epilogue == "same_resolution":
        return *names, *(f"collapse.{name}" for name, _ in module.collapse.named_parameters())
    return names


def _matrix_case(
    length: int,
    epilogue: MatrixEpilogue,
    *,
    policy: ScanMemoryPolicy,
    batch_size: int,
    modes: int,
) -> dict[str, object]:
    torch.manual_seed(8000 + length)
    candidate = _MatrixCase(modes, epilogue, policy).cuda().train()
    reference = _MatrixReference(candidate).cuda().train()
    reference_input = tuple(
        torch.randn(
            batch_size,
            length,
            length,
            modes,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(2)
    )
    candidate_input = tuple(value.detach().clone().requires_grad_() for value in reference_input)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = _outputs(reference(*reference_input))
        actual = _outputs(candidate(*candidate_input))
    generator = torch.Generator(device="cuda").manual_seed(9000 + length)
    cotangent = tuple(
        torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype)
        for value in actual
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (*reference_input, *_matrix_parameters(reference)),
        cotangent,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (*candidate_input, *_matrix_parameters(candidate)),
        cotangent,
    )
    output_errors = {
        f"output.{index}": _tensor_error(value, target)
        for index, (value, target) in enumerate(zip(actual, expected, strict=True))
    }
    gradient_errors = {
        name: _tensor_error(value, target)
        for name, value, target in zip(
            _matrix_gradient_names(candidate),
            actual_gradients,
            expected_gradients,
            strict=True,
        )
    }
    output_error = max(error["relative_l2"] for error in output_errors.values())
    gradient_error = max(error["relative_l2"] for error in gradient_errors.values())

    compiled = cast(
        "nn.Module",
        torch.compile(candidate, mode="default", dynamic=False, fullgraph=False),
    )
    compiled_input = tuple(value.detach().clone().requires_grad_() for value in candidate_input)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        compiled_output = _outputs(compiled(*compiled_input))
        compiled_loss = compiled_output[0].float().square().mean()
        for value in compiled_output[1:]:
            compiled_loss = compiled_loss + value.float().square().mean()
    compiled_loss.backward()
    compiled_gradients = (
        *(value.grad for value in compiled_input),
        *(parameter.grad for parameter in _matrix_parameters(candidate)),
    )
    compile_backward_complete = all(value is not None for value in compiled_gradients)
    compile_finite = compile_backward_complete and _finite(
        (*compiled_output, *(value for value in compiled_gradients if value is not None))
    )
    passed = max(output_error, gradient_error) <= PARITY_LIMIT and compile_finite
    return {
        "length": length,
        "epilogue": epilogue,
        "storage_dtype": "bfloat16",
        "forward_relative_l2": output_error,
        "backward_relative_l2": gradient_error,
        "forward_errors": output_errors,
        "backward_errors": gradient_errors,
        "compiled_forward_backward": compile_finite,
        "compiled_backward_gradients_complete": compile_backward_complete,
        "passed": passed,
    }


def _run_matrix(
    policy: ScanMemoryPolicy,
    *,
    batch_size: int,
    modes: int,
) -> list[dict[str, object]]:
    results = []
    for length, epilogue, supported in _matrix_specs():
        if not supported:
            results.append(
                {
                    "length": length,
                    "epilogue": epilogue,
                    "supported": False,
                    "status": "not_applicable",
                    "reason": "coarse endpoint packing requires even spatial dimensions",
                }
            )
            continue
        try:
            result = _matrix_case(
                length,
                epilogue,
                policy=policy,
                batch_size=batch_size,
                modes=modes,
            )
            result["supported"] = True
            result["status"] = "passed" if result["passed"] else "failed"
            results.append(result)
        except Exception as error:  # noqa: BLE001 - durable evidence must retain every cell.
            results.append(
                {
                    "length": length,
                    "epilogue": epilogue,
                    "supported": True,
                    "status": "error",
                    "passed": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        finally:
            torch.compiler.reset()
            torch.cuda.empty_cache()
    return results


def _matrix_passed(results: Sequence[dict[str, object]]) -> bool:
    expected = {(length, epilogue, supported) for length, epilogue, supported in _matrix_specs()}
    observed: set[tuple[int, str, bool]] = set()
    for result in results:
        length = result.get("length")
        epilogue = result.get("epilogue")
        supported = result.get("supported")
        if not isinstance(length, int) or not isinstance(epilogue, str):
            return False
        if not isinstance(supported, bool):
            return False
        observed.add((length, epilogue, supported))
    if observed != expected:
        return False
    return all(
        result.get("status") == "not_applicable"
        if not bool(result["supported"])
        else result.get("passed") is True
        for result in results
    )


def _policy_ratios(
    benchmark: dict[str, BenchmarkResult],
) -> dict[str, dict[str, float]]:
    baseline = benchmark["legacy"]
    return {
        name: {
            "time": result["median_ms"] / baseline["median_ms"],
            "memory": result["peak_bytes"] / baseline["peak_bytes"],
        }
        for name, result in benchmark.items()
        if name != "legacy"
    }


def _select_policy(
    benchmark: dict[str, BenchmarkResult],
    ratios: dict[str, dict[str, float]],
    parity: dict[str, dict[str, float]],
) -> str | None:
    eligible = [
        name
        for name, ratio in ratios.items()
        if max(parity.get(name, {"missing": float("inf")}).values()) <= PARITY_LIMIT
        and ratio["time"] <= TIME_LIMIT
        and ratio["memory"] <= MEMORY_LIMIT
    ]
    return min(eligible, key=lambda name: benchmark[name]["median_ms"], default=None)


def main() -> None:
    args = _arguments()
    provenance = _source_provenance()
    if args.provenance_only:
        _write_payload(
            {
                "schema": "lnet.r2k3.scan_unification.provenance.v1",
                "source_provenance": provenance,
            },
            args.output,
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("scan unification evaluation requires CUDA")
    if min(
        args.batch_size,
        args.modes,
        args.poles,
        args.steps,
        args.warmup,
        args.matrix_batch_size,
        args.matrix_modes,
    ) <= 0:
        raise ValueError("evaluation dimensions, warmup, and steps must be positive")

    torch.manual_seed(101)
    legacy = _Legacy(_block(args.modes, args.poles))
    candidates = {
        "recompute": _Candidate(deepcopy(legacy.block)),
        "retain": _Candidate(deepcopy(legacy.block)),
    }
    candidates["recompute"].block.pole_scan.scan_memory_policy = "recompute"
    candidates["retain"].block.pole_scan.scan_memory_policy = "retain"
    parity: dict[str, dict[str, float]] = {}
    parity_errors: dict[str, dict[str, str]] = {}
    for name, candidate in candidates.items():
        try:
            parity[name] = _parity(legacy, candidate, args.modes)
        except Exception as error:  # noqa: BLE001 - persist candidate failure in JSON.
            parity_errors[name] = {"type": type(error).__name__, "message": str(error)}

    benchmark, benchmark_errors = _benchmark(
        {"legacy": legacy, **candidates},
        batch_size=args.batch_size,
        modes=args.modes,
        warmup=args.warmup,
        steps=args.steps,
    )
    ratios = _policy_ratios(benchmark) if "legacy" in benchmark else {}
    provisional_policy = _select_policy(benchmark, ratios, parity)
    matrix = (
        _run_matrix(
            cast("ScanMemoryPolicy", provisional_policy),
            batch_size=args.matrix_batch_size,
            modes=args.matrix_modes,
        )
        if provisional_policy is not None
        else []
    )
    passed_matrix = _matrix_passed(matrix)
    selected_policy = provisional_policy if passed_matrix else None
    passed_parity = selected_policy is not None
    passed_performance = selected_policy is not None
    selected_ratios = ratios.get(provisional_policy or "", {})
    payload: dict[str, object] = {
        "schema": "lnet.r2k3.scan_unification.v2",
        "source_provenance": provenance,
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "performance_shape": [args.batch_size, 7, 7, args.modes],
        "poles": args.poles,
        "legacy_epsilon": DEFAULT_EPSILON,
        "policy_parity": parity,
        "policy_parity_errors": parity_errors,
        "benchmark_method": {
            "inputs": "preallocated frozen BF16 tensors",
            "latency": "counterbalanced CUDA events; gradient clearing excluded",
            "memory": "incremental peak allocated bytes; forward/backward only",
        },
        "baseline": benchmark.get("legacy"),
        "candidates": {name: benchmark[name] for name in candidates if name in benchmark},
        "benchmark_errors": benchmark_errors,
        "candidate_ratios": ratios,
        "provisional_policy": provisional_policy,
        "verification_matrix": matrix,
        "selected_policy": selected_policy,
        "selected_over_baseline_time": selected_ratios.get("time"),
        "selected_over_baseline_memory": selected_ratios.get("memory"),
        "passed_parity": passed_parity,
        "passed_performance": passed_performance,
        "passed_matrix": passed_matrix,
    }
    _write_payload(payload, args.output)
    if args.require_parity and not passed_parity:
        raise SystemExit(2)
    if args.require_no_regression and not passed_performance:
        raise SystemExit(3)
    if args.require_matrix and not passed_matrix:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
