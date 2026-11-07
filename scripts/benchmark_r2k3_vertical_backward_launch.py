#!/usr/bin/env python3
"""Screen low-register launch geometries for the D2262 56x56 PathH4 backward."""

from __future__ import annotations

# pyright: reportExplicitAny=false, reportPrivateUsage=false
import argparse
import json
import statistics
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import Tensor, nn

from lnet.pac_kernel_launch_config import (
    LaunchGeometry,
    LaunchScope,
    capture_launch_scopes,
    override_launch_geometry,
)
from lnet.pac_phase_gated_transition import PathOnlyCollapse
from lnet.pac_product_scan_ops import PATH_COLLAPSE_BACKWARD_LAUNCH_NAME
from lnet.pac_product_scan_pipeline import run_product_scan_path_collapse_pipeline
from lnet.pac_real2d_math import discrete_pole_real2d

Pole = tuple[Tensor, Tensor, Tensor, Tensor]
ComplexField = tuple[Tensor, Tensor]


class Timing(TypedDict):
    forward_ms: list[float]
    backward_ms: list[float]


class _ScanPathBlock(nn.Module):
    def __init__(self, modes: int) -> None:
        super().__init__()
        self.modes = modes
        self.path = PathOnlyCollapse(modes, path_hidden=8)
        damping = torch.linspace(0.04, 0.35, modes).reshape(1, 1, 1, modes)
        for axis, phase_scale in (("x", 0.7), ("y", 0.6)):
            phase = torch.linspace(0.0, phase_scale * torch.pi, modes).reshape(
                1,
                1,
                1,
                modes,
            )
            for name, value in zip(
                ("decay_real", "decay_imag", "gamma_real", "gamma_imag"),
                discrete_pole_real2d(damping, phase, 1.0),
                strict=True,
            ):
                self.register_parameter(f"{axis}_{name}", nn.Parameter(value))

    def _pole(self, axis: str) -> Pole:
        return cast(
            "Pole",
            tuple(
                getattr(self, f"{axis}_{name}")
                for name in ("decay_real", "decay_imag", "gamma_real", "gamma_imag")
            ),
        )

    def forward(self, real: Tensor, imag: Tensor) -> ComplexField:
        return run_product_scan_path_collapse_pipeline(
            self._pole("x"),
            self._pole("y"),
            (real, imag),
            self.path.packed_parameters(),
            gain_normalization="global",
            memory_policy="retain",
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--modes", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sample-groups", type=int, default=7)
    parser.add_argument("--minimum-backward-speedup", type=float, default=1.10)
    args = parser.parse_args()
    if min(args.batch_size, args.height, args.modes, args.warmup, args.sample_groups) <= 0:
        parser.error("benchmark dimensions must be positive")
    return args


def _geometries() -> dict[str, LaunchGeometry]:
    return {
        "bm4_w4": LaunchGeometry.build(
            num_warps=4,
            blocks={"BLOCK_LINES": 1, "BLOCK_MODES": 4},
        ),
        "bm2_w4": LaunchGeometry.build(
            num_warps=4,
            blocks={"BLOCK_LINES": 1, "BLOCK_MODES": 2},
        ),
        "bm2_w8": LaunchGeometry.build(
            num_warps=8,
            blocks={"BLOCK_LINES": 1, "BLOCK_MODES": 2},
        ),
        "bm1_w4": LaunchGeometry.build(
            num_warps=4,
            blocks={"BLOCK_LINES": 1, "BLOCK_MODES": 1},
        ),
        "bm1_w8": LaunchGeometry.build(
            num_warps=8,
            blocks={"BLOCK_LINES": 1, "BLOCK_MODES": 1},
        ),
    }


def _clear(module: nn.Module, source: ComplexField) -> None:
    module.zero_grad(set_to_none=True)
    source[0].grad = None
    source[1].grad = None


def _forward(module: nn.Module, source: ComplexField) -> tuple[ComplexField, Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = cast("ComplexField", module(*source))
        loss = output[0].float().square().mean() + output[1].float().square().mean()
    return output, loss


def _step(module: nn.Module, source: ComplexField) -> tuple[ComplexField, float, float]:
    _clear(module, source)
    start = torch.cuda.Event(enable_timing=True)
    middle = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, loss = _forward(module, source)
    middle.record()
    loss.backward()
    end.record()
    end.synchronize()
    return output, float(start.elapsed_time(middle)), float(middle.elapsed_time(end))


def _gradient_snapshot(module: nn.Module, source: ComplexField) -> dict[str, Tensor]:
    if source[0].grad is None or source[1].grad is None:
        raise RuntimeError("scan benchmark inputs are missing gradients")
    snapshot = {
        "input_real": source[0].grad.detach().clone(),
        "input_imag": source[1].grad.detach().clone(),
    }
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"scan benchmark parameter is missing a gradient: {name}")
        snapshot[f"parameter.{name}"] = parameter.grad.detach().clone()
    return snapshot


def _error(value: Tensor, target: Tensor) -> dict[str, float]:
    difference = value.float() - target.float()
    denominator = target.float().norm().clamp_min(1.0e-12)
    return {
        "relative_l2": float((difference.norm() / denominator).detach()),
        "max_abs": float(difference.abs().max().detach()),
    }


@contextmanager
def _geometry(
    scope: LaunchScope,
    geometry: LaunchGeometry,
) -> Generator[None]:
    with override_launch_geometry(
        PATH_COLLAPSE_BACKWARD_LAUNCH_NAME,
        scope,
        geometry,
    ):
        yield


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("vertical backward benchmark requires CUDA")
    torch.manual_seed(1201)
    module = _ScanPathBlock(args.modes).cuda().train()
    compiled = cast(
        "nn.Module",
        torch.compile(module, mode="default", fullgraph=False, dynamic=False),
    )
    source = cast(
        "ComplexField",
        tuple(
            torch.randn(
                args.batch_size,
                args.height,
                args.height,
                args.modes,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(2)
        ),
    )

    with capture_launch_scopes(PATH_COLLAPSE_BACKWARD_LAUNCH_NAME) as scopes:
        _step(compiled, source)
    if len(scopes) != 1:
        raise RuntimeError(f"expected one path-backward scope, observed {len(scopes)}")
    scope = next(iter(scopes))
    geometries = _geometries()
    errors: dict[str, dict[str, str]] = {}
    active: dict[str, LaunchGeometry] = {}
    for name, geometry in geometries.items():
        try:
            with _geometry(scope, geometry):
                for _ in range(args.warmup):
                    _step(compiled, source)
            active[name] = geometry
        except Exception as error:  # noqa: BLE001 - rejected compile evidence is required.
            errors[name] = {"type": type(error).__name__, "message": str(error)}

    if "bm4_w4" not in active:
        raise RuntimeError("baseline BLOCK_MODES=4 geometry did not compile")

    timings: dict[str, Timing] = {
        name: {"forward_ms": [], "backward_ms": []} for name in active
    }
    names = tuple(active)
    for group in range(args.sample_groups):
        order = names if group % 2 == 0 else tuple(reversed(names))
        for name in order:
            with _geometry(scope, active[name]):
                _, forward_ms, backward_ms = _step(compiled, source)
            timings[name]["forward_ms"].append(forward_ms)
            timings[name]["backward_ms"].append(backward_ms)

    baseline_output: ComplexField
    with _geometry(scope, active["bm4_w4"]):
        baseline_output, _, _ = _step(compiled, source)
    baseline_gradients = _gradient_snapshot(module, source)
    parity: dict[str, dict[str, object]] = {}
    peak_bytes: dict[str, int] = {}
    for name, geometry in active.items():
        torch.cuda.synchronize()
        starting_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with _geometry(scope, geometry):
            output, _, _ = _step(compiled, source)
        peak_bytes[name] = max(0, torch.cuda.max_memory_allocated() - starting_bytes)
        gradients = _gradient_snapshot(module, source)
        output_errors = {
            f"output.{index}": _error(value, target)
            for index, (value, target) in enumerate(zip(output, baseline_output, strict=True))
        }
        gradient_errors = {
            key: _error(value, baseline_gradients[key]) for key, value in gradients.items()
        }
        parity[name] = {
            "outputs": output_errors,
            "gradients": gradient_errors,
            "maximum_relative_l2": max(
                error["relative_l2"]
                for error in (*output_errors.values(), *gradient_errors.values())
            ),
        }

    medians = {
        name: {
            "forward_ms": statistics.median(values["forward_ms"]),
            "backward_ms": statistics.median(values["backward_ms"]),
            "total_ms": statistics.median(
                [
                    forward + backward
                    for forward, backward in zip(
                        values["forward_ms"],
                        values["backward_ms"],
                        strict=True,
                    )
                ]
            ),
        }
        for name, values in timings.items()
    }
    baseline_backward = medians["bm4_w4"]["backward_ms"]
    baseline_memory = peak_bytes["bm4_w4"]
    ratios = {
        name: {
            "backward": values["backward_ms"] / baseline_backward,
            "memory": peak_bytes[name] / max(baseline_memory, 1),
        }
        for name, values in medians.items()
    }
    eligible = [
        name
        for name in active
        if cast("float", parity[name]["maximum_relative_l2"]) <= 2.0e-2
        and ratios[name]["backward"] <= 1.0 / args.minimum_backward_speedup
        and ratios[name]["memory"] <= 1.0
    ]
    winner = min(eligible, key=lambda name: medians[name]["backward_ms"], default=None)
    payload: dict[str, object] = {
        "schema": "lnet.r2k3.vertical_backward_launch.v1",
        "device": torch.cuda.get_device_name(),
        "shape": [args.batch_size, args.height, args.height, args.modes],
        "scope": json.loads(scope.key),
        "geometries": {
            name: {"num_warps": geometry.num_warps, "blocks": geometry.blocks}
            for name, geometry in active.items()
        },
        "compile_errors": errors,
        "timing_samples": timings,
        "medians": medians,
        "incremental_peak_bytes": peak_bytes,
        "parity": parity,
        "ratios": ratios,
        "winner": winner,
        "passed": winner is not None,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
