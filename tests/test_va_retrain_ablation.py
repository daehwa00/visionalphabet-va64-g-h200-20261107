from __future__ import annotations

import argparse
import copy
import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

import pytest
import torch
from torch import nn

from lnet.complex_scan_stage import ComplexScanStage
from lnet.pac_mean_energy_readout import mean_direction_energy, mean_energy_product_scan
from lnet.pac_product_scan_reference import product_scan_full4_reference
from lnet.pac_real2d_math import discrete_pole_real2d
from scripts import run_va_retrain_imagenet1k as runner
from scripts.queue_va_retrain import aggregate


@pytest.fixture(autouse=True)
def threads() -> Generator[None]:
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_mean_energy_mathematics() -> None:
    real = torch.tensor([1.0, 3.0]).reshape(1, 1, 1, 2, 1).requires_grad_()
    imaginary = torch.zeros_like(real)
    out = mean_direction_energy(real, imaginary)
    torch.testing.assert_close(out, torch.tensor([[math.log(6)]]))
    out.sum().backward()
    torch.testing.assert_close(real.grad.flatten(), torch.tensor([1 / 6, 0.5]))
    assert not torch.allclose(out, torch.log1p(real.detach().square()).mean((1, 2, 3)))
    cancellation = torch.tensor([1.0, -1.0]).reshape_as(real)
    torch.testing.assert_close(
        mean_direction_energy(cancellation, imaginary), torch.tensor([[math.log(2)]])
    )
    torch.testing.assert_close(mean_direction_energy(real.flip(3), imaginary), out)
    zeros = torch.zeros_like(real)
    assert mean_direction_energy(zeros, zeros).eq(0).all()
    single = real[:, :, :, :1]
    torch.testing.assert_close(
        mean_direction_energy(single, single * 0),
        mean_direction_energy(single.expand(-1, -1, -1, 4, -1), torch.zeros(1, 1, 1, 4, 1)),
    )


def scan_inputs(device: str, dtype: torch.dtype = torch.float32) -> tuple:
    torch.manual_seed(31)
    poles = []
    for phase in (0.7, -0.3):
        damping = torch.full((1, 1, 1, 16), 0.2, device=device)
        angle = torch.full_like(damping, phase)
        poles.append(
            tuple(t.detach().requires_grad_() for t in discrete_pole_real2d(damping, angle, 1.0))
        )
    source = tuple(
        torch.randn(2, 5, 7, 16, device=device, dtype=dtype, requires_grad=True) for _ in range(2)
    )
    return poles[0], poles[1], source


@pytest.mark.parametrize("memory", ["retain", "recompute"])
@pytest.mark.parametrize("normalization", ["pointwise", "global"])
def test_mean_scan_reference_gradients(memory: str, normalization: str) -> None:
    inputs = scan_inputs("cpu")
    flat = (*inputs[0], *inputs[1], *inputs[2])
    clone = tuple(t.detach().clone().requires_grad_() for t in flat)
    expected_states = product_scan_full4_reference(
        clone[:4], clone[4:8], clone[8:], gain_normalization=normalization
    )
    expected = mean_direction_energy(*expected_states)
    actual = mean_energy_product_scan(
        *inputs, memory_policy=memory, gain_normalization=normalization
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected_grad = torch.autograd.grad(expected.sum(), clone)
    actual_grad = torch.autograd.grad(actual.sum(), flat)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_b_original_regression_and_shared_initialization() -> None:
    torch.manual_seed(0)
    original = runner.anchor._build(
        runner.anchor.D2262_P128_TERM, runner.ancestry.model_config(output_dim=1000)
    )
    original = runner.PrimaryLogitsAdapter(original).eval()
    torch.manual_seed(0)
    baseline = runner.build_model("B", prepared=False).eval()
    for name, value in original.state_dict().items():
        assert torch.equal(value, baseline.state_dict()[name]), name
    torch.manual_seed(0)
    averaged = runner.build_model("A", prepared=False).eval()
    for name, value in baseline.state_dict().items():
        if ".classifier." not in name:
            assert torch.equal(value, averaged.state_dict()[name]), name
    assert averaged.model.classifier.q4_dim == 128
    assert runner.parameter_inventory(averaged)["total"]["stored_elements"] == 4_699_176
    x = torch.randn(2, 3, 32, 32)
    target = torch.tensor([1, 2])
    loss0 = nn.functional.cross_entropy(original(x), target)
    loss1 = nn.functional.cross_entropy(baseline(x), target)
    torch.testing.assert_close(loss0, loss1, rtol=0, atol=0)
    loss0.backward()
    loss1.backward()
    for (_name, a), (_, b) in zip(
        original.named_parameters(), baseline.named_parameters(), strict=True
    ):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    out = averaged(x)
    assert out.shape == (2, 1000)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert averaged.model.terminal.phase_x.grad is not None


def test_phase_zero_constraint_and_forced_reference() -> None:
    torch.manual_seed(0)
    model = runner.build_model("P", prepared=False)
    runner.assert_zero_phases(model)
    inventory = runner.parameter_inventory(model)
    assert inventory["phase"]["stored_elements"] == 3712
    assert inventory["phase"]["trainable_elements"] == 0
    assert inventory["total"]["trainable_elements"] == 5_079_464
    stage = ComplexScanStage(16, maximum_phase=1.0, output_modes=None)
    forced = copy.deepcopy(stage)
    stage.phase_mode = "zero"
    with torch.no_grad():
        forced.phase_x.zero_()
        forced.phase_y.zero_()
    for a, b in zip(
        stage.pole_coefficients((2, 5, 7, 16)), forced.pole_coefficients((2, 5, 7, 16)), strict=True
    ):
        for aa, bb in zip(a, b, strict=True):
            torch.testing.assert_close(aa, bb, rtol=0, atol=0)
    optimizer = runner.runtime._build_optimizer(model, 0.003, torch.device("cpu"))
    assert not any(
        p is q
        for group in optimizer.param_groups
        for p in group["params"]
        for name, q in model.named_parameters()
        if name.endswith((".phase_x", ".phase_y"))
    )
    stage = next(m for m in model.modules() if isinstance(m, ComplexScanStage))
    ref = copy.deepcopy(stage)
    ref.phase_mode = "learned"
    shape = (2, 4, 4, stage.input_modes)
    x = tuple(torch.randn(shape, requires_grad=True) for _ in range(2))
    y = tuple(t.detach().clone().requires_grad_() for t in x)
    left = stage(*x)[1]
    right = ref(*y)[1]
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    for a, b in zip(
        torch.autograd.grad(left.sum(), x), torch.autograd.grad(right.sum(), y), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mean_scan_gpu_cpu_parity(dtype: torch.dtype) -> None:
    inputs = scan_inputs("cuda", dtype)
    flat = (*inputs[0], *inputs[1], *inputs[2])
    cpu = tuple(t.detach().cpu().requires_grad_() for t in flat)
    expected = mean_energy_product_scan(cpu[:4], cpu[4:8], cpu[8:])
    actual = mean_energy_product_scan(*inputs, memory_policy="recompute")
    tol = 0.025 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(actual.cpu(), expected, rtol=tol, atol=tol)
    a = torch.autograd.grad(actual.sum(), flat)
    b = torch.autograd.grad(expected.sum(), cpu)
    for aa, bb in zip(a, b, strict=True):
        torch.testing.assert_close(aa.cpu(), bb, rtol=tol, atol=tol)


def test_checkpoint_mode_binding_and_head_shape_rejected(tmp_path: Path) -> None:

    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps({"root": str(tmp_path.resolve()), "counts": {"train": 1281167, "val": 50000}})
    )
    args = argparse.Namespace(
        data_manifest=manifest,
        data_root=tmp_path,
        variant="B",
        seed=0,
        output_root=tmp_path / "runs",
        batch_size=128,
        workers=0,
        resume=False,
    )
    baseline = runner.make_task(args)
    contract = runner.runtime._contract(baseline, torch.device("cpu"))
    args.variant = "P"
    other = runner.make_task(args)
    mismatch = runner.runtime._contract(other, torch.device("cpu"))
    payload = {
        key: contract[key] for key in ("task_sha256", "source_digest_sha256", "source_sha256")
    }
    payload["schema"] = runner.runtime.SCHEMA
    payload["contract_sha256"] = runner.runtime._sha256_payload(contract)
    runner.runtime._validate_binding(payload, contract)
    with pytest.raises(RuntimeError):
        runner.runtime._validate_binding(payload, mismatch)
    a = nn.Linear(512, 1000)
    b = nn.Linear(128, 1000)
    with pytest.raises(RuntimeError):
        b.load_state_dict(a.state_dict(), strict=True)


def test_queue_aggregation_preserves_missing_and_paired_seeds(tmp_path: Path) -> None:

    for variant, scores in {"B": [75.0, 74.8, 75.2], "P": [74.0, 73.8, 74.2]}.items():
        for seed, score in enumerate(scores):
            path = tmp_path / "runs" / f"va_k128_retrain_{variant}" / f"seed_{seed}"
            path.mkdir(parents=True)
            (path / "result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "completed_epochs": 100,
                        "training_seconds": 10.0,
                        "final_validation": {"accuracy": score / 100, "top5_accuracy": 0.92},
                    }
                )
            )
    summary = aggregate(tmp_path)
    assert summary["completed"] == 6
    assert summary["groups"]["A"]["mean"] is None
    assert summary["paired"]["B-A"]["t95_ci"] is None
    assert summary["paired"]["B-P"]["mean_pp"] == pytest.approx(1.0)
    assert not (tmp_path / "rebuttal-results.en.md").exists()
