"""Verify cache-key preservation and BN moment calibration for the Raw G wrapper."""

from __future__ import annotations

import subprocess
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import automatic_training_layout as layout
import calibrated_evaluation
import effective_batch_runtime as effective_rt
import numpy as np
import pytest
import run_va_eight_ablations as base_runner
import run_va_rawg_bn_calibrated as wrapper_runner
import torch
import va_ablation_augment as augment
import va_imagenet100_runtime as runtime
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from lnet.classifier_heads import AffineCorrelationClassifier
from lnet.complex_scan_config import ComplexScanConfig
from lnet.image_layers import StandardizedAffineModalHead


class _ToyRawGModel(nn.Module):
    def __init__(self, in_dim: int = 8, num_classes: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_dim, in_dim)
        self.classifier = AffineCorrelationClassifier(
            in_dim,
            num_classes,
            main="affine",
            affine=StandardizedAffineModalHead(in_dim, num_classes),
            fusion=None,
            lrq=None,
            beta_lrq=None,
            affine_auxiliary_weight=0.0,
        )

    def forward(self, inputs: torch.Tensor):
        descriptor = self.encoder(inputs)
        return self.classifier(descriptor)


def test_wrapper_preserves_exact_layout_and_effective_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("LNET_GENERAL_SCANS", raising=False)
    monkeypatch.delenv("LNET_FUSED_FACTORIZED_READER", raising=False)

    model = _ToyRawGModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.05)
    base_recipe = {
        "batch_size": 128,
        "gradient_accumulation_steps": 8,
        "effective_batch_size": 1024,
        "warmup_epochs": 5,
        "precision": "bfloat16",
        "channels_last": True,
        "mixup_alpha": 0.8,
        "head_objective": "one MixUp cross-entropy on terminal-Q4 affine logits; no auxiliary term",
    }
    wrapper_recipe = {
        **base_recipe,
        "terminal_bn_calibration": "2048 fixed clean training images; temporary eval moments",
    }

    base_bindings = runtime.RunnerBindings(
        variants=base_runner.VARIANTS,
        seeds=base_runner.SEEDS,
        model_config=ComplexScanConfig,
        build_model=base_runner._build,
        contract=base_runner._contract,
        build_optimizer=base_runner._build_optimizer,
        prepare_model=runtime.prepare_model,
        train_epoch=partial(augment.train_epoch, cutmix=bool(base_runner.SPEC["cutmix"])),
        evaluate=runtime.evaluate,
        wandb_model_metrics=runtime.wandb_model_metrics,
        summarize=runtime.summarize,
    )
    wrapper_bindings = wrapper_runner.build_bindings()

    model_cfg = {"output_dim": 100, "stem_strides": (2, 2)}
    seed = 501
    gen = torch.Generator().manual_seed(seed)
    mix_gen = np.random.default_rng(seed)

    for rt_cls in (layout.AutomaticTrainingRuntime, effective_rt.EffectiveTrainingRuntime):
        base_rt = rt_cls(
            model,
            optimizer,
            base_recipe,
            model_cfg,
            base_runner.VARIANT,
            base_bindings,
            tmp_path / "base",
            gen,
            mix_gen,
        )
        wrap_rt = rt_cls(
            model,
            optimizer,
            wrapper_recipe,
            model_cfg,
            base_runner.VARIANT,
            wrapper_bindings,
            tmp_path / "wrap",
            gen,
            mix_gen,
        )
        assert wrap_rt.factories == base_rt.factories
        assert wrap_rt.schema == base_rt.schema
        assert wrap_rt.identity == base_rt.identity
        assert layout.digest(wrap_rt.identity) == layout.digest(base_rt.identity)
        assert wrap_rt.identity["scan_env"] == {
            "LNET_GENERAL_SCANS": "1",
            "LNET_FUSED_FACTORIZED_READER": "1",
        }


def test_calibrated_evaluation_uses_clean_moments_and_restores_buffers() -> None:
    torch.manual_seed(123)
    model = _ToyRawGModel(in_dim=8, num_classes=4)
    model.eval()
    bn = model.classifier.affine.standardizer
    assert bn is not None
    orig_mean = torch.full_like(bn.running_mean, 42.0)
    orig_var = torch.full_like(bn.running_var, 0.05)
    bn.running_mean.copy_(orig_mean)
    bn.running_var.copy_(orig_var)

    cal_x = torch.randn(32, 8) * 2.0 + 3.0
    cal_y = torch.zeros(32, dtype=torch.long)
    val_x = torch.randn(16, 8) * 2.0 + 3.0
    val_y = torch.randint(0, 4, (16,))

    cal_loader = DataLoader(TensorDataset(cal_x, cal_y), batch_size=8, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=8, shuffle=False)

    metrics = calibrated_evaluation.evaluate(
        model,
        model,
        val_loader,
        torch.device("cpu"),
        calibration_loader=cal_loader,
        precision="float32",
        channels_last=False,
    )

    # Buffers must be restored bit-for-bit after evaluation.
    assert torch.equal(bn.running_mean, orig_mean)
    assert torch.equal(bn.running_var, orig_var)

    # Metrics must match exact manual calibration with unbiased variance.
    with torch.no_grad():
        cal_desc = model.encoder(cal_x).float()
        val_desc = model.encoder(val_x).float()
        cal_var, cal_mean = torch.var_mean(cal_desc, dim=0, unbiased=True)
        expected_logits = model.classifier.affine.linear(
            functional.batch_norm(val_desc, cal_mean, cal_var, training=False, eps=bn.eps)
        )
        expected_ce = float(functional.cross_entropy(expected_logits, val_y))
        expected_acc = float(expected_logits.argmax(dim=-1).eq(val_y).float().mean())

    metric_atol = 1e-6
    assert metrics["cross_entropy"] == pytest.approx(expected_ce, abs=metric_atol)
    assert metrics["accuracy"] == pytest.approx(expected_acc, abs=metric_atol)
    assert metrics["affine_reconstruction_max_error"] == pytest.approx(0.0, abs=metric_atol)


def test_calibrated_evaluation_restores_buffers_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ToyRawGModel(in_dim=8, num_classes=4)
    bn = model.classifier.affine.standardizer
    assert bn is not None
    orig_mean = torch.linspace(-1.0, 1.0, 8)
    orig_var = torch.linspace(0.5, 2.0, 8)
    bn.running_mean.copy_(orig_mean)
    bn.running_var.copy_(orig_var)

    cal_loader = DataLoader(
        TensorDataset(torch.randn(16, 8), torch.zeros(16, dtype=torch.long)),
        batch_size=8,
    )

    def _boom(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise RuntimeError("injected eval failure")

    monkeypatch.setattr(runtime, "evaluate", _boom)
    with pytest.raises(RuntimeError, match="injected eval failure"):
        calibrated_evaluation.evaluate(
            model,
            model,
            cal_loader,
            torch.device("cpu"),
            calibration_loader=cal_loader,
            precision="float32",
        )

    assert torch.equal(bn.running_mean, orig_mean)
    assert torch.equal(bn.running_var, orig_var)


def test_wrapper_cli_help_and_contract_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "run_va_rawg_bn_calibrated.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo,
        env={"PYTHONPATH": f"{repo / 'src'}:{repo / 'scripts'}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    monkeypatch.setattr(
        runtime.data_runtime,
        "dataset_digest",
        lambda _root: ("deadbeef", 128000, 5000),
    )
    bindings = wrapper_runner.build_bindings()
    args = SimpleNamespace(
        root=tmp_path,
        data_root=tmp_path,
        epochs=100,
        batch_size=128,
        gradient_accumulation_steps=8,
        workers=0,
        precision="bfloat16",
        initialize_only=True,
    )
    contract = bindings.contract(args)
    assert contract["variants"] == list(base_runner.VARIANTS)
    assert "terminal_bn_calibration" in contract["recipe"]
    assert "calibrated_wrapper" in contract["source_sha256"]
    assert "calibrated_evaluation" in contract["source_sha256"]
