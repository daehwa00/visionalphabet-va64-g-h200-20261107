"""Verify live RMS, multiscale terminal readout, and deployment exports."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import alphabet_imagenet100_runtime as runtime
import alphabet_model_builder as builder
import calibrated_evaluation as calibrated
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from alphabet.image_layers import (
    LivePolewiseComplexRMS,
    StandardizedAffineModalHead,
    fold_terminal_rms_for_inference,
)
from alphabet.terminal_imaginary_correlation import complex_correlation_descriptor

if TYPE_CHECKING:
    from pathlib import Path


def _v1_scale(ms: Tensor) -> Tensor:
    pole = 0.3 / (ms + 1e-24).sqrt()
    return torch.cat((pole, pole))


@pytest.mark.parametrize("dim", [1024, 1280, 1536])
def test_rms_v1_parity(dim: int) -> None:
    generator = torch.Generator().manual_seed(dim)
    rms = LivePolewiseComplexRMS(dim)
    expected_ms = torch.zeros(dim // 2)
    expected_weight = torch.zeros(())
    for _ in range(3):
        inputs = torch.randn(5, dim, generator=generator, requires_grad=True)
        upstream = torch.randn(5, dim, generator=generator)
        real, imag = inputs.chunk(2, dim=-1)
        batch_ms = (real.square() + imag.square()).mean(0)
        expected = inputs * _v1_scale(batch_ms)
        actual = rms(inputs)
        actual_grad = torch.autograd.grad(actual, inputs, upstream)[0]
        expected_grad = torch.autograd.grad(expected, inputs, upstream)[0]
        expected_ms.mul_(0.95).add_(batch_ms.detach(), alpha=0.05)
        expected_weight = expected_weight * 0.95 + 0.05
        for left, right in (
            (actual, expected),
            (actual_grad, expected_grad),
            (rms.running_ms, expected_ms),
            (rms.ema_weight, expected_weight),
        ):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=0)
    expected = inputs * _v1_scale((expected_ms / expected_weight).clamp_min(0))
    torch.testing.assert_close(rms.eval()(inputs), expected, atol=1e-6, rtol=0)


def test_rms_non_contiguous_and_rank_three() -> None:
    inputs = torch.randn(5, 3, 512).transpose(0, 1)[..., ::2]
    assert not inputs.is_contiguous()
    rms = LivePolewiseComplexRMS(256)
    contiguous = deepcopy(rms)
    torch.testing.assert_close(rms(inputs), contiguous(inputs.contiguous()), rtol=0, atol=0)
    expected = LivePolewiseComplexRMS(256)(inputs.flatten(0, 1))
    torch.testing.assert_close(rms(inputs), expected.reshape_as(inputs))
    with pytest.raises(ValueError, match="matching input_dim"):
        rms(inputs[0, 0])
    with pytest.raises(ValueError, match="matching input_dim"):
        rms(inputs[..., :-1])
    for dim in (0, 1, -2):
        with pytest.raises(ValueError, match="positive even"):
            LivePolewiseComplexRMS(dim)


def test_head_folding_and_strict_reload() -> None:
    for dim in (8, 255, 257):
        assert isinstance(StandardizedAffineModalHead(dim, 7).standardizer, nn.BatchNorm1d)
    head = StandardizedAffineModalHead(256, 7)
    assert isinstance(head.standardizer, LivePolewiseComplexRMS)
    inputs = torch.randn(5, 256)
    head(inputs)
    expected = head.eval()(inputs)
    assert head.fold_standardizer_into_linear() is head
    weight = head.linear.weight.detach().clone()
    head.fold_standardizer_into_linear()
    torch.testing.assert_close(head.linear.weight, weight, rtol=0, atol=0)
    model = nn.Sequential(StandardizedAffineModalHead(256, 7)).eval()
    assert fold_terminal_rms_for_inference(model) is model
    model.load_state_dict(nn.Sequential(head).state_dict(), strict=True)
    torch.testing.assert_close(model(inputs), expected, atol=1e-5, rtol=0)
    with pytest.raises(RuntimeError, match="cannot run folded StandardizedAffineModalHead"):
        model.train()(inputs)


@pytest.mark.parametrize("output_dim", [100, 7])
def test_multiscale_model(output_dim: int) -> None:
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        base = builder.build_dw_terminal_rmsnorm(
            builder.model_config(output_dim=output_dim), factorized_reader=True
        )
        terminal = base.backbone.terminal
        terminal.pole_input_projection._real_training_vector_gradient = False
        rng = torch.get_rng_state()
        model = builder.add_multiscale_readout(base)
        assert torch.equal(torch.get_rng_state(), rng)
        assert isinstance(model, builder.MultiScaleVisionAlphabetIN100)
        assert sum(p.numel() for p in model.parameters()) == 3_114_852 + (output_dim - 100) * 1537
        terminal_s3 = model.backbone.terminal_s3
        for flag in ("_real_training_autotune", "_real_training_vector_gradient"):
            assert getattr(terminal_s3.pole_input_projection, flag) == getattr(
                terminal.pole_input_projection, flag
            )
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))
        assert output[0].shape == (2, output_dim)
        assert output[-1].shape == (2, 1536)
        optimizer = builder.build_optimizer(
            model,
            {
                "learning_rate": 3e-3,
                "weight_decay": 0.05,
                "modal_learning_rate_multiplier": 1.0,
                "pole_geometry_learning_rate_multiplier": 1.0,
            },
        )
        seen = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert seen == {id(p) for p in model.parameters()}
        assert {id(p) for p in terminal_s3.parameters()} <= seen
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("normalize", [False, True])
def test_forward_vertical_readout_matches_existing_direction_order(normalize: bool) -> None:
    px = tuple(torch.rand(3) * 0.2 for _ in range(4))
    py = tuple(torch.rand(3) * 0.2 for _ in range(4))
    drive = torch.randn(2, 5, 4, 6)
    full = complex_correlation_descriptor(px, py, drive, normalize=normalize)
    forward = complex_correlation_descriptor(
        px, py, drive, normalize=normalize, forward_vertical_only=True
    )
    expected = torch.cat((full[..., :6], full[..., 12:18]), dim=-1)
    torch.testing.assert_close(forward, expected, rtol=0, atol=0)


class _ToyLiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = builder.TerminalAffineClassifier(256, 256, 3)

    def forward(self, inputs: Tensor) -> tuple[Tensor, ...]:
        return self.classifier(inputs, terminal_only=True)


def _datasets() -> tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(117)
    train = TensorDataset(torch.randn(7, 256, generator=generator), torch.arange(7) % 3)
    validation = TensorDataset(torch.randn(4, 256, generator=generator), torch.arange(4) % 3)
    return train, validation


def _mock_clean_split(monkeypatch: pytest.MonkeyPatch, root: Path, dataset: TensorDataset) -> None:
    clean_transform = object()

    def image_folder(path: Path, transform: object) -> TensorDataset:
        assert path == root / "train"
        assert transform is clean_transform
        return dataset

    monkeypatch.setattr(
        calibrated.data, "transforms_for_imagenet100", lambda: (None, clean_transform)
    )
    monkeypatch.setattr(calibrated, "ImageFolder", image_folder)


def test_dual_evaluate_and_full_freeze(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train, validation = _datasets()
    train_loader = DataLoader(train, batch_size=3)
    val_loader = DataLoader(validation, batch_size=2)
    device = torch.device("cpu")
    model = _ToyLiveModel()
    model(train.tensors[0][:3])
    std = model.classifier.affine.standardizer
    std.running_mean.fill_(0.25)
    saved = {name: value.clone() for name, value in std.named_buffers()}
    expected_stored = runtime.evaluate(model, model, val_loader, device, precision="float32")
    _mock_clean_split(monkeypatch, tmp_path, train)
    log_path = tmp_path / "dual.jsonl"
    monkeypatch.setenv("ALPHABET_DUAL_EVAL_LOG", str(log_path))
    monkeypatch.delenv("ALPHABET_DUAL_EVAL_LOG_SKIP", raising=False)

    def dual_evaluate() -> dict[str, float]:
        return calibrated.evaluate(
            model, model, val_loader, device, calibration_loader=train_loader, precision="float32"
        )

    calls = []
    hook = model.register_forward_hook(lambda *_args: calls.append(1))
    try:
        metrics = dual_evaluate()
    finally:
        hook.remove()
    assert len(calls) == len(train_loader) + len(val_loader)
    record = json.loads(log_path.read_text())
    assert record["calibrated_rms_eval"] == metrics
    assert record["stored_rms_eval"] == pytest.approx(expected_stored, abs=1e-6)
    for name, value in std.named_buffers():
        assert torch.equal(value, saved[name])

    # Logging fails after the temporary moments have been installed.
    monkeypatch.setenv("ALPHABET_DUAL_EVAL_LOG", str(tmp_path / "missing" / "dual.jsonl"))
    with pytest.raises(FileNotFoundError):
        dual_evaluate()
    for name, value in std.named_buffers():
        assert torch.equal(value, saved[name])
    monkeypatch.setenv("ALPHABET_DUAL_EVAL_LOG_SKIP", "1")
    assert dual_evaluate() == metrics
    monkeypatch.delenv("ALPHABET_DUAL_EVAL_LOG_SKIP")
    monkeypatch.delenv("ALPHABET_DUAL_EVAL_LOG")
    monkeypatch.chdir(tmp_path)
    assert dual_evaluate() == metrics
    assert not (tmp_path / "dual-eval.jsonl").exists()

    ms, count, stats = calibrated.freeze_full_dataset_rms(
        model,
        model,
        tmp_path,
        device,
        batch_size=3,
        workers=0,
        precision="float32",
        channels_last=False,
    )
    real, imag = train.tensors[0].double().chunk(2, dim=-1)
    expected_ms = (real.square() + imag.square()).mean(0).float()
    torch.testing.assert_close(ms, expected_ms, rtol=0, atol=0)
    assert count == stats["sample_count"] == 7
    assert stats["poles"] == 128
    assert torch.equal(std.running_ms, expected_ms)
    assert std.ema_weight.item() == 1
    assert torch.count_nonzero(std.running_mean).item() == 0
    frozen = runtime.evaluate(model, model, val_loader, device, precision="float32")
    assert frozen == pytest.approx(metrics, abs=1e-6)


def test_final_epoch_rms_exports_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train, validation = _datasets()
    loaders = DataLoader(train, batch_size=7), DataLoader(validation, batch_size=2)
    _mock_clean_split(monkeypatch, tmp_path, train)
    monkeypatch.setattr(runtime.data_runtime, "loaders", lambda *_args, **_kwargs: loaders)
    monkeypatch.delenv("ALPHABET_COMPILE_MODE", raising=False)
    summary: dict[str, Any] = {}
    tracking = SimpleNamespace(summary=summary, log=lambda *_a, **_kw: None, finish=lambda: None)
    monkeypatch.setattr(runtime, "_initialize_wandb_run", lambda *_args, **_kwargs: tracking)

    def train_epoch(
        model: nn.Module,
        engine: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        **_kwargs: object,
    ) -> dict[str, float]:
        model.train()
        for inputs, labels in loader:
            loss = functional.cross_entropy(engine(inputs)[0], labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return {"loss": loss.item(), "mixed_accuracy": 0.0}

    # Distinct metrics let the test distinguish raw frozen validation from bindings.evaluate.
    original_metrics = {"accuracy": 0.25, "cross_entropy": 12.0}
    bindings = runtime.RunnerBindings(
        variants=("live",),
        seeds=(501,),
        model_config=lambda: None,
        build_model=lambda _variant, _config: _ToyLiveModel(),
        build_optimizer=lambda model, _recipe: torch.optim.AdamW(model.parameters()),
        contract=lambda _args: {},
        prepare_model=lambda model, _recipe: model,
        train_epoch=train_epoch,
        evaluate=lambda *_args, **_kwargs: original_metrics,
        wandb_model_metrics=lambda _model: {},
        summarize=lambda _root, _contract: None,
    )
    contract = {
        "model": {},
        "recipe": {"epochs": 1, "batch_size": 7, "mixup_alpha": 0.0, "precision": "float32"},
    }
    runtime._run_job(
        tmp_path,
        contract,
        variant="live",
        seed=501,
        data_root=tmp_path,
        workers=0,
        device=torch.device("cpu"),
        bindings=bindings,
    )
    stem = tmp_path / "checkpoints" / "live__seed501"
    checkpoint = torch.load(stem.with_suffix(".pt"), weights_only=False)
    assert checkpoint["global_step"] == checkpoint["epoch"] == 1
    assert {
        "optimizer",
        "scheduler",
        "history",
        "training_seconds",
        "training_generator_state",
        "torch_rng_state",
        "cuda_rng_state",
        "python_rng_state",
        "mixup_rng_state",
    } <= checkpoint.keys()
    resumed = _ToyLiveModel()
    resumed.load_state_dict(checkpoint["model"], strict=True)
    assert resumed.classifier.affine.standardizer.ema_weight.item() == pytest.approx(0.05)

    clean_export = torch.load(stem.with_name(stem.name + "_frozen_clean_rms.pt"))
    folded_export = torch.load(stem.with_name(stem.name + "_folded_inference.pt"))
    clean = _ToyLiveModel().eval()
    clean.load_state_dict(clean_export["model"], strict=True)
    folded = fold_terminal_rms_for_inference(_ToyLiveModel().eval())
    folded.load_state_dict(folded_export["model"], strict=True)
    assert folded_export["folded_standardizer"] is True
    inputs = validation.tensors[0]
    torch.testing.assert_close(clean(inputs)[0], folded(inputs)[0], rtol=0, atol=1e-6)

    result = json.loads((tmp_path / "results" / "live__seed501.json").read_text())
    expected = runtime.evaluate(clean, clean, loaders[1], torch.device("cpu"), precision="float32")
    assert result["final_validation"] == original_metrics
    assert result["frozen_clean_rms_validation"] == expected
    assert summary["frozen_clean_rms_validation"] == expected
    assert result["frozen_pole_rms"] == clean_export["frozen_pole_rms"]
    assert result["frozen_pole_rms"] == summary["frozen_pole_rms"]
    assert result["frozen_pole_rms"]["samples"] == len(train)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for graph capture check")
@pytest.mark.parametrize("dim", [1024, 1280, 1536])
def test_dynamo_single_graph_on_cuda(dim: int) -> None:
    head = StandardizedAffineModalHead(dim, 100).cuda()
    inputs = torch.randn(4, dim, device="cuda")
    for mode in ("train", "eval", "folded"):
        head.train(mode == "train")
        if mode == "folded":
            head.fold_standardizer_into_linear()
        expl = torch._dynamo.explain(head)(inputs)
        assert [expl.graph_count, expl.graph_break_count] == [1, 0]
