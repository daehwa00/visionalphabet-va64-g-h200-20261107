"""CPU equivalence checks for logical updates with independent activation microbatches."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import alphabet_imagenet100_runtime as va
import effective_batch_epoch as epoch
import effective_batch_update as update
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional


class ToyHeads(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.heads = nn.ModuleList(nn.Linear(3, 4) for _ in range(5))
        self.classifier = SimpleNamespace(affine_auxiliary_weight=0.3)

    def forward(self, x) -> tuple[torch.Tensor, ...]:
        features = self.bn(x).mean((2, 3))
        values = [head(features) * 8 for head in self.heads]
        return sum(values), *values[1:]


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=0.01, weight_decay=0.1)
        self.steps = 0
        self.zeroes = 0

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.zeroes += 1
        super().zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> torch.Tensor | None:
        self.steps += 1
        return super().step(closure=closure)


def independent_ce(logits, targets, other, weight) -> torch.Tensor:
    first = functional.cross_entropy(logits, targets, reduction="none", label_smoothing=0.1)
    second = functional.cross_entropy(logits, other, reduction="none", label_smoothing=0.1)
    return (weight * first + (1 - weight) * second).mean()


@pytest.mark.parametrize("per_sample", [False, True])
def test_update_matches_independent_microbatch_reference(per_sample, monkeypatch) -> None:
    torch.manual_seed(1024)
    model = ToyHeads().train()
    reference = deepcopy(model)
    x = torch.randn(7, 3, 2, 2)
    y = torch.tensor([0, 1, 3, 2, 1, 0, 2])
    other = y.roll(2)
    weight = torch.tensor([0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.3]) if per_sample else 0.37
    optimizer = CountingAdamW(model.parameters())
    ref_optimizer = CountingAdamW(reference.parameters())
    before = {name: p.detach().clone() for name, p in reference.named_parameters()}
    outputs = {name: [] for name in update.OUTPUT_NAMES}
    losses, joint_losses, affine_losses, correct = [], [], [], []
    ref_optimizer.zero_grad(set_to_none=True)
    for start, end in [(0, 3), (3, 6), (6, 7)]:
        values = reference(x[start:end])
        lam = weight[start:end] if per_sample else weight
        joint = independent_ce(values[0], y[start:end], other[start:end], lam)
        affine = independent_ce(values[1], y[start:end], other[start:end], lam)
        loss = joint + 0.3 * affine
        (loss * ((end - start) / 7)).backward()
        for name, value in zip(update.OUTPUT_NAMES, values, strict=True):
            outputs[name].append(value.detach())
        losses.append(loss.detach() * (end - start))
        joint_losses.append(joint.detach() * (end - start))
        affine_losses.append(affine.detach() * (end - start))
        correct.append(values[0].argmax(-1).eq(y[start:end]).sum())
    grads = {name: p.grad.detach().clone() for name, p in reference.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    assert norm > 1  # Exercise clipping and prove captures contain pre-clipping gradients.
    ref_optimizer.step()
    clip_calls = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def count_clip(parameters, max_norm) -> torch.Tensor:
        clip_calls.append(1)
        return real_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", count_clip)
    result = update.run_update(
        model, model, optimizer, (x, y, other, weight), 3, "float32", capture=True, metrics=True
    )
    assert result["samples"] == 7
    assert result["microbatches"] == 3
    assert result["optimizer_steps"] == optimizer.steps == optimizer.zeroes == len(clip_calls) == 1
    torch.testing.assert_close(result["loss"], sum(losses) / 7)
    torch.testing.assert_close(result["diagnostics"]["joint_ce"], sum(joint_losses) / 7)
    torch.testing.assert_close(result["diagnostics"]["affine_aux_ce"], sum(affine_losses) / 7)
    torch.testing.assert_close(result["correct"], sum(correct))
    torch.testing.assert_close(result["clip_norm"], norm)
    for name, chunks in outputs.items():
        torch.testing.assert_close(result["checks"]["outputs"][name], torch.cat(chunks))
    for name, p in reference.named_parameters():
        torch.testing.assert_close(result["checks"]["grads"][name], grads[name])
        torch.testing.assert_close(result["checks"]["updates"][name], p.detach() - before[name])
        torch.testing.assert_close(dict(model.named_parameters())[name], p)
    for name, buffer in reference.named_buffers():
        torch.testing.assert_close(dict(model.named_buffers())[name], buffer)
    assert model.bn.num_batches_tracked.item() == 3


@pytest.mark.parametrize("weight", [0.43, torch.tensor([0.1, 0.2, 0.7, 0.9])])
def test_mixed_cross_entropy_matches_samplewise_reference(weight) -> None:
    torch.manual_seed(22)
    logits = torch.randn(4, 4, requires_grad=True)
    y = torch.tensor([3, 0, 1, 2])
    other = y.flip(0)
    actual = va._mixed_cross_entropy(logits, y, other, weight)
    expected = independent_ce(logits, y, other, weight)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0]
    )


def test_prepare_and_combine_preserve_each_loader_mixup() -> None:
    torch.manual_seed(17)
    x = torch.randn(5, 3, 2, 2)
    y = torch.tensor([0, 1, 2, 3, 0])
    state = torch.get_rng_state()
    generator = np.random.default_rng(809)
    pieces = [
        epoch.prepare(x[:3], y[:3], generator, 0.8),
        epoch.prepare(x[3:], y[3:], generator, 0.8),
    ]
    torch.set_rng_state(state)
    independent_generator = np.random.default_rng(809)
    expected_pieces = []
    for inputs, targets in [(x[:3], y[:3]), (x[3:], y[3:])]:
        permutation = torch.randperm(len(targets))
        lam = float(independent_generator.beta(0.8, 0.8))
        expected_pieces.append(
            (lam * inputs + (1 - lam) * inputs[permutation], targets, targets[permutation], lam)
        )
    for actual, expected in zip(pieces, expected_pieces, strict=True):
        for a, e in zip(actual[:3], expected[:3], strict=True):
            torch.testing.assert_close(a, e)
        assert actual[3] == expected[3]
    assert pieces[0][3] != pieces[1][3]
    combined = epoch.combine(pieces)
    for index in range(3):
        torch.testing.assert_close(combined[index], torch.cat([p[index] for p in pieces]))
    torch.testing.assert_close(combined[3], torch.tensor([pieces[0][3]] * 3 + [pieces[1][3]] * 2))
    assert epoch.combine(pieces[:1]) is pieces[0]


def test_train_epoch_flushes_partial_group_with_sample_weighted_metrics() -> None:
    model = ToyHeads().eval()
    calls = []

    class Runtime(nn.Module):
        def logical_update(self, batch):
            calls.append(batch)
            n = len(batch[1])
            value = torch.tensor(float(len(calls)))
            return {
                "samples": n,
                "loss": value,
                "correct": torch.tensor(n - 1),
                "diagnostics": {"marker": value * 10},
            }

    runtime = Runtime().eval()
    loader = [(torch.randn(n, 3, 2, 2), torch.arange(n) % 4) for n in [2, 2, 1]]
    result = epoch.train_epoch(
        model,
        runtime,
        loader,
        device=torch.device("cpu"),
        mixup_generator=np.random.default_rng(809),
        mixup_alpha=0.8,
        group_size=2,
        channels_last=False,
    )
    assert model.training
    assert runtime.training
    assert [len(batch[1]) for batch in calls] == [4, 1]
    assert calls[0][3].shape == (4,)
    assert isinstance(calls[1][3], float)
    assert result == {"loss": pytest.approx(1.2), "mixed_accuracy": pytest.approx(0.6)}
    assert model._latest_training_diagnostics == {"marker": pytest.approx(12.0)}


@pytest.mark.parametrize(
    ("extra", "batch", "accum"),
    [
        ([], 128, 8),
        (["--batch-size", "64", "--effective-batch-size", "1024"], 64, 16),
        (["--effective-batch-size", "256"], 128, 2),
        (["--batch-size", "64"], 64, 1),
        (["--batch-size", "32", "--gradient-accumulation-steps", "4"], 32, 4),
        (["--gradient-accumulation-steps", "2"], 128, 2),
        (
            [
                "--batch-size",
                "64",
                "--effective-batch-size",
                "1024",
                "--gradient-accumulation-steps",
                "16",
            ],
            64,
            16,
        ),
    ],
)
def test_cli_effective_batch_defaults_and_explicit_arguments(
    monkeypatch, extra, batch, accum
) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--root", "run", "--data-root", "data", *extra])
    args = va.parse_args(variants=("toy",), seeds=(809,))
    assert (args.batch_size, args.gradient_accumulation_steps) == (batch, accum)


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-size", "0"],
        ["--effective-batch-size", "0"],
        ["--batch-size", "64", "--effective-batch-size", "1000"],
        ["--effective-batch-size", "1024", "--gradient-accumulation-steps", "2"],
    ],
)
def test_cli_rejects_inconsistent_effective_batches(monkeypatch, extra) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--root", "run", "--data-root", "data", *extra])
    with pytest.raises(SystemExit):
        va.parse_args(variants=("toy",), seeds=(809,))


def test_base_contract_disables_precision_cast_emulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--root", "run", "--data-root", "data"])
    monkeypatch.setattr(va.data_runtime, "dataset_digest", lambda _: ("fixture", 1, 1))
    args = va.parse_args(variants=("toy",), seeds=(809,))
    contract = va.base_contract(args, variants=("toy",), seeds=(809,))
    assert contract["recipe"]["emulate_precision_casts"] is False


@pytest.mark.parametrize("compile_mode", ["reduce-overhead", "max-autotune"])
@pytest.mark.parametrize(("scan_flag", "expected_trees"), [(None, True), ("1", True), ("0", False)])
def test_replayed_stages_keep_cudagraph_trees_with_general_scans(
    monkeypatch: pytest.MonkeyPatch,
    scan_flag: str | None,
    compile_mode: str,
    expected_trees: bool,
) -> None:
    model = ToyHeads()
    if scan_flag is None:
        monkeypatch.delenv("ALPHABET_GENERAL_SCANS", raising=False)
    else:
        monkeypatch.setenv("ALPHABET_GENERAL_SCANS", scan_flag)
    monkeypatch.delenv("ALPHABET_COMPILE_MODE", raising=False)
    monkeypatch.setenv("ALPHABET_COMPILE_PROFILE", "")
    monkeypatch.setenv("ALPHABET_CUDAGRAPHS_ACTIVE", "")
    monkeypatch.setenv("ALPHABET_LAUNCH_CACHE", "")
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    monkeypatch.setattr(va, "_has_replayed_cudagraph_stages", lambda _: True)
    options_seen: dict[str, object] = {}

    def compile_stub(
        module: nn.Module,
        *,
        options: dict[str, object],
        **_kwargs: object,
    ) -> nn.Module:
        options_seen.update(options)
        return module

    monkeypatch.setattr(torch, "compile", compile_stub)
    with torch._inductor.config.patch(
        {"emulate_precision_casts": False, "triton.cudagraph_trees": True},
    ):
        assert va._build_runtime(model, {"compile_mode": compile_mode}) is model
        assert (
            options_seen.get(
                "triton.cudagraph_trees",
                torch._inductor.config.triton.cudagraph_trees,
            )
            is expected_trees
        )
    assert options_seen["triton.cudagraphs"] is True
    assert options_seen["emulate_precision_casts"] is False
    assert "post_grad_custom_pre_pass" not in options_seen
