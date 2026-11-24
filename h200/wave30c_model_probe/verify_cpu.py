"""Run CPU Dynamo, inference folding, and cross-revision checkpoint checks."""

# ruff: noqa: S101, T201

import argparse
from pathlib import Path

import torch
from run_va_eight_ablations import build_variant
from torch import Tensor
from torch.nn import Module

from h200.liverms_v2_opt.bench_h200 import VARIANTS
from lnet.image_layers import StandardizedAffineModalHead, fold_terminal_rms_for_inference


def check_dynamo(inputs: Tensor) -> None:
    heads = [(str(dim), StandardizedAffineModalHead(dim, 100), torch.randn(4, dim))
             for dim in (1024, 1280, 1536)]
    models = [(variant, build_variant(variant), inputs) for variant in VARIANTS]
    for label, model, sample in (*heads, *models):
        for mode in ("train", "eval", "folded"):
            model.train(mode == "train")
            if mode == "folded":
                fold_terminal_rms_for_inference(model)
            graph = torch._dynamo.explain(model)(sample)  # noqa: SLF001
            result = [graph.graph_count, graph.graph_break_count]
            print(label, mode, result, flush=True)
            assert result == [1, 0]


@torch.no_grad()
def check_fold(model: Module, inputs: Tensor) -> None:
    model(inputs)
    model.eval()
    head = model.classifier.affine
    descriptors = []
    with head.register_forward_pre_hook(lambda _, args: descriptors.append(args[0])):
        expected = model(inputs)[0]
    (descriptor,) = descriptors
    expected_head = head(descriptor)

    fold_terminal_rms_for_inference(model)
    actual = model(inputs)[0]
    head_diff = (head(descriptor) - expected_head).abs().max().item()
    full_diff = (actual - expected).abs().max().item()

    reloaded = fold_terminal_rms_for_inference(build_variant(VARIANTS[0]).eval())
    reloaded.load_state_dict(model.state_dict(), strict=True)
    reload_diff = (reloaded(inputs)[0] - actual).abs().max().item()
    try:
        reloaded.classifier.affine.train()(descriptor)
    except RuntimeError as error:
        guard = "cannot run folded StandardizedAffineModalHead" in str(error)
    else:
        guard = False
    print(f"FP32 highest head={head_diff:.9g} model={full_diff:.9g} "
          f"strict_reload={reload_diff} folded_train_guard={guard}")
    assert head_diff < 1e-6
    assert full_diff < 1e-6
    assert reload_diff == 0
    assert guard


def check_checkpoint(model: Module, source: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    current = model.state_dict()
    assert len(current) == len(checkpoint) == 197
    for name, param in current.items():
        saved = checkpoint[name]
        assert param.dtype == saved.dtype
        assert param.shape == saved.shape
        assert torch.equal(param.reshape(-1).contiguous().view(torch.uint8),
                           saved.reshape(-1).contiguous().view(torch.uint8)), name
    print(f"strict bytewise reload {len(current)} tensors from {source.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dynamo", "fold", "checkpoint"))
    parser.add_argument("--load", type=Path)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    torch.manual_seed(501)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if args.mode == "checkpoint":
        model = build_variant(VARIANTS[0])
        assert type(model.backbone.input_norm) is torch.nn.RMSNorm
        assert "backbone.input_norm.weight" in model.state_dict()
        if args.load:
            check_checkpoint(model, args.load)
        if args.save:
            torch.save(model.state_dict(), args.save)
        return
    inputs = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    if args.mode == "dynamo":
        check_dynamo(inputs)
    else:
        check_fold(build_variant(VARIANTS[0]), inputs)


if __name__ == "__main__":
    main()
