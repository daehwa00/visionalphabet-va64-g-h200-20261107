"""CPU-only Wave 35C fullgraph, checkpoint, and folding verification."""

# ruff: noqa: S101, SLF001, T201
import argparse
import hashlib
from pathlib import Path

import torch
from run_va_eight_ablations import build_variant

from lnet.image_layers import StandardizedAffineModalHead, fold_terminal_rms_for_inference

VARIANTS = ("17-p1-nr4-1024d-v3", "17-live-complex-rms-s03", "17-e2-multiscale-s3s4-live-rms-v2")


def digest(state: dict[str, torch.Tensor]) -> str:
    value = hashlib.sha256()
    for key, tensor in state.items():
        value.update(repr((key, tensor.shape, tensor.dtype)).encode())
        value.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return value.hexdigest()


def graph(model: torch.nn.Module, inputs: torch.Tensor) -> list[int]:
    result = torch._dynamo.explain(model)(inputs)
    counts = [result.graph_count, result.graph_break_count]
    if counts != [1, 0]:
        raise AssertionError(str(result.break_reasons))
    return counts


def verify_models(reference: Path) -> None:
    sample = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    for key in VARIANTS:
        torch.manual_seed(501)
        model = build_variant(key)
        baseline = torch.load(reference / f"{key}.pt", weights_only=True)
        model.load_state_dict(baseline, strict=True)
        assert digest(model.state_dict()) == digest(baseline)
        print(key, len(baseline), digest(baseline), "checkpoint strict byte equality", flush=True)
        print(key, "train", graph(model, sample), flush=True)
        with torch.no_grad():
            model.eval()
            print(key, "eval", graph(model, sample), flush=True)
            expected = model(sample)[0]
            fold_terminal_rms_for_inference(model)
            print(key, "folded", graph(model, sample), flush=True)
            output = model(sample)[0]
            reloaded = fold_terminal_rms_for_inference(build_variant(key).eval())
            reloaded.load_state_dict(model.state_dict(), strict=True)
            parity = (output - expected).abs().max().item()
            reload_delta = (output - reloaded(sample)[0]).abs().max().item()
            try:
                width = model.classifier.affine.input_dim
                reloaded.classifier.affine.train()(torch.zeros(2, width))
            except RuntimeError as error:
                guard = "cannot run folded StandardizedAffineModalHead" in str(error)
            else:
                guard = False
            print(key, "fold", parity, reload_delta, guard, flush=True)
            assert parity <= 1e-6
            assert reload_delta == 0
            assert guard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    for width in (1024, 1280, 1536):
        head = StandardizedAffineModalHead(width, 100)
        inputs = torch.randn(4, width)
        print(width, "train", graph(head, inputs), flush=True)
        with torch.no_grad():
            head.eval()
            print(width, "eval", graph(head, inputs), flush=True)
            head.fold_standardizer_into_linear()
            print(width, "folded", graph(head, inputs), flush=True)
    verify_models(args.reference)


if __name__ == "__main__":
    main()
