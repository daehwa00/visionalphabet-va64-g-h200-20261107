"""Recompute a paired VA-family arithmetic ledger from audited counts and live reader shapes.

Retains every non-reader term from the paper's source audit. Thus the revised
whole-model ledger is conservative: it does not claim additional savings from
constant folding, omitted descriptors or coarse scan fusion. MAC-only totals
cover the original registered convolution/matrix operations, not the full model.
"""

from __future__ import annotations

# Canonical builders are script-level imports, as in the training entry points.
# pyright: reportImplicitRelativeImport=false
import argparse
import hashlib
import importlib.util
import json
from math import prod
from pathlib import Path
from typing import TypedDict

import torch
from va_family_inference import build_va_family

from lnet.pac_factorized_complex_scan_reader import FactorizedComplexConv2dReader

ROOT = Path(__file__).resolve().parents[1]


class ReaderShape(TypedDict):
    name: str
    batch: int
    height: int
    width: int
    input_complex_modes: int
    poles: int
    complex_rank: int
    kernel_size: int
    cuda_execution_mode: str
    dense_macs: int
    factorized_macs: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=(32, 48, 64, 96, 128), default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--audit-root", type=Path,
        default=ROOT / "releases/submission/anonymous-code.GBcMHI/visionalphabet/efficiency",
    )
    args = parser.parse_args()
    auditor_path = args.audit_root / "rebuttal_full_flops_audit.py"
    spec = importlib.util.spec_from_file_location("original_forward_audit", auditor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the original source auditor")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    model_key = f"va_k{args.width}"
    original = auditor.audit(model_key, batch=1)
    stored = json.loads((args.audit_root / "rebuttal-full-flops-audit.json").read_text())
    published = next(row for row in stored["models"] if row["model"] == model_key)
    if original["flops_per_image"] != published["flops_per_image"]:
        raise RuntimeError("original audit no longer reproduces the published arithmetic")
    core = json.loads((args.audit_root / f"rebuttal-flops/{model_key}-b1.json").read_text())

    torch.set_num_threads(4)
    torch.manual_seed(809)
    model = build_va_family(args.width).eval()
    checkpoint = args.checkpoint
    if checkpoint is None and args.width == 128:
        checkpoint = ROOT / "reviewer-code/checkpoints/va_k128_seed501_ep100.pt"
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(state, strict=True)
    readers: list[ReaderShape] = []
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, FactorizedComplexConv2dReader):
            continue

        def record(
            reader: torch.nn.Module, inputs: tuple[torch.Tensor, ...], name: str = name,
        ) -> None:
            if not isinstance(reader, FactorizedComplexConv2dReader):
                raise TypeError("unexpected reader hook target")
            batch, height, width, _ = inputs[0].shape
            dense = 4 * batch * height * width * reader.input_modes * reader.output_modes
            dense *= reader.kernel_size**2
            factorized = 4 * batch * height * width * reader.output_modes * reader.rank
            factorized *= reader.input_modes + reader.kernel_size**2
            readers.append({
                "name": name, "batch": batch, "height": height, "width": width,
                "input_complex_modes": reader.input_modes, "poles": reader.output_modes,
                "complex_rank": reader.rank, "kernel_size": reader.kernel_size,
                "cuda_execution_mode": reader.execution_mode,
                "dense_macs": dense, "factorized_macs": factorized,
            })

        handles.append(module.register_forward_pre_hook(record))
    # CPU is used only to observe architecture/shape; its dense fallback is not
    # presented as the optimized execution's profiler count.
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 224, 224))
    for handle in handles:
        handle.remove()
    if len(readers) != 12 or not bool(torch.isfinite(output).all()):
        raise RuntimeError("unexpected reader inventory or nonfinite forward")
    observed = [[row["batch"], row["height"], row["width"], row["poles"]] for row in readers]
    if observed != core["scan_drive_shapes"]:
        raise RuntimeError("current reader shapes differ from the paper's scan inventory")
    if any(row["cuda_execution_mode"] != "factorized_triton" for row in readers):
        raise RuntimeError("a reader is not configured for factorized CUDA execution")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != original["params"]:
        raise RuntimeError("model parameter inventory changed")
    dense_macs = sum(row["dense_macs"] for row in readers)
    factorized_macs = sum(row["factorized_macs"] for row in readers)
    census = json.loads((args.audit_root / f"rebuttal-dispatch-census/{model_key}.json").read_text())
    original_readers = [
        event for event in census["events"]
        if event["op"] == "aten.conv2d.default"
        and event["args"][1]["shape"][1:] == [2 * args.width, 3, 3]
    ]
    original_reader_macs = sum(
        prod(event["output"]["shape"]) * prod(event["args"][1]["shape"][1:])
        for event in original_readers
    )
    if len(original_readers) != len(readers) or original_reader_macs != dense_macs:
        raise RuntimeError("reader formula does not match the original convolution census")
    delta = 2 * (dense_macs - factorized_macs)
    original_full = original["flops_per_image"]
    original_core = original["components"]["core_registry"]
    report = {
        "model": model_key, "parameters": parameters, "batch": 1, "image_size": 224,
        "measurement": "live CPU reader-shape hooks + source formulas; not a new complete GPU profile",
        "checkpoint_sha256": (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint is not None else None
        ),
        "shape_capture_weights": "checkpoint" if checkpoint is not None else "initialization",
        "auditor_sha256": hashlib.sha256(auditor_path.read_bytes()).hexdigest(),
        "original_source_audit": original,
        "reader": {
            "dense_macs": dense_macs, "factorized_macs": factorized_macs,
            "original_convolution_census_macs": original_reader_macs,
            "flops_saved_at_two_per_mac": delta,
            "reduction": 1 - factorized_macs / dense_macs,
            "dense_formula": "4*B*H*W*I*P*K^2",
            "factorized_formula": "4*B*H*W*P*R*(I+K^2)",
        },
        "paper_convention_conservative_ledger": {
            "original_flops": original_full,
            "optimized_flops": original_full - delta,
            "reduction": delta / original_full,
            "scope": "all original non-reader scan/gain/norm/activation/fusion costs retained; only reader convolution execution replaced",
        },
        "conv_matrix_mac_only_ledger": {
            "original_macs": original_core / 2,
            "optimized_macs": (original_core - delta) / 2,
            "reduction": delta / original_core,
            "scope": "registered convolution/matrix MACs only; excludes scan, norm, activation, scalar operations and data movement; NOT total FLOPs",
        },
        "additional_optimization_savings_counted": False,
        "readers": readers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for key in ("reader", "paper_convention_conservative_ledger", "conv_matrix_mac_only_ledger"):
        print(key, json.dumps(report[key]))


if __name__ == "__main__":
    main()
