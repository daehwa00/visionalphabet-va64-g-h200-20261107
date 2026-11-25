#!/usr/bin/env python3
"""Count post-grad CPU Inductor FX copies for the Wave 49 and Wave 50A paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from torch._inductor import config, decomposition
from torch.utils._pytree import tree_leaves

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_reader_e2e_micro import SHAPES, make_reader
from run_va_eight_ablations import build_variant

from lnet.memory_training import enable_memory_training_

MODES = ("off", "dense_nhwc_inductor", "dense_nhwc", "auto_roofline")
VARIANT = "17-p1-nr4-1024d-v3"
COPIES = {"clone", "copy_", "_to_copy", "contiguous", "copy", "permute_copy"}
WATCH = COPIES | {
    "empty_like", "permute", "_unsafe_view", "convolution", "convolution_backward",
    "dense_nhwc_conv2d", "dense_nhwc_conv2d_backward",
}


def count_nodes(graph: torch.fx.Graph, totals: dict[str, dict[str, int]]) -> None:
    for node in graph.nodes:
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            continue
        name = schema.name.split("::")[-1]
        if name not in WATCH:
            continue
        totals[name]["nodes"] += 1
        if name in COPIES:
            for value in tree_leaves(node.meta.get("val")):
                if isinstance(value, torch.Tensor):
                    totals[name]["elements"] += value.numel()
                    totals[name]["bytes"] += value.numel() * value.element_size()
        if name == "permute":
            for user in node.users:
                user_schema = getattr(user.target, "_schema", None)
                if user_schema and user_schema.name.split("::")[-1] in COPIES:
                    totals["permute_to_copy"]["nodes"] += 1


def audit(
    model: torch.nn.Module, sample: torch.Tensor, mode: str, *, backend: str, full_model: bool,
) -> tuple[dict, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    os.environ["LNET_READER_EXEC_MODE"] = mode
    torch._dynamo.reset()
    buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    explanation = torch._dynamo.explain(model)(sample)
    for buffer, initial in buffers:
        buffer.copy_(initial)
    graph_counts = {}
    phase = "forward"

    def record(graph: torch.fx.Graph) -> None:
        phase_counts = graph_counts.setdefault(
            phase, defaultdict(lambda: {"nodes": 0, "elements": 0, "bytes": 0}),
        )
        count_nodes(graph, phase_counts)

    with config.patch(post_grad_custom_post_pass=record, force_disable_caches=True, compile_threads=2):
        # PyTorch 2.9.1's complex add decomposition reinterprets a strided
        # complex tensor as real in the CPU terminal scan backward. Give
        # that decomposition contiguous complex operands for this audit.
        add = torch.ops.aten.add.Tensor
        table = decomposition.select_decomp_table()
        original_add = table[add]

        def strided_complex_add(left: torch.Tensor, right: torch.Tensor, **kwargs: dict) -> object:
            if not all(torch.is_tensor(t) and t.is_complex() for t in (left, right)):
                return NotImplemented
            return original_add(left.contiguous(), right.contiguous(), **kwargs)

        compatibility = (
            patch.dict(table, {add: strided_complex_add})
            if full_model and backend == "inductor" else nullcontext()
        )
        with compatibility:
            compiled = torch.compile(model, backend=backend, fullgraph=True)
            outputs = tuple(v for v in tree_leaves(compiled(sample)) if isinstance(v, torch.Tensor))
            phase = "backward"
            gradients = torch.autograd.grad(outputs[0].square().mean(), tuple(model.parameters()))
    if backend == "inductor" and set(graph_counts) != {"forward", "backward"}:
        raise RuntimeError("Inductor did not emit both graph phases")
    row = {
        "dynamo_graphs": explanation.graph_count,
        "dynamo_breaks": explanation.graph_break_count,
        "dynamo_break_reasons": [reason.reason for reason in explanation.break_reasons],
        "finite_gradients": sum(int(grad.isfinite().all()) for grad in gradients),
        "parameters": len(gradients),
        "parameter_elements": sum(param.numel() for param in model.parameters()),
        **graph_counts,
    }
    return row, tuple(t.detach() for t in outputs), gradients


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("reader", "model", "both"), default="both")
    parser.add_argument("--backend", choices=("inductor", "aot_eager"), default="inductor")
    parser.add_argument("--parity", action="store_true", help="Compare activations and grads to off")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES[:3])
    args = parser.parse_args()
    torch.set_num_threads(2)
    for target in ("reader", "model") if args.target == "both" else (args.target,):
        reference = None
        for mode in args.modes:
            torch.manual_seed(174)
            if target == "reader":
                shape = SHAPES[2]
                model = make_reader(shape, torch.device("cpu"))
                sample = torch.randn(args.batch, shape.height, shape.width, shape.channels)
            else:
                model = build_variant(VARIANT).train()
                enable_memory_training_(model)
                sample = torch.randn(args.batch, 3, 224, 224)
            row, outputs, gradients = audit(
                model, sample.requires_grad_(), mode, backend=args.backend, full_model=target == "model",
            )
            if row["finite_gradients"] != row["parameters"]:
                raise RuntimeError(f"Non-finite gradients in {target}/{mode}")
            if args.parity:
                if mode == "off":
                    reference = outputs, gradients
                if reference is None:
                    parser.error("--parity requires off as the first mode")
                torch.testing.assert_close((outputs, gradients), reference, rtol=2e-3, atol=2e-5)
                for name, actual, expected in zip(
                    ("outputs", "gradients"), (outputs, gradients), reference, strict=True,
                ):
                    row[f"{name}_max_abs"] = max(
                        (a - b).abs().max().item() for a, b in zip(actual, expected, strict=True)
                    )
                    row[f"{name}_bit_equal"] = all(
                        torch.equal(a, b) for a, b in zip(actual, expected, strict=True)
                    )
            print(json.dumps({"target": target, "mode": mode, **row}), flush=True)


if __name__ == "__main__":
    main()
