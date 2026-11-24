"""Compile the full training graph and inspect both AOT graphs."""

import json

import torch
from run_va_eight_ablations import build_variant
from torch._dynamo.backends.debugging import boxed_nop
from torch._dynamo.backends.registry import lookup_backend

from lnet.model_w36b import model_w36b_features, prepare_model_w36b_


def main() -> None:
    torch.manual_seed(501)
    model = prepare_model_w36b_(build_variant("17-p1-nr4-1024d-v3"))
    image = torch.randn(2, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    matrix_exp = []
    ortho = []

    def inspect(graph: torch.fx.GraphModule) -> None:
        ortho.extend(str(node.target) for node in graph.graph.nodes
                     if str(node.target) == "lnet.ortho_mm.default")
        matrix_exp.extend(str(node.target) for node in graph.graph.nodes
                          if "linalg_matrix_exp" in str(node.target))

    backend = lookup_backend("aot_eager")

    def checked_backend(graph: torch.fx.GraphModule, inputs: list) -> object:
        inspect(graph)
        return backend(graph, inputs, fw_compiler=checked_aot, bw_compiler=checked_aot)

    def checked_aot(graph: torch.fx.GraphModule, inputs: list) -> object:
        inspect(graph)
        return boxed_nop(graph, inputs)

    compiled = torch.compile(model, backend=checked_backend, fullgraph=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(compiled(image)[0], torch.arange(2))
    loss.backward()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
    finite_grads = sum(bool(grad.isfinite().all()) for grad in grads)
    print(json.dumps({"loss": loss.item(), "grads": finite_grads,  # noqa: T201
                      "linalg_matrix_exp": len(matrix_exp), "ortho_mm": len(ortho)}))
    if not loss.isfinite() or len(grads) != 193 or finite_grads != 193:
        raise RuntimeError("Expected finite loss and 193 finite gradients")
    if "capture_safe_ortho" in model_w36b_features() and (matrix_exp or not ortho):
        raise RuntimeError(f"Unexpected matrix exponential nodes: {matrix_exp}")


if __name__ == "__main__":
    main()
