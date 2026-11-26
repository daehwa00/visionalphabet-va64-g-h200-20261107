"""Full-model Dynamo regression for the packed scan op cache keys."""

import bench_wave66_split_bf16 as bench
import torch
from run_va_eight_ablations import build_variant

from lnet import product_scan_pipeline, w72_split_fp8_scan


def model_scan_graph(monkeypatch, *, fp8, bf16=True) -> list:
    for key, value in bench.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("VA_W66_SPLIT_BF16", str(int(bf16)))
    monkeypatch.setenv("VA_W72_SPLIT_FP8", str(int(fp8)))
    monkeypatch.setattr(w72_split_fp8_scan, "_is_hopper", lambda _: True)

    def reference(inputs, _path, epilogue, _eps, _kind, _swiglu, planar):
        drive = inputs[8]
        n, h, w, channels = drive.shape
        if epilogue == "coarse_path":
            h, w = h // 2, w // 2
        shape = (2, n, h, w, channels // 2) if planar else (n, h, w, channels)
        return drive.new_empty(shape), drive.new_empty((0,))

    monkeypatch.setattr(product_scan_pipeline, "_packed_reference", reference)
    model = build_variant("17-p1-nr4-1024d-v3").eval()
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        # Only inspect FX. No split CUDA kernel should execute on CPU.
        nout = len(next(n for n in graph.graph.nodes if n.op == "output").args[0])
        return lambda *_: (torch.empty(0),) * nout

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.compile(model, backend=backend, fullgraph=True)(torch.zeros(1, 3, 224, 224))
    assert len(graphs) == 1
    return [
        (str(node.target), tuple(node.args[0][8].meta["example_value"].shape[1:3]), node.args[2])
        for node in graphs[0].graph.nodes
        if "lnet.product_scan_pipeline_" in str(node.target)
    ]


def test_ablation17_scan_graphs(monkeypatch) -> None:
    ops = model_scan_graph(monkeypatch, fp8=False)
    assert len(ops) == 11, ops
    assert all(name.endswith("_w66.default") for name, _, _ in ops)
