import ast
import inspect
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_wave61a_scan_fp8_handoff as bench
from lnet import scan_fp8_handoff as fp


@pytest.fixture(scope="module")
def audit():
    return bench.offline()


@pytest.mark.parametrize("shape", [*bench.MODEL_SHAPES, *bench.EDGE])
def test_offline(audit, shape):
    name, h, w, m, c, _ = shape
    row = audit[name]
    for result in row["resources"].values():
        assert all(result[k] == 0 for k in ("stack", "spill_stores", "spill_loads", "local"))
    assert 16 <= row["resources"]["A"]["aw_sm"] <= 64
    assert row["resources"]["A"]["smem"] <= 49152
    assert all(metric["cos"] >= 0.995 for metric in row["precision"].values())
    traffic = row["traffic"]
    assert traffic["hf_write_bf16"] == 2 * (traffic["hf_write_fp8"] - traffic["state_scale"])
    assert traffic["hb_grad_bf16"] == 2 * (traffic["hb_grad_fp8"] - traffic["state_scale"])
    assert traffic["vf_read_a"] == 8 * 1024 * h * (w // (1 + c)) * m
    assert traffic["saved_state_bytes"] == traffic["hf_write_bf16"] - traffic["hf_write_fp8"]


def test_graph_hot_paths():
    tree = ast.parse(inspect.getsource(fp))
    forbidden = {"item", "cpu", "synchronize", "tolist", "numpy"}
    assert not [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM90a")
@pytest.mark.parametrize("shape", [*bench.MODEL_SHAPES, *bench.EDGE])
def test_cuda_fp32_parity_and_graph(shape):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90a")
    _, h, w, m, c, _ = shape
    metrics = bench.device_precision(h, w, m, c)
    assert all(metric["cos"] >= 0.995 for metric in metrics.values()), metrics
