import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_wave62b_cudnn_fp8_conv as bench


@pytest.mark.parametrize("h,c,k,calls", bench.ops.SHAPES)
def test_cpu_simulation(h, c, k, calls):
    torch.manual_seed(225)
    assert all(m["cosine"] >= 0.995 for m in bench.parity(h, c, k).values())


@pytest.mark.parametrize("h,c,k,calls", bench.ops.SHAPES)
def test_layout(h, c, k, calls):
    x = torch.empty((1024, h, h, c), device="meta")
    view = x.permute(0, 3, 1, 2)
    assert view._base is x and view.stride() == (h * h * c, 1, h * c, c)
    assert view.is_contiguous(memory_format=torch.channels_last)


def test_hot_contract():
    bench.hot_check()


def test_sm90():
    assert len(bench.compile_kernels()) == 9
