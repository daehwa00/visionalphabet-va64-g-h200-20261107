"""Wave 62A layout, CPU recurrence and optional device equivalence checks."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_wave62a_packed_fp8_scan as bench
from lnet import packed_fp8_scan as fp


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_word_byte_order_and_column_stride(dtype):
    x = torch.arange(4 * 2 * 3 * 5 * 33).reshape(4, 2, 3, 5, 33).float().remainder(17) - 8
    q, back, counts = bench.pack(x, 1, dtype)
    assert q.shape == (2, 2, 5, 3, 32) and q.dtype == torch.uint32
    assert q.stride(-2) * q.element_size() == 128
    # Interpret bytes directly to catch swapping both encoder and decoder.
    bytes_ = q.unsqueeze(-1).view(torch.uint8)
    expected = x[:, 1, 2, 4, 32].to(dtype).view(torch.uint8)
    torch.testing.assert_close(bytes_[1, 1, 4, 2, 0], expected)
    assert not bytes_[1, 1, 4, 2, 1:].any()
    torch.testing.assert_close(back, x.to(dtype).float())
    assert not any(counts.values())


@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("full", [False, True])
def test_odd_cpu_adjoint(coarse, full):
    torch.set_num_threads(1)
    precision, counts = bench.simulate(13, 17, 33, coarse, full)
    assert len(precision) == 12
    assert min(v["cos"] for v in precision.values()) >= 0.995
    assert max(v["nonfinite"] for v in counts.values()) == 0


def test_scale_can_underflow_without_nonfinites():
    x = torch.ones(4, 1, 1, 1, 32)
    x[2, ..., 17] = 2**-40
    _, _, counts = bench.pack(x, 1 / 64, torch.float8_e5m2)
    assert counts == {"underflow": 1, "saturation": 0, "nonfinite": 0}


def test_resource_budget_and_bank_layout():
    assert fp.shared_bytes(fp.SEG, 56, 56, False) + 1024 <= (228 * 1024) // 4
    assert fp.shared_bytes(fp.VF, 56, 56, False) + 1024 <= (228 * 1024) // 4
    assert "wt[k][lane]=v" in fp._BACKWARD
    assert "weights[k][lane]=v" in fp.CUDA
    assert "lane>=8" not in fp.CUDA + fp._BACKWARD
    assert ".item()" not in Path(fp.__file__).read_text()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper device")
@pytest.mark.parametrize("shape", [*bench.MODEL_SHAPES, *bench.EDGE], ids=lambda s: s[0])
def test_device_fp32_parity(shape):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90")
    _, h, w, m, c, _ = shape
    result = bench.device_precision(h, w, m, c)
    assert min(v["cos"] for mode in result.values() for v in mode.values()) >= 0.995
