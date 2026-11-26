import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_wave64_hfma2_scan as bench


def test_56d_cubins() -> None:
    bench.check_56d()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires H200")
@pytest.mark.parametrize(
    ("h", "w", "m", "coarse"),
    [
        (56, 56, 160, False),
        (56, 56, 160, True),
        (28, 28, 160, False),
        (14, 14, 160, False),
        (7, 7, 128, False),
        (13, 17, 33, False),
    ],
)
def test_vertical(h, w, m, coarse, monkeypatch) -> None:
    for key, value in bench.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(key, value)
    bench.parity(h, w, m, coarse, graph=h == 56 and not coarse)
    if h == 56 and not coarse:
        bench.parity(h, w, m, coarse, tiny=True)
