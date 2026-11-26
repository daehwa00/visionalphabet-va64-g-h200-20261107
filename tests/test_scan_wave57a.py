"""Offline launch contracts and device integration for the production scan policies."""

import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from lnet import general_cuda_scans as scans

SHAPES = [(h, coarse) for h in (56, 28, 14, 7) for coarse in (False, True) if h != 7 or not coarse]
CONFIG = dict(W51A="fp32", W53A="w57_lockin", HSCAN="h55_lockin", SGS="strict_ro", VFWD="batch4_w2")


@pytest.fixture(autouse=True)
def configure(monkeypatch):
    planners = [
        fn
        for name, fn in vars(scans).items()
        if name.startswith(("_plan_", "_launch_for_shape", "_lockin_"))
        and hasattr(fn, "cache_clear")
    ]
    for fn in planners:
        fn.cache_clear()
    for name, value in CONFIG.items():
        monkeypatch.setenv("LNET_SCAN_" + name, value)
    monkeypatch.setenv("LNET_VSCAN_DECOUPLED", "off")
    yield
    for fn in planners:
        fn.cache_clear()


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(scans, "device_limits", lambda _: (132, 227 << 10, 64))
    monkeypatch.setattr(scans, "_sgs_device_limits", lambda _: (228 << 10, 65536, 1024))
    monkeypatch.setattr(scans, "_l2_size", lambda _: 50 << 20)
    monkeypatch.setattr(scans, "_scan_workspace_budget", lambda _: 128 << 20)
    monkeypatch.setattr(
        scans,
        "_resident",
        lambda _, k, smem: min(
            64 // k.warps, (228 << 10) // (1024 + smem), 2 if k.vertical and k.backward else 16
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))


def plan(dtype, h, w, m, coarse, vertical, backward):
    src = SimpleNamespace(
        dtype=dtype,
        is_cuda=True,
        device=torch.device("cuda"),
        numel=lambda: 1024 * h * w * m * (4 if vertical else 1),
    )
    return scans._plan_launch(
        src,
        (1024, h, w, m),
        vertical=vertical,
        backward=backward,
        hidden=8 if vertical else 0,
        precise=vertical and not backward,
        coarse=vertical and coarse,
        compact=backward and coarse,
        interleaved=(True, True),
    )


# Baseline (grid, threads, dynamic shared bytes), ordered VF, VB, HF, HB.
LAUNCH = {
    torch.bfloat16: [
        [(4225, 64, 11264), (2110, 128, 112128), (43008, 128, 57344), (1840, 128, 28672)],
        [(4225, 128, 5632), (260, 128, 112128), (43008, 128, 57344), (9504, 128, 28672)],
        [(4680, 128, 5632), (260, 128, 91648), (35840, 128, 14336), (42240, 96, 21504)],
        [(8445, 128, 19968), (525, 128, 105984), (35840, 128, 14336), (3960, 128, 14336)],
        [(8445, 128, 19968), (790, 128, 77312), (17920, 128, 7168), (3960, 128, 14336)],
        [(8960, 128, 12800), (790, 128, 55808), (17920, 128, 7168), (4220, 128, 7168)],
        [(7168, 128, 12800), (792, 128, 47104), (3584, 128, 0), (1792, 128, 7168)],
    ],
    torch.float16: [
        [(4225, 64, 11264), (2110, 128, 112128), (43008, 128, 57344), (1840, 128, 28672)],
        [(4225, 128, 5632), (260, 128, 112128), (43008, 128, 57344), (9504, 128, 28672)],
        [(4680, 128, 5632), (260, 128, 91648), (35840, 128, 14336), (42240, 96, 21504)],
        [(8445, 128, 19968), (525, 128, 105984), (35840, 128, 14336), (3960, 128, 14336)],
        [(8445, 128, 19968), (790, 128, 77312), (17920, 128, 7168), (3960, 128, 14336)],
        [(8960, 128, 12800), (790, 128, 55808), (17920, 128, 7168), (4220, 128, 7168)],
        [(7168, 128, 12800), (792, 128, 47104), (3584, 128, 0), (1792, 128, 7168)],
    ],
}


def test_active_modes_and_legacy_off(monkeypatch, offline):
    allowed = {
        "W51A": scans.W51A_MODES,
        "SGS": scans.SGS_MODES,
        "VFWD": scans.VFWD_MODES,
        "W53A": scans.W53A_MODES,
        "HSCAN": scans.HSCAN_MODES,
    }
    for option, modes in allowed.items():
        assert modes == ("off", CONFIG[option])
    names = ("LNET_SCAN_W52A", "LNET_SCAN_W51B", "LNET_SCAN_VSCAN_SAVED")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    args = (torch.float16, 28, 28, 160, True, True, True)
    baseline = plan(*args)
    for name in names:
        monkeypatch.setenv(name, "off")
    assert plan(*args) == baseline


@pytest.mark.parametrize("dtype", LAUNCH)
def test_lockin_launches(offline, dtype):
    for (h, coarse), expected in zip(SHAPES, LAUNCH[dtype], strict=True):
        modes = 128 if h == 7 else 160
        for slot, (vertical, backward) in enumerate(
            ((True, False), (True, True), (False, False), (False, True))
        ):
            launch = plan(dtype, h, h, modes, coarse, vertical, backward)
            key = launch.key
            grid = scans._ceil(modes, 32 * (1 + key.horizontal_vec2)) * scans._ceil(
                launch.workers, key.warps
            )
            assert (grid, 32 * key.warps, launch.plan.smem) == expected[slot]
            if vertical:
                weights = "v55_bf16x2_bf16w"
                if h == 56 and not coarse and not backward:
                    weights = "off"
                if h == 7 and backward:
                    weights = "v55_bf16x2_predp"
                assert key.vertical_w55 == weights
                assert key.seg16 == (backward and (h == 56 or (coarse and h == 14)))
                assert (key.vertical_w56 == "v56_noround") == (
                    not backward and h < (48 if coarse else 24)
                )


@pytest.mark.parametrize("arch", ["75", "86"])
@pytest.mark.parametrize("dtype,h,w,m", [(torch.float32, 56, 56, 160), (torch.float16, 15, 17, 33)])
def test_fallback(offline, monkeypatch, tmp_path, arch, dtype, h, w, m):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: tuple(map(int, arch)))
    for vertical in (True, False):
        for backward in (True, False):
            launch = plan(dtype, h, w, m, True, vertical, backward)
            assert not launch.key.seg16 and not launch.key.horizontal_vec2
            if dtype == torch.float32 or arch == "75":
                assert launch.key.vertical_w55 == "off"
            if not shutil.which("nvcc"):
                pytest.skip("nvcc required for offline cubin check")
            source = tmp_path / "scan.cu"
            key = scans._effective_kernel(launch.key, tuple(map(int, arch)))
            source.write_text(scans._kernel_source(key))
            subprocess.run(
                [
                    "nvcc",
                    "-arch=sm_" + arch,
                    "-std=c++17",
                    "--cubin",
                    str(source),
                    "-o",
                    str(tmp_path / "scan.cubin"),
                ],
                check=True,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_pipeline(monkeypatch, dtype):
    torch.manual_seed(141)
    h, m = 7, 128

    def sample(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype) * 0.1

    x, y = sample(2, h, h, 2 * m).chunk(2, dim=-1)
    pole = [torch.full((m,), v, device="cuda") for v in (0.73, 0.13, 0.21, 0.09)]
    gains = [1 + torch.rand(2, h, m, device="cuda") for _ in range(2)]
    path = [sample(*s) for s in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))]
    dr, di = sample(2, h, h, 2 * m).chunk(2, dim=-1)

    def run():
        states = scans.horizontal_forward(pole, x, y)
        out = scans.vertical_forward(pole, states, gains, path)
        dv = scans.vertical_backward(pole, states, gains, path, dr, di)
        dh = scans.horizontal_backward(pole, [x, y], states, dv[4])
        return (*out, *dv, *dh)

    with monkeypatch.context() as off:
        for name in CONFIG:
            off.setenv("LNET_SCAN_" + name, "off")
        reference = run()
    eager = run()
    for actual, expected in zip(eager, reference, strict=True):
        error = (actual.float() - expected.float()).norm()
        assert error <= 0.035 * expected.float().norm() + 1e-5
    torch.testing.assert_close(torch.compile(run, backend="aot_eager", fullgraph=True)(), eager)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        replay = run()
    x.mul_(0.5)
    capture.replay()
    torch.testing.assert_close(replay, run())
