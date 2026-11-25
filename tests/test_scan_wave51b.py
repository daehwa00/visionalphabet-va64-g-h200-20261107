"""CPU assertions for the experimental backward tile policy."""

from dataclasses import replace

import pytest
import torch

from lnet import general_cuda_scans as scans


@pytest.mark.parametrize("mode", scans.W51B_MODES)
def test_w51b_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.delenv("LNET_SCAN_W51B", raising=False)
    assert scans._scan_mode("LNET_SCAN_W51B", scans.W51B_MODES) == "off"
    monkeypatch.setenv("LNET_SCAN_W51B", mode)
    assert scans._scan_mode("LNET_SCAN_W51B", scans.W51B_MODES) == mode
    monkeypatch.setenv("LNET_SCAN_W51B", "invalid")
    with pytest.raises(ValueError, match="LNET_SCAN_W51B"):
        scans._scan_mode("LNET_SCAN_W51B", scans.W51B_MODES)


def test_w51b_off_source(monkeypatch: pytest.MonkeyPatch) -> None:
    key = scans.Kernel(torch.bfloat16, 4, backward=True, vertical=True, hidden8=True)
    monkeypatch.setenv("LNET_SCAN_W51B", "tiled_dw")
    source = scans._kernel_source(key)
    monkeypatch.setenv("LNET_SCAN_W51B", "off")
    assert scans._kernel_source(key) == source
    assert "flush_dw1" not in source
    assert "flush_dw1" in scans._kernel_source(replace(key, dw1_tile=4))


@pytest.mark.parametrize("height", [17, 31, 56])
def test_w51b_plan_limits(monkeypatch: pytest.MonkeyPatch, height: int) -> None:
    monkeypatch.setattr(scans, "device_limits", lambda _: (132, 227 * 1024, 64))
    monkeypatch.setattr(scans, "_sgs_device_limits", lambda _: (228 * 1024, 65536, 1024))
    monkeypatch.setattr(scans, "_l2_size", lambda _: 60 * 1024**2)
    monkeypatch.setattr(scans, "_resident", lambda _device, key, _smem: key.min_blocks)
    key = scans.Kernel(torch.bfloat16, 4, backward=True, vertical=True, hidden8=True)
    base = scans.Launch(scans.ScanPlan(4, 0, 8), key, 4, 1)
    scans._plan_w51b.cache_clear()
    try:
        shape = (1, height, 3, 33)
        assert scans._plan_w51b(0, base, shape, "off") is base
        launch = scans._plan_w51b(0, base, shape, "tiled_dw")
        candidate = launch.key
        threads = candidate.warps * 32
        boundary_count = (height - 1) // 8
        assert launch.plan.smem == 11264 + threads * (
            16 * (8 + boundary_count) + 32 * candidate.dw1_tile
        )
        assert candidate.min_blocks * candidate.warps == 16
        assert launch.workers == launch.partials == 3
        assert scans._plan_w51b(0, base, (1, 7, 3, 33), "tiled_dw") is base
    finally:
        scans._plan_w51b.cache_clear()


@pytest.mark.parametrize(("height", "coarse"), [(17, False), (31, True), (33, True)])
def test_w51b_odd_tail(height: int, coarse: bool) -> None:
    # Descending chunks 8 rows wide preserve coarse sampling phase.
    span, tile = 8, 4
    chunks = range(0, height, span)
    rows = []
    for start in reversed(chunks):
        end = min(start + span, height)
        selected = [y for y in range(end - 1, start - 1, -1) if not coarse or y % 2]
        rows.extend(selected[i:i + tile] for i in range(0, len(selected), tile))
    p = torch.arange(height * 8, dtype=torch.float32).reshape(height, 8) / 128
    dz = p.flip(-1) / 8
    accum = torch.zeros(8, 8)
    for group in rows:
        accum += dz[group].T @ p[group]
    step = 2 if coarse else 1
    torch.testing.assert_close(accum, dz[int(coarse)::step].T @ p[int(coarse)::step])
    assert [y for group in rows for y in group] == list(
        reversed(range(int(coarse), height, step))
    )
