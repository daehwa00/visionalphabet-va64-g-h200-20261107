"""Packed storage, source surgery, and CPU reference tests for the W57 experiment."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_wave63_w57_packed_scan as bench


@pytest.fixture(scope="module")
def plans():
    return list(bench.offline_plans())


@pytest.mark.parametrize("variant", bench.VARIANTS[1:])
def test_surgical_kernel_matrix(plans, variant):
    assert len(plans) == 32
    for label, op, w, m, launch in plans:
        key = launch.key
        src = bench.kernel_source(key, variant, w, 13 if label == "odd" else w, m)
        base = bench.scans._kernel_source(key)
        assert src.count('extern "C" __global__') == base.count('extern "C" __global__')
        assert src.count("line += workers") == base.count("line += workers")
        assert src.count("line+=workers") == base.count("line+=workers")
        assert src.count("path.input_group(") == base.count("path.input_group(")
        assert src.count("tanh") == base.count("tanh")
        assert "__launch_bounds__(" + str(32 * key.warps) + ")" in src
        if op == "VB" and variant == "packed_fp8":
            assert "c.inv_ds=row_inverse" in src
            assert "for(int yo=0;yo<oh;" not in src


@pytest.mark.parametrize("variant", bench.VARIANTS[1:])
@pytest.mark.parametrize("shape", bench.SHAPES, ids=lambda t: t[0])
def test_cpu_handoff_precision(shape, variant):
    torch.set_num_threads(2)
    _, h, w, m, c = shape
    data = bench.precision(h, w, m, c, variant=variant)
    assert data["saturation"] == 0
    assert all(row["cos"] >= 0.995 for row in data.values() if isinstance(row, dict))


def test_four_byte_order_and_mode_tail():
    raw = torch.tensor([1.0, 2.0, -4.0, 8.0]).view(4, 1, 1, 1, 1).expand(4, 2, 3, 5, 33)
    scale = torch.ones(2, 2, 1, 3, 1, 1)
    words, values, _ = bench.pack(raw, scale, torch.float8_e4m3fn)
    assert words.shape == (2, 2, 5, 3, 32)
    assert words.is_contiguous()
    assert torch.equal(
        words.unsqueeze(-1).view(torch.uint8)[1, 1, 4, 2, 0],
        torch.tensor([0x38, 0x40, 0xC8, 0x50], dtype=torch.uint8),
    )
    assert torch.equal(values, raw)


@pytest.mark.parametrize("active", (16, 32))
def test_vec2_word_exchange_is_coalesced_and_invertible(active):
    half = active // 2
    first = [2 * i for i in range(active)]
    second = [i + 1 for i in first]
    output = [None] * (2 * active)
    positions = []
    for lane in range(active):
        low = lane < half
        other = lane ^ half
        swap = first[other] if low else second[other]
        dest = 2 * (lane & (half - 1)) + int(not low)
        positions.append(dest)
        output[dest] = first[lane] if low else swap
        output[dest + active] = swap if low else second[lane]
    assert sorted(positions) == list(range(active))
    assert output == list(range(2 * active))
    for lane, dest in enumerate(positions):
        low = lane < half
        peer = positions[lane ^ half]
        exchanged = output[peer] if low else output[peer + active]
        assert (output[dest] if low else exchanged) == first[lane]
        assert (exchanged if low else output[dest + active]) == second[lane]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("shape", bench.SHAPES, ids=lambda t: t[0])
def test_hopper_pipeline_against_reference(monkeypatch, shape):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90a")
    for k, v in bench.lockin_arm_env("w57_lockin").items():
        monkeypatch.setenv(k, v)
    _, h, w, m, c = shape
    for variant in bench.VARIANTS[1:]:
        data = bench.precision(h, w, m, c, "cuda", variant)
        assert all(v["cos"] >= 0.995 for v in data.values() if isinstance(v, dict))
        if variant == "packed_bf16":
            assert data["w57_control_rel_l2"] == 0


@pytest.mark.parametrize("variant", bench.VARIANTS[1:])
def test_host_pipeline_allocations_and_abi(monkeypatch, plans, variant):
    """Use CPU storage and inspect the actual driver argument vectors."""
    fp = bench.fp
    ops = {op: launch for name, op, _, _, launch in plans if name == "14c"}

    def plan(src, shape, *, vertical, backward, **kwargs):
        return ops[("V" if vertical else "H") + ("B" if backward else "F")]

    monkeypatch.setattr(bench.scans, "_plan_launch", plan)
    monkeypatch.setattr(fp, "_check_source", lambda *args: None)
    monkeypatch.setattr(
        bench.scans,
        "_reduce_partials",
        lambda partial, params, *args: list(map(torch.zeros_like, params)),
    )
    calls = []

    def dispatch(real, launch, variant, width, args):
        calls.append(len(args))
        assert width == 14 and real.shape[-1] == 160

    monkeypatch.setattr(fp, "_dispatch", dispatch)
    b, h, w, m = 2, 14, 14, 160
    r, i = torch.zeros((b, h, w, 2 * m), dtype=torch.bfloat16).split(m, -1)
    pole = [torch.ones(m) for _ in range(4)]
    aux = [torch.ones(2, n, m) for n in (w, h)]
    path = [
        torch.zeros(shape, dtype=torch.bfloat16) for shape in ((m, 8, 8), (m, 8), (m, 2, 8), (m, 2))
    ]
    state = fp.horizontal(pole, r, i, variant)
    assert state[0].shape == (b, 5, w, h, 32)
    out = fp.vertical(pole, r, state, aux, path, True, variant)
    assert out.shape == (b, h // 2, w // 2, 2 * m)
    grads = fp.vertical(pole, r, state, aux, path, True, variant, out.split(m, -1))
    assert grads[4].shape == (b, 5, w // 2, h, 32)
    final = fp.horizontal(pole, r, i, variant, state, grads[4:6], True)
    assert final[-1].shape == (b, h, w, 2 * m)
    assert calls == [17, 31, 31, 18]
