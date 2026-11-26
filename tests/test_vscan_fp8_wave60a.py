import math

import pytest
import torch
from bench_scan_micro import DTYPES, MODEL_SHAPES, make_inputs
from bench_wave60a_vscan_fp8_bwd import audit

from lnet import vscan_fp8_bwd as tc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("mode", tc.MODES)
@pytest.mark.parametrize(
    "shape",
    (*MODEL_SHAPES, ("odd-coarse", 15, 15, 160, True, 1), ("odd-dense", 15, 15, 63, False, 1)),
    ids=lambda x: x[0],
)
@pytest.mark.parametrize("dtype", DTYPES.values())
def test_cuda(mode, shape, dtype, monkeypatch):
    from bench_wave60a_vscan_fp8_bwd import configure_baseline, parity

    configure_baseline(monkeypatch.setenv)
    _, h, w, m, coarse, _ = shape
    torch.manual_seed(212)
    inputs = make_inputs(3, h, w, m, coarse, dtype)
    for compact in [False, True] if coarse else [False]:
        for reference, errors in parity(inputs, mode, coarse, compact).items():
            assert all(v["relative_l2"] < 0.145 and v["cosine"] >= 0.99 for v in errors.values()), (
                reference,
                errors,
            )

        graph = torch.cuda.CUDAGraph()
        expected = tc.vscan_fp8_bwd(*inputs, coarse, compact, 1e-6, mode)
        with torch.cuda.graph(graph):
            replayed = tc.vscan_fp8_bwd(*inputs, coarse, compact, 1e-6, mode)
        graph.replay()
        for got, want in zip(replayed, expected, strict=True):
            assert got.isfinite().all()
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("mode", tc.MODES)
def test_graph_compile(mode):
    args = make_inputs(3, 7, 7, 128, False, torch.float16)

    def run():
        return tc.vscan_fp8_bwd(*args, False, False, 1e-6, mode)

    expected = run()
    compiled = torch.compile(run, fullgraph=True)
    for a, b in zip(compiled(), expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = run()
    graph.replay()
    for a, b in zip(replayed, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("coarse,compact", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", DTYPES.values())
@pytest.mark.parametrize("mode", tc.MODES)
def test_fake_contract(coarse, compact, dtype, mode):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        b, h, w, m = 3, 7, 7, 128

        def empty(*shape):
            return torch.empty(shape, dtype=dtype)

        pole = [empty(m).float() for _ in range(4)]
        path = [empty(m, 8, 8), empty(m, 8), empty(m, 2, 8), empty(m, 2)]
        state = empty(4, b, h, w, m)
        gains = [empty(2, w, m), empty(2, h, m)]
        dy = empty(b, h // (1 + coarse), w // (1 + coarse), m)
        outputs = tc.vscan_fp8_bwd(pole, state, gains, path, dy, dy, coarse, compact, 1e-6, mode)
        assert [x.dtype for x in outputs] == [
            *(t.dtype for t in pole),
            state.dtype,
            *(t.dtype for t in path),
        ]
        with pytest.raises(ValueError):
            tc.vscan_fp8_bwd(pole, state, gains, path, dy, dy, False, True, 1e-6, mode)
        assert [x.shape for x in outputs] == [
            *(x.shape for x in pole),
            (4, b, h, w // 2 if compact else w, m),
            *(x.shape for x in path),
        ]


@pytest.mark.parametrize("cells,modes", [(53, 65), (3136, 160), (1, 1)])
def test_tile_major(cells, modes):
    blocks = tc.padded(cells) // 16
    buf = torch.arange(((modes + 31) // 32) * blocks * 4096).view(-1, blocks, 32, 16, 8)
    m = torch.arange(modes)[:, None, None]
    n = torch.arange(cells)[None, :, None]
    k = torch.arange(8)
    linear = (((m // 32 * blocks + n // 16) * 32 + m % 32) * 16 + n % 16) * 8 + k
    assert torch.equal(buf[m // 32, n // 16, m % 32, n % 16, k], linear)
    assert buf.stride()[1] == 4096


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_column_precision(dtype):
    rng = torch.Generator().manual_seed(218)
    features = torch.randn(8, 56, 8, generator=rng)
    dy = torch.randn(8, 56, 8, generator=rng)
    scale = features.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6) / torch.finfo(dtype).max
    approx = (features / scale).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max).to(
        dtype
    ).float() * scale
    total = torch.zeros(8, 8)
    for f, d in zip(approx, dy, strict=True):
        col = torch.zeros(8, 4, 2, dtype=torch.float16)
        for row, grad in zip(f, d, strict=True):
            col = (
                col.float()
                + ((grad / 16).half().float()[:, None, None] * row.half().float().view(1, 4, 2))
            ).half()
        total += col.float().view(8, 8) * 16
    exact = torch.einsum("bhu,bhk->uk", dy, features)
    cosine = torch.nn.functional.cosine_similarity(total.flatten(), exact.flatten(), dim=0)
    print(dtype, "column cosine", float(cosine))
    assert cosine >= 0.995


def test_fastsilu_derivative():
    z = torch.linspace(-8, 8, 4097, dtype=torch.float64, requires_grad=True)
    t = (z / 2).tanh()
    y = z * (0.5 + 0.5 * t)
    derivative = 0.5 + 0.5 * t + 0.25 * z * (1 - t * t)
    torch.testing.assert_close(torch.autograd.grad(y.sum(), z)[0], derivative)
    torch.testing.assert_close(derivative, z.sigmoid() * (1 + z * (1 - z.sigmoid())))


@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda s: s[0])
@pytest.mark.parametrize("dtype", DTYPES.values())
def test_cubins(shape, dtype):
    _, h, w, m, c, _ = shape
    for mode in tc.MODES:
        for k in tc.kernels(mode, (1024, h, w, m), dtype, c, compact=c):
            info = audit(k)
            assert not any(info[key] for key in ("stack", "stores", "loads")), (k, info)
            assert info["fp8"]
            if k.name == "tc_path_mlp_bwd_fp8":
                assert info["hmma"] >= 3 and info["block_barriers"] == 0
            if k.name == "column_fp8":
                assert info["hfma2"] > 0 and not info["hmma"]


@pytest.mark.parametrize("lines", [1, 3, 4, 5, 17, 28])
@pytest.mark.parametrize("height", [1, 3, 7, 14, 28, 56])
def test_panel_and_tail_cell_order(lines, height):
    groups = lines // 4 if height >= 4 else 0
    rows = height // 4 * 4
    decoded = []
    for tile in range(tc.padded(lines * height) // 16):
        for r in range(16):
            n = tile * 16 + r
            if n >= lines * height:
                continue
            if n < groups * 4 * rows:
                line = tile // (rows // 4) * 4 + r % 4
                y = (tile % (rows // 4)) * 4 + r // 4
            else:
                tail = n - groups * 4 * rows
                if tail < groups * 4 * (height - rows):
                    line, offset = divmod(tail, height - rows)
                    y = rows + offset
                else:
                    line, y = divmod(n - groups * 4 * height, height)
                    line += groups * 4
            encoded = (
                line * height + y
                if line >= groups * 4
                else (line // 4 * rows + y) * 4 + line % 4
                if y < rows
                else groups * 4 * rows + line * (height - rows) + y - rows
            )
            assert encoded == n
            decoded.append((line, y))
    assert len(set(decoded)) == height * lines


def test_large_column_scale():
    h = 56
    feature = torch.tensor([1.0e5, -2.0e5, 3.0e5, -4.0e5, 4.0e5, -3.0e5, 2.0e5, -1.0e5])
    dz = torch.linspace(2.0e3, 6.0e3, h)
    acc = torch.zeros(4, 2, dtype=torch.half)
    scale = 16.0
    for d in dz:
        bound = h / 2048 * float(abs(d)) * float(feature.abs().max())
        if bound > scale:
            new = 2.0 ** math.ceil(math.log2(bound))
            acc = (acc * (scale / new)).half()
            scale = new
        xs = max(1.0, float(feature.abs().max()) / 16384.0)
        acc = (
            acc.float() + (d / scale * xs).half().float() * (feature / xs).half().view(4, 2).float()
        ).half()
    result = acc.float().flatten() * scale
    target = feature * dz.sum()
    assert result.isfinite().all()
    assert (result - target).norm() / target.norm() < 0.01
    assert torch.nn.functional.cosine_similarity(result, target, dim=0) >= 0.995
