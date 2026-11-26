import pytest
import torch
from bench_scan_micro import DTYPES, MODEL_SHAPES, make_inputs
from bench_wave59a_vscan_v2_bwd import audit
from lnet import vscan_v2_bwd as tc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("mode", tc.MODES)
@pytest.mark.parametrize(
    "shape",
    (*MODEL_SHAPES, ("odd-coarse", 15, 15, 160, True, 1), ("odd-dense", 15, 15, 63, False, 1)),
    ids=lambda x: x[0],
)
@pytest.mark.parametrize("dtype", DTYPES.values())
def test_cuda(mode, shape, dtype, monkeypatch):
    from bench_wave59a_vscan_v2_bwd import configure_baseline, parity

    configure_baseline(monkeypatch.setenv)
    _, h, w, m, coarse, _ = shape
    torch.manual_seed(212)
    inputs = make_inputs(3, h, w, m, coarse, dtype)
    for compact in [False, True] if coarse else [False]:
        for reference, errors in parity(inputs, mode, coarse, compact).items():
            limit = (
                (0.025 if dtype == torch.bfloat16 else 0.004)
                if reference == "fp32"
                else (0.015 if dtype == torch.bfloat16 else 0.003)
            )
            assert all(value < limit for value in errors.values()), (reference, errors)

        graph = torch.cuda.CUDAGraph()
        expected = tc.vscan_v2_bwd(*inputs, coarse, compact, 1e-6, mode)
        with torch.cuda.graph(graph):
            replayed = tc.vscan_v2_bwd(*inputs, coarse, compact, 1e-6, mode)
        graph.replay()
        for got, want in zip(replayed, expected, strict=True):
            assert got.isfinite().all()
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("mode", tc.MODES)
def test_graph_compile(mode):
    args = make_inputs(3, 7, 7, 128, False, torch.float16)

    def run():
        return tc.vscan_v2_bwd(*args, False, False, 1e-6, mode)

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
        outputs = tc.vscan_v2_bwd(pole, state, gains, path, dy, dy, coarse, compact, 1e-6, mode)
        assert [x.dtype for x in outputs] == [
            *(t.dtype for t in pole),
            state.dtype,
            *(t.dtype for t in path),
        ]
        with pytest.raises(ValueError):
            tc.vscan_v2_bwd(pole, state, gains, path, dy, dy, False, True, 1e-6, mode)
        assert [x.shape for x in outputs] == [
            *(x.shape for x in pole),
            (4, b, h, w // 2 if compact else w, m),
            *(x.shape for x in path),
        ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("real_packed,imag_packed", [(False, False), (True, False), (False, True)])
def test_independent_adjoint_strides(real_packed, imag_packed):
    args = make_inputs(3, 14, 14, 160, True, torch.bfloat16)
    mode = next(iter(tc.MODES))
    expected = tc.vscan_v2_bwd(*args, True, True, 1e-6, mode)
    mixed = args._replace(
        gr=args.gr if real_packed else args.gr.contiguous(),
        gi=args.gi if imag_packed else args.gi.contiguous(),
    )
    actual = tc.vscan_v2_bwd(*mixed, True, True, 1e-6, mode)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_auxiliary_lane_reduction():
    # Tagged integer contributions make dropped or duplicated cells visible.
    cells = torch.arange(16).reshape(16, 1)
    hidden = torch.arange(8).reshape(1, 8)
    dz = (1 + cells) * (1 + hidden)
    act = 3 + cells - hidden
    dy = torch.stack((cells[:, 0] + 1, 17 - cells[:, 0]), dim=1)
    fields = torch.stack((dz, dy[:, :1] * act, dy[:, 1:] * act))
    partial = torch.zeros(32, 3, dtype=torch.int64)
    bias = torch.empty(32, dtype=torch.int64)
    for lane in range(32):
        group, pair = lane // 4, 2 * (lane % 4)
        unit = pair + group % 2
        for cell in (group, group ^ 1, group + 8, (group ^ 1) + 8):
            partial[lane] += fields[:, cell, unit]
        bias[lane] = dy[lane % 16, lane // 16]
    out = torch.empty(3, 8, dtype=torch.int64)
    for lane in range(8):
        out[:, 2 * (lane % 4) + lane // 4] = partial[lane::8].sum(0)
    assert torch.equal(out, fields.sum(1))
    assert torch.equal(bias.reshape(2, 16).sum(1), dy.sum(0))


@pytest.mark.parametrize("mode", tc.MODES)
def test_reject_compact_dense(mode):
    with pytest.raises(ValueError):
        tc.kernels(mode, (3, 15, 15, 160), torch.bfloat16, False, True)


@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda s: s[0])
def test_workspace(shape):
    _, h, w, m, c, _ = shape
    cells = (h // (1 + c)) * (w // (1 + c))
    chunk = tc.chunk_size(tc.MODES[1], 1024, cells, m)
    size = 36 * m * tc.padded(chunk * cells)
    assert chunk & (chunk - 1) == 0 and 1024 // chunk <= 64 and size < 600 << 20
    assert size <= 36 << 20 or 1024 // chunk == 64
    if shape[0] == "56d":
        assert chunk == 16 and size == 289013760


def fragment_mma(af, bf):
    """Apply one MMA to the 32 sets of packed A and B register operands."""
    out = torch.zeros(16, 8, dtype=torch.float32)
    for lane in range(32):
        for r in range(2):
            for j in range(2):
                y = lane // 4 + 8 * r
                x = 2 * (lane % 4) + j
                for t in range(16):
                    out[y, x] += (
                        af[(y % 8) * 4 + t % 8 // 2, (t // 8) * 4 + r * 2 + t % 2]
                        * bf[x * 4 + t % 8 // 2, (t // 8) * 2 + t % 2]
                    )
    return out


@pytest.mark.parametrize("mode", tc.MODES)
@pytest.mark.parametrize("dtype", DTYPES.values())
def test_fragments(mode, dtype):
    gen = torch.Generator().manual_seed(215)

    def sample(*shape):
        return torch.randint(-2, 3, shape, generator=gen).float()

    tw = {tc.MODES[2]: 8, tc.MODES[3]: 4}.get(mode, 1)
    # Select a nonzero mode after scan lanes deposit column vectors.
    cells = sample(16 // tw, tw, 32, 8)
    p = cells.permute(2, 0, 1, 3).contiguous()[17].reshape(16, 8)
    w = sample(8, 8)
    dy = sample(16, 2)
    wo = sample(2, 8)
    lanes = torch.arange(32)
    group = lanes // 4
    pair = 2 * (lanes % 4)

    def load(matrix, offset=0):
        return torch.stack([matrix[group + offset, pair + i] for i in range(2)], dim=1)

    def transpose(reg):
        if mode in tc.MODES[2:]:
            # SHFL.IDX and PRMT select one halfword from each neighboring row.
            owner = (lanes % 4) * 8 + group // 2
            return torch.stack([reg[owner + delta, group % 2] for delta in (0, 4)], dim=1)
        # The decoupled warp writes a 10-halfword pitch and reads the transpose.
        scratch = torch.zeros(8, 10)
        for i in range(2):
            scratch[group, pair + i] = reg[:, i]
        return torch.stack([scratch[pair + i, group] for i in range(2)], dim=1)

    a, b, wt = load(p), load(p, 8), load(w)
    zero = torch.zeros_like(a)
    pack = lambda *regs: torch.cat(regs, dim=1)
    pre = fragment_mma(pack(a, b, zero, zero), pack(wt, zero))
    assert (pre - p @ w.T).abs().max().item() == 0.0
    bias = sample(8)
    z = (pre.to(dtype).float() + bias).to(dtype).float()
    sig = z.sigmoid()
    dz = ((dy @ wo).to(dtype).float() * (sig * (1 + z * (1 - sig)))).to(dtype).float()
    d0, d1 = load(dz), load(dz, 8)
    dp = fragment_mma(pack(d0, d1, zero, zero), pack(transpose(wt), zero))
    dw = fragment_mma(
        pack(transpose(d0), zero, transpose(d1), zero), pack(transpose(a), transpose(b))
    )[:8]
    assert (dp - dz @ w).abs().max().item() == 0.0
    assert (dw - dz.T @ p).abs().max().item() == 0.0


@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda s: s[0])
@pytest.mark.parametrize("dtype", DTYPES.values())
def test_cubins(shape, dtype):
    _, h, w, m, c, _ = shape
    for mode in tc.MODES:
        for k in tc.kernels(mode, (1024, h, w, m), dtype, c):
            v = audit(k)
            assert (v["stack"], v["stores"], v["loads"]) == (0, 0, 0), (k, v)
            max_regs = {
                "pure_vscan_fwd_p": 48,
                "pure_vscan_bwd_gs": 64,
                "tc_path_mlp_bwd": 64,
                "fused": 128,
            }
            min_warps = {
                "pure_vscan_fwd_p": 1,
                "pure_vscan_bwd_gs": 32,
                "tc_path_mlp_bwd": 48,
                "fused": 16,
            }
            assert v["regs"] <= max_regs[k.name] and v["aw"] >= min_warps[k.name], (k, v)
            assert v["smem"] <= 227 * 1024
            if k.name == "tc_path_mlp_bwd":
                assert v["block_barriers"] == 0 and v["smem"] <= 4096
            if k.cluster > 1:
                assert v["cluster_waits"] >= 3
            assert v["hmma"] >= 3 if k.name in ("fused", "tc_path_mlp_bwd") else v["hmma"] == 0


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", DTYPES.values())
def test_pole_sensitivities(reverse, dtype):
    gen = torch.Generator().manual_seed(215)

    def value():
        return torch.complex(torch.randn(56, generator=gen), torch.randn(56, generator=gen))

    def rounded(t):
        return torch.complex(t.real.to(dtype).float(), t.imag.to(dtype).float())

    x, inject = value(), rounded(value())
    a = torch.tensor(0.71 + (0.19j if reverse else -0.19j))
    g = torch.tensor(0.37 - 0.24j)
    y = sa = sg = da = dg = torch.zeros((), dtype=torch.complex64)
    order = list(range(55, -1, -1) if reverse else range(56))
    prev = {}
    for n in order:
        prev[n] = rounded(y)
        sa = a * sa + prev[n]
        sg = a * sg + x[n]
        y = a * y + g * x[n]
        da = da + inject[n] * sa.conj()
        dg = dg + inject[n] * sg.conj()
    adj = ra = rg = torch.zeros_like(y)
    for n in reversed(order):
        adj = a.conj() * adj + inject[n]
        ra = ra + adj * prev[n].conj()
        rg = rg + adj * x[n].conj()
    torch.testing.assert_close(torch.stack([da, dg]), torch.stack([ra, rg]), rtol=2e-6, atol=1e-5)


@pytest.mark.parametrize("h", [13, 14, 53, 56])
@pytest.mark.parametrize("coarse", [False, True])
def test_cluster_partition(h, coarse):
    k = tc.kernels(tc.MODES[3], (3, h, 15, 160), torch.bfloat16, coarse)[0]
    step = 1 + coarse
    cells = (h // step + k.cluster - 1) // k.cluster
    blocks = [min(h // step, rank * cells) * step for rank in range(k.cluster)] + [h]
    assert max(b - a for a, b in zip(blocks, blocks[1:])) <= k.stripe
    x = torch.arange(h, dtype=torch.float64) + 1

    def scan(rows, seed, vec):
        for y in rows:
            seed = 0.75 * seed + vec[y]
        return seed

    for rank, (lo, hi) in enumerate(zip(blocks, blocks[1:])):
        forward = reverse = 0.0
        for peer in range(rank):
            forward = scan(range(blocks[peer], blocks[peer + 1]), forward, x)
        for peer in range(k.cluster - 1, rank, -1):
            reverse = scan(range(blocks[peer + 1] - 1, blocks[peer] - 1, -1), reverse, x)
        assert forward == scan(range(lo), 0.0, x)
        assert reverse == scan(range(h - 1, hi - 1, -1), 0.0, x)
