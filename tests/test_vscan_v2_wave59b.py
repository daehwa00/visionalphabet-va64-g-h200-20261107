"""SM90a assembly and optional device screening for Wave 59B."""

import shutil
import math

import pytest
import torch

from scripts.bench_wave59b_vscan_v2_fwd import audit_case, audit_cases, verify
from scripts.bench_scan_micro import MODEL_SHAPES, make_inputs
from lnet import vscan_v2_fwd as wave
from lnet import general_cuda_scans as scans

DTYPES = (torch.bfloat16, torch.float16)
CASES = [(mode, shape, dtype) for mode in wave.MODES for shape in MODEL_SHAPES for dtype in DTYPES]


@pytest.mark.parametrize("case", audit_cases())
def test_offline(case):
    if not shutil.which("nvcc") or not shutil.which("cuobjdump"):
        pytest.skip("CUDA toolkit required")
    verify(audit_case(case))


@pytest.mark.parametrize("shape", MODEL_SHAPES)
def test_chunk(shape):
    label, h, w, m, c, _ = shape
    oh, ow = h // (1 + c), w // (1 + c)
    chunk = wave.chunk_batch(1024, h, w, m, c)
    size = 16 * m * wave.pitch(chunk, oh, ow)
    assert math.ceil(1024 / chunk) <= 64
    assert size <= (260_000_000 if label == "56d" else 32 << 20)


def test_mma_fragments():
    # Encode integer-valued operands, including the physical zero half of K.
    a = torch.arange(128, dtype=torch.float32).reshape(16, 8) % 7
    b = torch.arange(64, dtype=torch.float32).reshape(8, 8) % 11
    af = torch.zeros(32, 4, 2)
    bf = torch.zeros(32, 2, 2)
    for lane in range(32):
        g, k = divmod(lane, 4)
        af[lane, 0] = a[g, 2 * k : 2 * k + 2]
        af[lane, 1] = a[8 + g, 2 * k : 2 * k + 2]
        bf[lane, 0] = b[g, 2 * k : 2 * k + 2]
    result = torch.empty(16, 8)
    for lane in range(32):
        g, k = divmod(lane, 4)
        for half in range(2):
            for q in range(2):
                col = 2 * k + q
                result[g + 8 * half, col] = sum(
                    af[4 * g + j, half + 2 * upper, t] * bf[4 * col + j, upper, t]
                    for upper in range(2)
                    for j in range(4)
                    for t in range(2)
                )
    assert (result - a @ b.T).abs().max().item() == 0.0


def test_precise_rejected():
    with pytest.raises(ValueError, match="precise"):
        wave.vscan_fwd([], None, [], [], False, precise=True)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU hidden or unavailable")


@gpu
@pytest.mark.parametrize("mode,shape,dtype", CASES)
def test_device(mode, shape, dtype, monkeypatch):
    from h200.bench_h200 import LOCKIN_ENV

    for key, value in LOCKIN_ENV.items():
        monkeypatch.setenv(key, value)
    _, h, w, m, coarse, _ = shape
    torch.manual_seed(213)
    pole, states, aux, path, *_ = make_inputs(3, h, w, m, coarse, dtype)
    args = (pole, states, aux, path, coarse, 1e-6)
    fp = scans._reference_vertical(
        pole, states.float(), aux, [p.float() for p in path], coarse, 1e-6
    )
    baseline = scans.vertical_forward_packed(*args, planar=False).split(m, dim=-1)
    for packed, planar in ((False, False), (True, True), (True, False)):
        out = wave.vscan_fwd(*args, mode=mode, packed=packed, planar=planar)
        if packed:
            out = out.unbind() if planar else out.split(m, dim=-1)
        for x, y, z in zip(out, baseline, fp, strict=True):
            for ref in (y, z):
                assert (x.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12) < (
                    0.025 if dtype == torch.bfloat16 else 0.004
                )


@gpu
@pytest.mark.parametrize("mode", wave.MODES)
def test_graph_compile(mode):
    args = make_inputs(3, 7, 7, 128, False, torch.float16)[:4]

    def run(pole, states, aux, path):
        return wave.vscan_fwd(pole, states, aux, path, False, mode=mode, packed=True)

    expected = run(*args)
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        result = run(*args)
    graph.replay()
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.compile(run, fullgraph=True)(*args), expected)


def test_fake_trace():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    with FakeTensorMode():
        states = torch.empty((4, 3, 7, 7, 128), dtype=torch.bfloat16)
        for planar in (False, True):

            def run(x):
                return wave.vscan_fwd(
                    [], x, [], [], False, mode=wave.MODES[2], packed=True, planar=planar
                )

            graph = make_fx(run)(states)
            assert "lnet.vscan_v2_fwd_packed" in graph.code
            assert graph(states).shape == ((2, 3, 7, 7, 128) if planar else (3, 7, 7, 256))


@gpu
@pytest.mark.parametrize("mode", wave.MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_odd_coarse(mode, dtype):
    # The last input row contributes to the reverse recurrence. Zero padding
    # gives an FP32 reference for the production odd-coarse dtype restriction.
    pole, states, aux, path, *_ = make_inputs(3, 15, 15, 128, True, dtype)
    pad = torch.nn.functional.pad
    extended = pad(states.float(), (0, 0, 0, 1, 0, 1))
    gains = [pad(v, (0, 0, 0, 1), value=1) for v in aux]
    refs = scans._reference_vertical(pole, extended, gains, [p.float() for p in path], True, 1e-6)
    out = wave.vscan_fwd(pole, states, aux, path, True, mode=mode)
    for value, ref in zip(out, refs, strict=True):
        ref = ref[:, :7, :7]
        assert (value.float() - ref).norm() / ref.norm().clamp_min(1e-12) < (
            0.025 if dtype == torch.bfloat16 else 0.004
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "packed,planar", [(False, False), (False, True), (True, False), (True, True)]
)
def test_fake_layout(dtype, packed, planar):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(4, 3, 15, 17, 32, dtype=dtype)
        out = wave.vscan_fwd([], x, [], [], True, packed=packed, planar=planar)
        if packed:
            assert out.shape == ((2, 3, 7, 8, 32) if planar else (3, 7, 8, 64))
            assert out.dtype == dtype
        else:
            assert len(out) == 2 and all(t.shape == (3, 7, 8, 32) and t.dtype == dtype for t in out)


@pytest.mark.parametrize("batch,oh,ow", [(3, 7, 7), (3, 7, 8), (1, 1, 1), (3, 1, 3), (3, 28, 28)])
def test_workspace_permutation(batch, oh, ow):
    # Scatter unique cell IDs in the producer's physical layout. Decode them
    # independently in the consumer's linear 16-cell iteration order.
    tile_cells = 16 * math.ceil(oh / 4)
    packed = [None] * wave.pitch(batch, oh, ow)
    for tile in range(math.ceil(batch * ow / 4)):
        for top in range(oh - 1, -1, -4):
            first = tile * tile_cells + (oh - 1 - top) * 4
            assert first % 16 == 0
            for row in range(16):
                yo = top - row // 4
                line = tile * 4 + row % 4
                if yo >= 0 and line < batch * ow:
                    packed[first + row] = (line // ow, yo, line % ow)
    seen = set()
    for n, cell in enumerate(packed):
        tile, sub = divmod(n, tile_cells)
        rev, col = divmod(sub, 4)
        b, x = divmod(tile * 4 + col, ow)
        if b < batch and rev < oh:
            assert cell == (b, oh - 1 - rev, x)
            seen.add(cell)
        else:
            assert cell is None
    assert len(seen) == batch * oh * ow


@gpu
def test_partial_chunk_graph(monkeypatch):
    # Force a short final chunk at parity-test size. Modify inputs between
    # replays to catch stale workspace or captured pointers to prior outputs.
    args = make_inputs(3, 15, 15, 35, True, torch.float16)[:4]
    monkeypatch.setattr(wave, "chunk_batch", lambda *args: 2)

    def run(mode):
        return wave.vscan_fwd(*args, True, mode=mode, packed=True)

    run(wave.MODES[3])
    with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
        out = run(wave.MODES[3])
    for scale in (0.5, -1):
        args[1].mul_(scale)
        graph.replay()
        torch.testing.assert_close(out, run(wave.MODES[2]), atol=0, rtol=0)
