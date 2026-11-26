import json
import re
import shutil
import subprocess

import pytest
import torch

from bench_scan_micro import DTYPES, MODEL_SHAPES, make_inputs
from lnet import vscan_tc_bwd as tc


def compile_case(p, directory):
    cu = directory / "wave58.cu"
    cubin = directory / "wave58.cubin"
    cu.write_text(tc.source(p))
    build = subprocess.run(
        [
            "nvcc",
            "-arch=sm_90a",
            "-std=c++17",
            "--cubin",
            "--ptxas-options=-v",
            str(cu),
            "-o",
            str(cubin),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    sass = subprocess.check_output(["cuobjdump", "--dump-sass", str(cubin)], text=True)
    regs = int(re.search(r"Used (\d+) registers", build.stderr)[1])
    memory = re.search(
        r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", build.stderr
    )
    hmma = sass.count("HMMA.16816")
    # SM90 rounds a warp's registers to groups of 256 and reserves 1 KiB/CTA.
    shared = (p.smem + 1024 + 255) // 256 * 256
    aw = 8 * min(8, (228 * 1024) // shared, 65536 // (8 * ((regs + 7) // 8) * 256))
    return dict(
        regs=regs,
        smem=p.smem,
        aw=aw,
        hmma=hmma,
        stack=int(memory[1]),
        stores=int(memory[2]),
        loads=int(memory[3]),
    )


@pytest.mark.parametrize("mode", tc.MODES)
@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda x: x[0])
@pytest.mark.parametrize("dtype", DTYPES.values())
def test_offline(mode, shape, dtype, tmp_path, record_property):
    _, h, w, m, coarse, _ = shape
    p = tc.plan(mode, (3, h, w, m), dtype, coarse)
    assert p.grid(3, w, m) == ((3 * (w // (1 + coarse)) + p.tw - 1) // p.tw, m // p.tm, 1)
    if coarse:
        assert tc.plan(mode, (3, h, w, m), dtype, True, True) == p
    else:
        with pytest.raises(ValueError):
            tc.plan(mode, (3, h, w, m), dtype, False, True)
    assert "mma.sync.aligned.m16n8k16" in tc.source(p)
    if not all(shutil.which(tool) for tool in ("nvcc", "cuobjdump")):
        pytest.skip("CUDA toolkit required for cubin checks")
    info = compile_case(p, tmp_path)
    record_property(
        "resources", json.dumps(dict(mode=mode, shape=shape[0], dtype=str(dtype), **info))
    )
    assert info["hmma"] and max(info[k] for k in ("stack", "stores", "loads")) == 0, info
    assert info["regs"] <= 128 and info["aw"] >= 16, info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
@pytest.mark.parametrize("mode", tc.MODES)
@pytest.mark.parametrize("shape", MODEL_SHAPES, ids=lambda x: x[0])
@pytest.mark.parametrize("dtype", DTYPES.values())
def test_cuda(mode, shape, dtype, monkeypatch):
    from bench_wave58a_vscan_tc_bwd import configure_baseline, parity

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Hopper")
def test_graph_compile():
    args = make_inputs(3, 7, 7, 128, False, torch.float16)

    def run():
        return tc.vscan_tc_bwd(*args, False, False, 1e-6, next(iter(tc.MODES)))

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
def test_fake_contract(coarse, compact):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        b, h, w, m = 3, 7, 7, 128

        def empty(*shape):
            return torch.empty(shape, dtype=torch.bfloat16)

        pole = [empty(m) for _ in range(4)]
        path = [empty(m, 8, 8), empty(m, 8), empty(m, 2, 8), empty(m, 2)]
        state = empty(4, b, h, w, m)
        gains = [empty(2, w, m), empty(2, h, m)]
        dy = empty(b, h // (1 + coarse), w // (1 + coarse), m)
        outputs = tc.vscan_tc_bwd(
            pole, state, gains, path, dy, dy, coarse, compact, 1e-6, next(iter(tc.MODES))
        )
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
    expected = tc.vscan_tc_bwd(*args, True, True, 1e-6, mode)
    mixed = args._replace(
        gr=args.gr if real_packed else args.gr.contiguous(),
        gi=args.gi if imag_packed else args.gi.contiguous(),
    )
    actual = tc.vscan_tc_bwd(*mixed, True, True, 1e-6, mode)
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
