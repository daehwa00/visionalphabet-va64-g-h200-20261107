"""SM90a assembly and optional device screening for Wave 58B."""

import json
import os
import re
import shutil
import subprocess

import pytest
import torch

from scripts.bench_scan_micro import MODEL_SHAPES, make_inputs
from lnet import vscan_tc_fwd as wave
from lnet import general_cuda_scans as scans

DTYPES = (torch.bfloat16, torch.float16)
CASES = [(mode, shape, dtype) for mode in wave.MODES for shape in MODEL_SHAPES for dtype in DTYPES]


@pytest.mark.parametrize("mode,shape,dtype", CASES)
def test_offline(mode, shape, dtype, tmp_path):
    label, h, w, m, coarse, _ = shape
    cfg = wave.config(mode, 1024, h, w, m, coarse)
    tm = 16 if mode == wave.MODES[2] else 32
    assert cfg.grid * cfg.threads == 1024 * (w // (1 + coarse)) * m
    assert cfg.threads % tm == 0 and cfg.smem == cfg.threads * (h + h // (1 + coarse)) * 8
    code = wave.source(mode, h, w, m, coarse, dtype)
    assert "src[y*N+tid]=q;" in code
    if not shutil.which("nvcc") or not shutil.which("cuobjdump"):
        pytest.skip("CUDA toolkit required for offline assembly")
    cubin, log = wave.compile_cubin(code, tmp_path)
    assert "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads" in log
    sass = subprocess.check_output(["cuobjdump", "--dump-sass", cubin], text=True)
    hmma = sass.count("HMMA.16816")
    assert bool(hmma) == (mode == wave.MODES[2])
    regs = int(re.search(r"Used (\d+) registers", log)[1])
    warps = cfg.threads // 32
    resident = min(
        32,
        64 // warps,
        (228 * 1024) // (1024 + cfg.smem),
        65536 // (warps * 256 * ((regs + 7) // 8)),
    )
    if dest := os.environ.get("WAVE58B_ASSEMBLY_JSONL"):
        with open(dest, "a") as stream:
            print(
                json.dumps([mode, label, str(dtype), regs, cfg.smem, resident * warps, hmma]),
                file=stream,
            )


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
    path[1].zero_()
    path[3].zero_()
    args = (pole, states, aux, path, coarse, 1e-6)
    fp = scans._reference_vertical(
        pole, states.float(), aux, [p.float() for p in path], coarse, 1e-6
    )
    baseline = scans._vscan(*args, False)
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
            assert "lnet.wave58b_packed" in graph.code
            assert graph(states).shape == ((2, 3, 7, 7, 128) if planar else (3, 7, 7, 256))


@gpu
@pytest.mark.parametrize("mode", wave.MODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_odd_coarse(mode, dtype):
    # The last input row contributes to the reverse recurrence. Zero padding
    # gives an FP32 reference for the production odd-coarse dtype restriction.
    pole, states, aux, path, *_ = make_inputs(3, 7, 7, 128, True, dtype)
    pad = torch.nn.functional.pad
    extended = pad(states.float(), (0, 0, 0, 1, 0, 1))
    gains = [pad(v, (0, 0, 0, 1), value=1) for v in aux]
    refs = scans._reference_vertical(pole, extended, gains, [p.float() for p in path], True, 1e-6)
    out = wave.vscan_fwd(pole, states, aux, path, True, mode=mode)
    for value, ref in zip(out, refs, strict=True):
        ref = ref[:, :3, :3]
        assert (value.float() - ref).norm() / ref.norm().clamp_min(1e-12) < (
            0.025 if dtype == torch.bfloat16 else 0.004
        )
