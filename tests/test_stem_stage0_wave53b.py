"""Stage 0 launch geometry, offline Hopper compilation and CUDA selftests."""

import re
import subprocess
import sys

import bench_wave53b_stem_type1 as bench
import pytest
import torch
from bench_wave52b_stem_type1 import reference_gradients
from bench_wave53b_stem_type1 import DTYPES, LAYOUTS, MODES, check_case, check_large_reductions

from lnet import fused_stem_ops as stem
from lnet.image_layers import LayerNorm2d, _fused_stem_block

V3 = MODES[2:]


@pytest.mark.parametrize("mode", V3)
def test_config(mode, monkeypatch) -> None:
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", mode)
    assert stem.stem_stage0_mode() == mode
    for width in (223, 224):
        assert stem.resolve_stage0_candidate(mode, needs_dx=False, width=width) == mode
        assert stem.resolve_stage0_candidate(mode, needs_dx=True, width=width) == "cudnn_fused"
    x = torch.empty((1024, 3, 224, 224), device="meta")
    f, fw, b, bw, workers = stem._V3_CONFIG[mode]
    assert workers in (1056, 2112, 4224)
    assert fw in (4, 8)
    assert bw in (4, 8)
    for backward, block in ((False, f), (True, b)):
        assert stem._tensor_core_tiles(x, mode, backward=backward) == (
            block,
            1024 * 112**2 // block,
        )


def test_cpu_check_case_has_no_grad_fn_at_capture(monkeypatch) -> None:
    x = torch.randn(1, 3, 5, 7, requires_grad=True)
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
    norm = LayerNorm2d(32)
    dy = torch.randn(1, 32, 3, 4)
    assert all(t.grad_fn is None for t in reference_gradients(x, conv, norm, dy))
    monkeypatch.setattr(bench, "stem_case", lambda *args, **kwargs: (x, conv, norm, dy))
    monkeypatch.setattr(bench, "stem_forward", lambda *_: lambda v: _fused_stem_block(v, conv, norm))
    captures = []

    def capture_probe(step):
        # Inspect check_case at the exact boundary where CUDA capture would begin.
        scope = sys._getframe(1).f_locals
        assert all(t.grad_fn is None for t in scope.values() if isinstance(t, torch.Tensor))
        assert all(t.grad_fn is None for name in ("expected", "eager") for t in scope[name])
        assert all(t.grad_fn is None for t in step())
        captures.append(True)
        return True

    monkeypatch.setattr(bench, "graph_finite", capture_probe)
    check_case(torch.contiguous_format, torch.float32, needs_dx=True, bias=True)
    assert captures == [True]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scenario", ["pass", "aot", "expected", "eager"])
def test_cpu_check_case_compile_parity(dtype, scenario, monkeypatch):
    x = torch.ones(1, 3, 5, 7)
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
    norm = LayerNorm2d(32)
    dy = torch.ones(1, 32, 3, 4)
    monkeypatch.setattr(bench, "stem_case", lambda *args, **kwargs: (x, conv, norm, dy))
    # Exercise the CUDA-only decision with CPU operators and controlled FP32 dw differences.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: tensor is x))
    monkeypatch.setattr(bench, "stem_forward", lambda *_: lambda v: _fused_stem_block(v, conv, norm))
    backends = []
    eager_limit = 0.015 if dtype == torch.bfloat16 else 0.005
    ref_limit = 0.04 if dtype == torch.bfloat16 else 0.008

    def compile_probe(forward, *, fullgraph, backend="inductor"):
        assert fullgraph
        backends.append(backend)
        return (forward, backend)

    def step(forward, *args):
        backend = "eager"
        if isinstance(forward, tuple):
            forward, backend = forward
        y, dw, *rest = run_step(forward, *args)
        shift = ref_limit - eager_limit / 4 if scenario == "expected" else 0
        if backend == "aot_eager" and scenario == "aot":
            shift += eager_limit / 2
        if backend == "inductor":
            shift += eager_limit * (1.25 if scenario == "eager" else 0.5)
        return (y, dw * (1 + shift), *rest)

    run_step = bench.run_step
    monkeypatch.setattr(torch, "compile", compile_probe)
    monkeypatch.setattr(bench, "run_step", step)
    monkeypatch.setattr(bench, "graph_finite", lambda _: True)
    if scenario == "pass":
        check_case(torch.contiguous_format, dtype, needs_dx=False, bias=True)
    else:
        message = {"aot": "not close", "expected": "Inductor vs FP32", "eager": "Inductor vs eager"}
        with pytest.raises(AssertionError, match=message[scenario]):
            check_case(torch.contiguous_format, dtype, needs_dx=False, bias=True)
    assert backends == (["aot_eager"] if scenario == "aot" else ["aot_eager", "inductor"])


def _compile(kernel, pointers, constants, warps, stages, tmp_path, spill_limit=(0, 0)):
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget  # noqa: PLC0415
    from triton.compiler import ASTSource  # noqa: PLC0415

    signature = dict(zip(kernel.arg_names, pointers + ["constexpr"] * len(constants), strict=True))
    aligned = {(i,): [["tt.divisibility", 16]] for i in range(len(pointers))}
    source = ASTSource(
        kernel,
        signature,
        dict(zip(kernel.arg_names[len(pointers) :], constants, strict=True)),
        attrs=aligned,
    )
    compiled = triton.compile(
        source, target=GPUTarget("cuda", 90, 32), options={"num_warps": warps, "num_stages": stages}
    )
    ptx = tmp_path / "kernel.ptx"
    ptx.write_text(compiled.asm["ptx"])
    result = subprocess.run(
        [
            triton.knobs.nvidia.ptxas.path,
            "-lineinfo",
            "-v",
            "--gpu-name=sm_90a",
            str(ptx),
            "-o",
            str(tmp_path / "kernel.cubin"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    regs = int(re.search(r"Used (\d+) registers", result.stderr)[1])
    spills = tuple(
        map(
            int,
            re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", result.stderr).groups(),
        )
    )
    print(f" regs={regs} shared={compiled.metadata.shared} spill={spills}", flush=True)  # noqa: T201
    assert 0 < regs <= 255
    assert compiled.metadata.shared <= 227328
    assert all(actual <= limit for actual, limit in zip(spills, spill_limit, strict=True))


@pytest.mark.parametrize("mode", V3)
@pytest.mark.parametrize("bwd", [False, True])
@pytest.mark.parametrize("cl", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_sm90_stage(mode, bwd, cl, bias, tmp_path) -> None:
    f, fw, b, bw, _ = stem._V3_CONFIG[mode]
    _compile(
        stem.stem_conv_tensor_stages,
        ["*bf16"] * 2 + ["*fp32"] * 3 + ["*bf16"] * 2 + ["*fp32"] * 2,
        [
            224,
            224,
            3,
            32,
            1024 * 112**2,
            1e-5,
            cl,
            bias,
            bwd,
            b if bwd else f,
            32,
            True,
            True,
            False,
        ],
        bw if bwd else fw,
        2 if bwd else 3,
        tmp_path,
        (24, 12) if bwd and bias and mode == "triton_v3_m128_w8" else (0, 0),
    )


@pytest.mark.parametrize("workers", [1024, 1056, 2112, 4224])
@pytest.mark.parametrize("bias", [False, True])
def test_sm90_reduce(workers, bias, tmp_path) -> None:
    _compile(
        stem.stem_conv_tensor_reduce,
        ["*fp32", "*bf16", "*fp32", "*fp32", "*fp32"],
        [3, 32, bias, workers, 256 if workers > 1024 else 1024],
        4,
        3,
        tmp_path,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_reductions() -> None:
    check_large_reductions()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("layout", LAYOUTS.values())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dx", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_cuda(mode, layout, dtype, dx, bias, monkeypatch) -> None:
    monkeypatch.setenv("LNET_FUSED_STEM_CONV", "on")
    monkeypatch.setenv("LNET_STEM_STAGE0_MODE", mode)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        check_case(layout, dtype, needs_dx=dx, bias=bias)
