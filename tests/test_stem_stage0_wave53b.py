"""Stage 0 launch geometry, offline Hopper compilation and CUDA selftests."""

import re
import subprocess

import pytest
import torch
from bench_wave53b_stem_type1 import DTYPES, LAYOUTS, MODES, check_case, check_large_reductions

from lnet import fused_stem_ops as stem

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
