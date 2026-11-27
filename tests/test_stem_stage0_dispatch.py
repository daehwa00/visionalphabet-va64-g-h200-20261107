"""Stage 0 launch geometry, offline Hopper compilation and CUDA selftests."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from alphabet import fused_stem_ops as stem
from alphabet.fused_stem_ops import STAGE0_CANDIDATES, stem_conv_tensor_reduce
from alphabet.image_layers import LayerNorm2d, _fused_stem_block

if TYPE_CHECKING:
    from collections.abc import Callable

MODES = ("off", *STAGE0_CANDIDATES)
DTYPES = (torch.bfloat16, torch.float16)
LAYOUTS = {"nchw": torch.contiguous_format, "channels_last": torch.channels_last}


def stem_case(
    shape: tuple[int, int, int, int],
    layout: torch.memory_format,
    dtype: torch.dtype,
    *,
    needs_dx: bool,
) -> tuple[Tensor, nn.Conv2d, LayerNorm2d, Tensor]:
    torch.manual_seed(185)
    x = torch.randn(shape, device="cuda").to(dtype).contiguous(memory_format=layout)
    x.requires_grad_(needs_dx)
    conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, device="cuda")
    norm = LayerNorm2d(32).cuda()
    dy = torch.randn_like(x[:, :1, ::2, ::2].expand(-1, 32, -1, -1))
    return x, conv, norm, dy


def stem_forward(conv: nn.Conv2d, norm: LayerNorm2d, dtype: torch.dtype) -> Callable:
    def forward(x: Tensor) -> Tensor:
        with torch.autocast("cuda", dtype=dtype):
            return _fused_stem_block(x, conv, norm)

    return forward


def stem_parameters(
    x: Tensor, conv: nn.Conv2d, norm: LayerNorm2d,
) -> tuple[Tensor, ...]:
    params = (conv.weight, norm.norm.weight, norm.norm.bias)
    if conv.bias is not None:
        params += (conv.bias,)
    return (*params, x) if x.requires_grad else params


def reference_gradients(
    x: Tensor, conv: nn.Conv2d, norm: LayerNorm2d, dy: Tensor,
) -> tuple[Tensor, ...]:
    x_fp32 = x.detach().float().requires_grad_(x.requires_grad)
    z = functional.conv2d(x_fp32, conv.weight, conv.bias, stride=2, padding=1)
    z = functional.layer_norm(
        z.permute(0, 2, 3, 1), (32,), norm.norm.weight, norm.norm.bias, norm.norm.eps,
    )
    y = functional.gelu(z).permute(0, 3, 1, 2)
    grads = torch.autograd.grad(y, stem_parameters(x_fp32, conv, norm), dy.float())
    return (y.detach(), *grads)


def run_step(
    forward: Callable, x: Tensor, params: tuple[Tensor, ...], dy: Tensor,
) -> tuple[Tensor, ...]:
    y = forward(x)
    grads = torch.autograd.grad(y, params, dy)
    return (y.detach(), *grads)


def graph_finite(fn: Callable[[], tuple[Tensor, ...]]) -> bool:
    torch.cuda.synchronize()
    warmup = torch.cuda.Stream()
    with torch.cuda.stream(warmup):
        for _ in range(3):
            # Drop each warmup result before capture on this stream.
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=warmup):
        result = fn()
    graph.replay()
    torch.cuda.synchronize()
    return all(bool(t.isfinite().all()) for t in result)


def check_large_reductions() -> None:
    # Integer-valued FP32 partials have an exact sum for all worker counts.
    for workers in (1056,):
        for bias in (False, True):
            fields = 29 + int(bias)
            partial = torch.randint(-2, 3, (workers, fields, 32), device="cuda").float()
            dw = partial.new_empty((27, 32), dtype=torch.bfloat16)
            dg, db, dc = (partial.new_empty(32) for _ in range(3))
            stem_conv_tensor_reduce[(fields,)](partial, dw, dg, db, dc, 3, 32, bias, workers, 256)
            ref = partial.sum(0)
            torch.testing.assert_close(dw, ref[:27].to(dw.dtype), rtol=0, atol=0)
            torch.testing.assert_close(dg, ref[27], rtol=0, atol=0)
            torch.testing.assert_close(db, ref[28], rtol=0, atol=0)
            if bias:
                torch.testing.assert_close(dc, ref[29], rtol=0, atol=0)


def check_case(
    layout: torch.memory_format, dtype: torch.dtype, *, needs_dx: bool, bias: bool
) -> dict:
    x, conv, norm, dy = stem_case((4, 3, 33, 63), layout, dtype, needs_dx=needs_dx)
    if not bias:
        conv.bias = None
    forward = stem_forward(conv, norm, dtype)
    params = stem_parameters(x, conv, norm)
    x_ref = x.detach().float().requires_grad_(needs_dx)
    z = functional.conv2d(x_ref, conv.weight, conv.bias, stride=2, padding=1)
    z = functional.layer_norm(
        z.permute(0, 2, 3, 1), (32,), norm.norm.weight, norm.norm.bias, norm.norm.eps
    )
    y = functional.gelu(z).permute(0, 3, 1, 2)
    ref_params = stem_parameters(x_ref, conv, norm)
    ref_grads = torch.autograd.grad(y, ref_params, dy.float())
    expected = (y.detach(), *ref_grads)
    del y, z, x_ref, ref_grads, ref_params
    eager = run_step(forward, x, params, dy)
    names = ["y", "dw", "dg", "db"] + (["dc"] if bias else []) + (["dx"] if needs_dx else [])

    def check_relative_l2(actual, reference, limit, comparison):
        errors = {
            name: float((a.float() - r.float()).norm() / r.float().norm().clamp_min(1e-8))
            for name, a, r in zip(names, actual, reference, strict=True)
        }
        if not all(v < limit for v in errors.values()):
            raise AssertionError(f"{comparison} (limit {limit}): {errors}")
        return errors

    limit = 0.04 if dtype == torch.bfloat16 else 0.008
    errors = check_relative_l2(eager, expected, limit, "eager vs FP32")
    torch._dynamo.reset()
    compiled_aot = torch.compile(forward, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(run_step(compiled_aot, x, params, dy), eager)
    if x.is_cuda:
        compiled = torch.compile(forward, fullgraph=True)
        compiled_outputs = run_step(compiled, x, params, dy)
        check_relative_l2(compiled_outputs, expected, limit, "Inductor vs FP32")
        eager_limit = 0.015 if dtype == torch.bfloat16 else 0.005
        check_relative_l2(compiled_outputs, eager, eager_limit, "Inductor vs eager")
        del compiled_outputs
    if not graph_finite(lambda: run_step(forward, x, params, dy)):
        raise AssertionError("Nonfinite CUDA Graph replay")
    return {
        "dtype": str(dtype),
        "needs_dx": needs_dx,
        "bias": bias,
        "relative_l2": errors,
        "compile_fullgraph": True,
        "cuda_graph_finite": True,
    }


V3 = STAGE0_CANDIDATES


@pytest.mark.parametrize("mode", V3)
def test_config(mode, monkeypatch) -> None:
    monkeypatch.setenv("ALPHABET_STEM_STAGE0_MODE", mode)
    assert stem.stem_stage0_mode() == mode
    x = torch.empty((1024, 3, 224, 224), device="meta")
    f, fw, b, bw, workers = stem._V3_CONFIG[mode]
    assert (f, fw, b, bw, workers) == (128, 4, 64, 4, 1056)
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
    monkeypatch.setattr(
        sys.modules[__name__], "stem_case", lambda *_args, **_kwargs: (x, conv, norm, dy)
    )
    monkeypatch.setattr(
        sys.modules[__name__], "stem_forward", lambda *_: lambda v: _fused_stem_block(v, conv, norm)
    )
    captures = []

    def capture_probe(step):
        # Inspect check_case at the exact boundary where CUDA capture would begin.
        scope = sys._getframe(1).f_locals
        assert all(t.grad_fn is None for t in scope.values() if isinstance(t, torch.Tensor))
        assert all(t.grad_fn is None for name in ("expected", "eager") for t in scope[name])
        assert all(t.grad_fn is None for t in step())
        captures.append(True)
        return True

    monkeypatch.setattr(sys.modules[__name__], "graph_finite", capture_probe)
    check_case(torch.contiguous_format, torch.float32, needs_dx=True, bias=True)
    assert captures == [True]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scenario", ["pass", "aot", "expected", "eager"])
def test_cpu_check_case_compile_parity(dtype, scenario, monkeypatch) -> None:
    x = torch.ones(1, 3, 5, 7)
    conv = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
    norm = LayerNorm2d(32)
    dy = torch.ones(1, 32, 3, 4)
    monkeypatch.setattr(
        sys.modules[__name__], "stem_case", lambda *_args, **_kwargs: (x, conv, norm, dy)
    )
    # Exercise the CUDA-only decision with CPU operators and controlled FP32 dw differences.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: tensor is x))
    monkeypatch.setattr(
        sys.modules[__name__], "stem_forward", lambda *_: lambda v: _fused_stem_block(v, conv, norm)
    )
    backends = []
    eager_limit = 0.015 if dtype == torch.bfloat16 else 0.005
    ref_limit = 0.04 if dtype == torch.bfloat16 else 0.008

    def compile_probe(forward, *, fullgraph, backend="inductor"):
        assert fullgraph
        backends.append(backend)
        return (forward, backend)

    def step(forward, *args: object):
        backend = "eager"
        if isinstance(forward, tuple):
            forward, backend = forward
        y, dw, *rest = original_step(forward, *args)
        shift = ref_limit - eager_limit / 4 if scenario == "expected" else 0
        if backend == "aot_eager" and scenario == "aot":
            shift += eager_limit / 2
        if backend == "inductor":
            shift += eager_limit * (1.25 if scenario == "eager" else 0.5)
        return (y, dw * (1 + shift), *rest)

    original_step = run_step
    monkeypatch.setattr(torch, "compile", compile_probe)
    monkeypatch.setattr(sys.modules[__name__], "run_step", step)
    monkeypatch.setattr(sys.modules[__name__], "graph_finite", lambda _: True)
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
        ],
        bw if bwd else fw,
        2 if bwd else 3,
        tmp_path,
    )


@pytest.mark.parametrize("workers", [1024, 1056])
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
    monkeypatch.setenv("ALPHABET_FUSED_STEM_CONV", "on")
    monkeypatch.setenv("ALPHABET_STEM_STAGE0_MODE", mode)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        check_case(layout, dtype, needs_dx=dx, bias=bias)
