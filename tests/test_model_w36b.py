"""Wave 36B compiler policy, CUDA Graph regression checks."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from run_va_eight_ablations import build_variant
from torch import nn

from lnet import capture_safe_orthogonal
from lnet.capture_safe_orthogonal import fixed_taylor_matrix_exp
from lnet.model_w36b import (
    compile_model_w36b,
    count_cudagraph_replays,
    mark_model_w36b_step_begin,
    model_w36b_features,
    prepare_model_w36b_,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_compile_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    model = nn.Linear(4, 4)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda module, **kwargs: calls.append(kwargs) or module)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: calls.append("step"))
    monkeypatch.setenv("LNET_MODEL_W36B", "off")
    assert compile_model_w36b(model) is model
    assert not calls
    monkeypatch.setenv("LNET_MODEL_W36B", "cuda_graphs")
    assert compile_model_w36b(model) is model
    assert calls.pop()["mode"] == "reduce-overhead"
    monkeypatch.setenv("LNET_MODEL_W36B", "cuda_graphs, max_autotune_gemm")
    assert compile_model_w36b(model) is model
    assert calls.pop()["options"] == {
        "triton.cudagraphs": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "aggressive_fusion": False,
    }
    mark_model_w36b_step_begin()
    assert calls == ["step"]
    monkeypatch.setenv("LNET_MODEL_W36B", "cuda_graphs,max_autotune_gemm_aten")
    assert compile_model_w36b(model) is model
    assert calls.pop()["options"]["max_autotune_gemm_backends"] == "ATEN"
    monkeypatch.setenv("LNET_MODEL_W36B", "off,cuda_graphs")
    with pytest.raises(ValueError, match="LNET_MODEL_W36B"):
        model_w36b_features()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("autocast", [False, True])
def test_ortho_mm_custom_op_parity_and_fullgraph(
    device: str, autocast: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    if device == "cuda" and autocast and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16")
    torch.manual_seed(25)
    left = torch.randn(8, 8, device=device, requires_grad=True)
    right = torch.randn(8, 8, device=device, requires_grad=True)
    matrix = (torch.randn(8, 8, device=device) * 0.1).requires_grad_()
    mm = torch.ops.lnet.ortho_mm.default

    def run(a, b, m):
        return mm(a, b), fixed_taylor_matrix_exp(m)

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        actual = compiled(left, right, matrix)
    with monkeypatch.context() as patch:
        patch.setattr(capture_safe_orthogonal, "_ortho_mm", torch.mm)
        expected = torch.mm(left, right), fixed_taylor_matrix_exp(matrix)
    actual_grads = torch.autograd.grad(
        actual, (left, right, matrix), tuple(map(torch.ones_like, actual)),
    )
    expected_grads = torch.autograd.grad(
        expected, (left, right, matrix), tuple(map(torch.ones_like, expected)),
    )
    for got, want in zip((*actual, *actual_grads), (*expected, *expected_grads), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def _cuda_sample():
    if not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16")
    torch.manual_seed(501)
    return torch.randn(2, 3, 224, 224, device="cuda").contiguous(
        memory_format=torch.channels_last,
    )


@requires_cuda
def test_variant_compiled_cudagraph_replay(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setenv("LNET_MODEL_W36B", "cuda_graphs")
    skip_logger = torch._logging.getArtifactLogger("torch._inductor.cudagraph_utils", "perf_hints")
    monkeypatch.setattr(skip_logger, "propagate", True)
    caplog.set_level("WARNING", logger=skip_logger.name)
    sample = _cuda_sample()
    torch._dynamo.mark_static_address(sample)
    eager = prepare_model_w36b_(build_variant("17-p1-nr4-1024d-v3").cuda())
    model = deepcopy(eager)
    compiled = compile_model_w36b(model)
    step_replays = []
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        eager.zero_grad(set_to_none=True)
        mark_model_w36b_step_begin()
        with count_cudagraph_replays() as forward, torch.autocast("cuda", dtype=torch.bfloat16):
            actual = compiled(sample)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = eager(sample)[0]
            loss = actual.float().square().mean()
            reference = expected.float().square().mean()
        with count_cudagraph_replays() as backward:
            loss.backward()
        step_replays.append((len(forward), len(backward)))
        reference.backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        grads = [(got.grad, want.grad) for got, want in zip(
            model.parameters(), eager.parameters(), strict=True,
        ) if got.grad is not None or want.grad is not None]
        assert len(grads) == 193
        for got, want in grads:
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        del actual, expected, loss, reference, grads
    skips = [m for m in caplog.messages if "skipping cudagraph" in m.lower()]
    print(f"CUDA Graph replays (forward, backward) per step = {step_replays}; skips = {skips}")  # noqa: T201
    assert all(min(counts) >= 1 for counts in step_replays[-2:]), (step_replays, skips)
    for name, buffer in model.named_buffers():
        if name.endswith(("running_ms", "ema_weight")):
            torch.testing.assert_close(buffer, dict(eager.named_buffers())[name])


@requires_cuda
@pytest.mark.parametrize("features", ["capture_safe_ortho,max_autotune_gemm",
                                     "cuda_graphs,max_autotune_gemm"])
def test_variant_ortho_plus_max_autotune_gemm(
    monkeypatch: pytest.MonkeyPatch, features: str,
) -> None:
    monkeypatch.setenv("LNET_MODEL_W36B", features)
    sample = _cuda_sample()
    model = build_variant("17-p1-nr4-1024d-v3").cuda()
    compiled = compile_model_w36b(model)
    mark_model_w36b_step_begin()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = compiled(sample)[0].float().square().mean()
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert len(grads) == 193
    assert loss.isfinite()
    assert all(gradient.isfinite().all() for gradient in grads)
