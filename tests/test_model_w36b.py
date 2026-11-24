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
    compile_model_w36b_update,
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


@pytest.mark.parametrize(("features", "expected"), [
    ("tma_gemm,max_autotune_gemm", {
        "triton.enable_persistent_tma_matmul": True, "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON", "max_autotune_gemm_search_space": "DEFAULT",
        "aggressive_fusion": False,
    }),
    ("max_autotune_pointwise", {"max_autotune_pointwise": True, "epilogue_fusion": True}),
    ("cuda_graphs,max_autotune_gemm,coord_descent,aggressive_fusion", {
        "triton.cudagraphs": True, "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON", "aggressive_fusion": True,
        "coordinate_descent_tuning": True,
    }),
])
def test_autotune_features(monkeypatch: pytest.MonkeyPatch, features, expected) -> None:
    assert expected.keys() <= torch._inductor.config.get_config_copy().keys()
    monkeypatch.setenv("LNET_MODEL_W36B", features)
    monkeypatch.setattr(torch, "compile", lambda _, **kwargs: kwargs)
    options = compile_model_w36b(nn.Linear(4, 4))
    assert options["options"] == expected
    assert options["fullgraph"] == ("cuda_graphs" in features)


@pytest.mark.parametrize("features", [
    "", "typo", "cuda_graphs,", "off,compiled_update",
    "max_autotune_gemm,max_autotune_gemm_aten", "tma_gemm,max_autotune_gemm_aten",
])
def test_invalid_features(monkeypatch: pytest.MonkeyPatch, features: str) -> None:
    monkeypatch.setenv("LNET_MODEL_W36B", features)
    with pytest.raises(ValueError, match="LNET_MODEL_W36B"):
        model_w36b_features()


@pytest.mark.parametrize("fused", [False, True])
def test_compiled_update_parity(monkeypatch: pytest.MonkeyPatch, fused: bool) -> None:
    torch.manual_seed(52)
    eager = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2), nn.Linear(2, 1))
    model = deepcopy(eager)
    baseline = torch.optim.AdamW(eager.parameters(), fused=fused)
    optimizer = torch.optim.AdamW(model.parameters(), fused=fused)
    monkeypatch.setenv("LNET_MODEL_W36B", "off")
    assert compile_model_w36b_update(model, optimizer) is None
    monkeypatch.setenv("LNET_MODEL_W36B", "compiled_update")
    assert compile_model_w36b(model) is model
    update = compile_model_w36b_update(model, optimizer, backend="aot_eager")
    assert update is not None
    for scale in (10.0, 0.1, 50.0):
        for got, want in zip(model.parameters(), eager.parameters(), strict=True):
            # The final layer is intentionally unused.
            if got.numel() > 2:
                want.grad = torch.randn_like(want) * scale
                got.grad = want.grad.clone()
        clipped = [param.grad for param in model.parameters() if param.grad is not None]
        expected_norm = nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
        expected_clipped = [param.grad for param in eager.parameters() if param.grad is not None]
        baseline.step()
        baseline.zero_grad(set_to_none=True)
        torch.testing.assert_close(update(), expected_norm, rtol=0, atol=0)
        torch.testing.assert_close(clipped, expected_clipped, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), eager.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), baseline.state_dict(), rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in model.parameters())


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


def _assert_training_parity(model, reference, actual, expected, *, atol, rtol) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    grads = [(got.grad, want.grad) for got, want in zip(
        model.parameters(), reference.parameters(), strict=True,
    ) if got.grad is not None or want.grad is not None]
    assert len(grads) == 193
    for got, want in grads:
        assert got is not None
        assert want is not None
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
    buffers = dict(reference.named_buffers())
    for name, buffer in model.named_buffers():
        if name.endswith("running_ms"):
            torch.testing.assert_close(buffer, buffers[name], atol=atol, rtol=rtol)
        elif name.endswith("ema_weight"):
            # EMA weight depends on step count and momentum, not the BF16 activations.
            torch.testing.assert_close(buffer, buffers[name], atol=1e-6, rtol=1e-6)


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
    no_graph_model = deepcopy(eager)
    with monkeypatch.context() as patch:
        patch.setenv("LNET_MODEL_W36B", "capture_safe_ortho")
        prepare_model_w36b_(no_graph_model)
    # Keep the Dynamo graph and lowering options common to both compiled arms.
    options = dict(torch._inductor.list_mode_options("reduce-overhead"))
    options["triton.cudagraphs"] = False
    no_graph = torch.compile(no_graph_model, options=options, fullgraph=True, dynamic=False)
    step_replays = []
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        eager.zero_grad(set_to_none=True)
        no_graph_model.zero_grad(set_to_none=True)
        mark_model_w36b_step_begin()
        with count_cudagraph_replays() as forward, torch.autocast("cuda", dtype=torch.bfloat16):
            actual = compiled(sample)[0]
        loss = actual.float().square().mean()
        with count_cudagraph_replays() as backward:
            loss.backward()
        step_replays.append((len(forward), len(backward)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = eager(sample)[0]
            no_graph_logits = no_graph(sample)[0]
        reference = expected.float().square().mean()
        no_graph_loss = no_graph_logits.float().square().mean()
        reference.backward()
        no_graph_loss.backward()
        _assert_training_parity(
            model, eager, (actual, loss), (expected, reference), atol=2e-2, rtol=2e-2,
        )
        _assert_training_parity(
            model, no_graph_model, (actual, loss), (no_graph_logits, no_graph_loss),
            atol=1e-4, rtol=1e-4,
        )
        del actual, loss, no_graph_logits, no_graph_loss, expected, reference
    skips = [m for m in caplog.messages if "skipping cudagraph" in m.lower()]
    print(f"CUDA Graph replays (forward, backward) per step = {step_replays}; skips = {skips}")  # noqa: T201
    assert all(min(counts) >= 1 for counts in step_replays[-2:]), (step_replays, skips)
    assert not skips


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
