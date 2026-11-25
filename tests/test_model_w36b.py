"""Wave 36B compiler policy, CUDA Graph regression checks."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import pytest
import torch
from run_va_eight_ablations import build_variant
from torch import nn

from lnet import capture_safe_orthogonal, memory_backbone, model_w28c, model_w29c
from lnet.capture_safe_orthogonal import fixed_taylor_matrix_exp
from lnet.model_w36b import (
    _clip_static_grads,
    _tensor_update,
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
    ("cuda_graphs,tma_gemm,max_autotune_pointwise,shape_padding,compiled_update", {
        "triton.cudagraphs": True, "triton.enable_persistent_tma_matmul": True,
        "max_autotune_gemm": True, "max_autotune_gemm_backends": "ATEN,TRITON",
        "max_autotune_gemm_search_space": "DEFAULT", "aggressive_fusion": False,
        "max_autotune_pointwise": True, "epilogue_fusion": True, "shape_padding": True,
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


@pytest.mark.parametrize("extra", [
    "",
    ",inductor_norms",
    ",inductor_norms,aggressive_epilogue",
    ",inductor_norms,aggressive_epilogue,multi_kernel",
])
def test_inductor_fusion_options(monkeypatch: pytest.MonkeyPatch, extra: str) -> None:
    monkeypatch.setenv("LNET_MODEL_W36B", f"cuda_graphs,tma_gemm,max_autotune_pointwise{extra}")
    monkeypatch.setattr(torch, "compile", lambda _, **kwargs: kwargs)
    triton = torch._inductor.config.triton
    expected = {
        "triton.cudagraphs": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
        "max_autotune_gemm_search_space": "DEFAULT",
        "aggressive_fusion": "aggressive_epilogue" in extra,
        "max_autotune_pointwise": True,
        "epilogue_fusion": True,
    }
    if hasattr(triton, "enable_persistent_tma_matmul"):
        expected["triton.enable_persistent_tma_matmul"] = True
    if "aggressive_epilogue" in extra:
        for name in ("tiling_prevents_pointwise_fusion", "tiling_prevents_reduction_fusion"):
            if hasattr(triton, name):
                expected[f"triton.{name}"] = False
    if "multi_kernel" in extra and hasattr(triton, "multi_kernel"):
        expected["triton.multi_kernel"] = 1
    assert compile_model_w36b(nn.Linear(4, 4)) == {
        "options": expected, "fullgraph": True, "dynamic": False,
    }


def test_gemm_respects_explicit_aggressive_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W36B", "tma_gemm")
    monkeypatch.setattr(torch, "compile", lambda _, **kwargs: kwargs)
    compiled = compile_model_w36b(nn.Linear(4, 4), options={"aggressive_fusion": True})
    assert compiled["options"]["aggressive_fusion"] is True


@pytest.mark.parametrize("policy", ["off", "cuda_graphs,tma_gemm,max_autotune_pointwise"])
def test_inductor_norms_dispatch(monkeypatch: pytest.MonkeyPatch, policy: str) -> None:
    device = torch.device("cuda")
    monkeypatch.setenv("LNET_MODEL_OPT", "auto")
    monkeypatch.setenv("LNET_MODEL_W28C", "combined")
    monkeypatch.setenv("LNET_MODEL_W29C", "combined_full")
    monkeypatch.setattr(model_w28c, "model_cuda_supported", lambda _: True)
    monkeypatch.setattr(model_w29c, "model_cuda_supported", lambda _: True)
    monkeypatch.setenv("LNET_MODEL_W36B", policy)

    def selections() -> tuple[bool, ...]:
        return (
            model_w28c.model_w28c_enabled(device, "swiglu"),
            model_w29c.model_w29c_enabled(device, "swiglu_vec2d"),
            model_w29c.model_w29c_enabled(device, "rmsnorm_bwd"),
            model_w29c.model_w29c_enabled(device, "rmsnorm_full"),
            model_w28c.model_w28c_enabled(device, "nr4"),
            model_w29c.model_w29c_enabled(device, "nr4"),
        )

    assert selections() == (True, True, True, True, True, True)
    monkeypatch.setenv("LNET_MODEL_W36B", "cuda_graphs, inductor_norms")
    assert selections() == (False, False, False, False, True, True)


def test_inductor_norms_native_rmsnorm_fullgraph(monkeypatch: pytest.MonkeyPatch) -> None:
    norm = memory_backbone.ChannelRMSNorm(4)
    inputs = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
    graphs = []

    def record(graph: torch.fx.GraphModule, _inputs: list[torch.Tensor]):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(norm, backend=record, fullgraph=True)
    monkeypatch.setenv("LNET_MODEL_W36B", "off")
    compiled(inputs)
    monkeypatch.setenv("LNET_MODEL_W36B", "inductor_norms")
    actual = compiled(inputs)
    actual.sum().backward()

    assert len(graphs) == 2
    assert not any(node.target is torch.rms_norm for node in graphs[0].graph.nodes)
    assert any(node.target is torch.rms_norm for node in graphs[1].graph.nodes)
    expected = torch.nn.functional.rms_norm(
        inputs, (4,), norm.weight.to(inputs.dtype), norm.epsilon,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert inputs.grad is not None
    assert inputs.grad.isfinite().all()


@pytest.mark.parametrize("features", [
    "", "typo", "cuda_graphs,", "off,compiled_update",
    "max_autotune_gemm,max_autotune_gemm_aten", "tma_gemm,max_autotune_gemm_aten",
])
def test_invalid_features(monkeypatch: pytest.MonkeyPatch, features: str) -> None:
    monkeypatch.setenv("LNET_MODEL_W36B", features)
    with pytest.raises(ValueError, match="LNET_MODEL_W36B"):
        model_w36b_features()


@pytest.mark.parametrize(("fused", "backend"), [
    (False, "eager"), (True, "eager"), (True, "aot_eager"),
])
def test_compiled_update_parity(monkeypatch: pytest.MonkeyPatch, fused: bool, backend: str) -> None:
    torch.manual_seed(52)
    eager = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2), nn.Linear(2, 1))
    model = deepcopy(eager)
    baseline = torch.optim.AdamW(eager.parameters(), fused=fused, capturable=fused)
    optimizer = torch.optim.AdamW(model.parameters(), fused=fused, capturable=fused)
    monkeypatch.setenv("LNET_MODEL_W36B", "off")
    assert compile_model_w36b_update(model, optimizer) is None
    monkeypatch.setenv("LNET_MODEL_W36B", "compiled_update")
    assert compile_model_w36b(model) is model
    update = compile_model_w36b_update(model, optimizer, backend=backend)
    assert update is not None
    for index, scale in enumerate((10.0, 0.1, 50.0, 0.25, 1.0)):
        # The final layer is unused on most iterations. Its stale zero buffer
        # must not cause a momentum or weight-decay update on the last step.
        inputs = torch.randn(6, 4) * scale
        eager[:2 if index != 3 else 3](inputs).sum().backward()
        model[:2 if index != 3 else 3](inputs).sum().backward()
        expected_norm = nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
        expected_clipped = [p.grad for p in eager.parameters() if p.grad is not None]
        clipped = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        if index == 4 and fused:
            clipped = clipped[:len(expected_clipped)]
        torch.testing.assert_close(_clip_static_grads(clipped), expected_norm, rtol=0, atol=0)
        torch.testing.assert_close(clipped, expected_clipped, rtol=0, atol=0)
        baseline.step()
        baseline.zero_grad(set_to_none=True)
        torch.testing.assert_close(update(), expected_norm, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), eager.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), baseline.state_dict(), rtol=0, atol=0)
        for parameter in model.parameters():
            if fused and parameter.grad is not None:
                assert not parameter.grad.count_nonzero()
            else:
                assert parameter.grad is None
    if fused:
        assert all(parameter.grad is None for parameter in model[-1].parameters())
        model[:2](inputs).sum().backward()
        active = {p: p.grad for p in model.parameters() if p.grad is not None}
        report = torch._dynamo.explain(_tensor_update)(
            list(active.values()), update._inputs(active), functional=backend == "aot_eager",
        )
        assert (report.graph_count, report.graph_break_count) == (1, 0), report.break_reasons


@requires_cuda
def test_compiled_update_cudagraph_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W36B", "cuda_graphs,compiled_update")
    torch.manual_seed(52)
    eager = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).cuda()
    model = deepcopy(eager)
    forward = compile_model_w36b(model)
    baseline = torch.optim.AdamW(eager.parameters(), fused=True, capturable=True)
    optimizer = torch.optim.AdamW(model.parameters(), fused=True, capturable=True)
    update = compile_model_w36b_update(model, optimizer)
    inputs = torch.empty(16, 8, device="cuda")
    torch._dynamo.mark_static_address(inputs)
    for index in range(5):
        inputs.normal_().mul_(index + 1)
        # Both optimizers receive identical gradients while compiled backward
        # still exercises accumulation into the persistent grad tensors.
        mark_model_w36b_step_begin()
        with count_cudagraph_replays() as forwards:
            loss = forward(inputs).sum()
        with count_cudagraph_replays() as backwards:
            loss.backward()
        for ref, param in zip(eager.parameters(), model.parameters(), strict=True):
            ref.grad = param.grad.clone()
        expected = nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
        baseline.step()
        baseline.zero_grad(set_to_none=True)
        with count_cudagraph_replays() as updates:
            actual = update()
        if index >= 3:
            assert min(len(forwards), len(backwards), len(updates)) >= 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
        torch.testing.assert_close(model.state_dict(), eager.state_dict(), rtol=0, atol=1e-7)
        torch.testing.assert_close(optimizer.state_dict(), baseline.state_dict(), rtol=0, atol=1e-7)
        assert all(p.grad is not None and not p.grad.count_nonzero() for p in model.parameters())
        del loss


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


def _assert_training_parity(model, reference, actual, expected, *, atol, rtol, label) -> None:
    pairs = defaultdict(list, {
        "logits": [(actual[0], expected[0])],
        "loss": [(actual[1], expected[1])],
    })
    pairs["grads"] = [
        (got.grad, want.grad)
        for got, want in zip(model.parameters(), reference.parameters(), strict=True)
        if got.grad is not None or want.grad is not None
    ]
    assert len(pairs["grads"]) == 193
    assert all(got is not None and want is not None for got, want in pairs["grads"])
    buffers = dict(reference.named_buffers())
    for name, buffer in model.named_buffers():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in {"running_ms", "ema_weight"}:
            pairs[suffix].append((buffer, buffers[name]))
    with torch.no_grad():
        maxima = {
            name: torch.stack([(got.float() - want.float()).abs().max() for got, want in values])
            .max().item()
            for name, values in pairs.items()
        }
    print(f"{label} max_abs = {maxima}")  # noqa: T201
    for name, values in pairs.items():
        tolerance = {"atol": 1e-6, "rtol": 1e-6} if name == "ema_weight" else {
            "atol": atol, "rtol": rtol,
        }
        torch.testing.assert_close(*zip(*values, strict=True), **tolerance)


def _graph_train_step(compiled, sample, bf16: bool):
    mark_model_w36b_step_begin()
    with count_cudagraph_replays() as forward, torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=bf16,
    ):
        actual = compiled(sample)[0]
    loss = actual.float().square().mean()
    with count_cudagraph_replays() as backward:
        loss.backward()
    return (actual, loss), (len(forward), len(backward))


def _graph_replay_self_consistency(model, compiled, sample, bf16: bool) -> None:
    buffers = [(buf, buf.clone()) for buf in model.buffers()]
    snapshots = []
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for buffer, initial in buffers:
                buffer.copy_(initial)
        (logits, loss), replays = _graph_train_step(compiled, sample, bf16)
        assert min(replays) >= 1
        snapshots.append((
            logits.detach().clone(), loss.detach().clone(),
            [p.grad.clone() for p in model.parameters() if p.grad is not None],
            [buf.clone() for buf, _ in buffers],
        ))
        del logits, loss
    torch.testing.assert_close(*snapshots, atol=1e-6, rtol=1e-6)


@requires_cuda
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16"])
def test_variant_compiled_cudagraph_replay(monkeypatch: pytest.MonkeyPatch, caplog, bf16) -> None:
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        _check_variant_compiled_cudagraph_replay(monkeypatch, caplog, bf16)
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def _check_variant_compiled_cudagraph_replay(monkeypatch, caplog, bf16) -> None:
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
    no_graph = compile_model_w36b(no_graph_model, options={"triton.cudagraphs": False})
    step_replays = []
    tolerance = 5e-3 if bf16 else 1e-4
    for index in range(5):
        sample.mul_(-1)
        for arm in (model, eager, no_graph_model):
            arm.zero_grad(set_to_none=True)
        actual, replays = _graph_train_step(compiled, sample, bf16)
        step_replays.append(replays)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            no_graph_logits = no_graph(sample)[0]
            expected = eager(sample)[0] if bf16 else None
        no_graph_loss = no_graph_logits.float().square().mean()
        no_graph_loss.backward()
        _assert_training_parity(
            model, no_graph_model, actual, (no_graph_logits, no_graph_loss),
            atol=tolerance, rtol=tolerance, label=f"step {index}, graphs vs no graphs",
        )
        if expected is not None:
            reference = expected.float().square().mean()
            reference.backward()
            _assert_training_parity(
                model, eager, actual, (expected, reference),
                atol=2e-2, rtol=2e-2, label=f"step {index}, graphs vs eager",
            )
            del expected, reference
        del actual, no_graph_logits, no_graph_loss
    _graph_replay_self_consistency(model, compiled, sample, bf16)
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
