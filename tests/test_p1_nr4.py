"""Check Nyquist-bounded poles and the four orthogonal terminal correlations."""

from __future__ import annotations

import math
from copy import deepcopy

import pytest
import torch
from run_va_eight_ablations import build_variant
from torch import Tensor

from lnet.complex_scan_stage import ComplexScanStage
from lnet.image_layers import fold_terminal_rms_for_inference
from lnet.same_resolution_depth import SameResolutionPoleScanBlock
from lnet.terminal_imaginary_correlation import PoleTensors, complex_correlation_descriptor

P1_NR4 = "17-p1-nr4-1024d-v3"
CPU_CUDA = ("cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA",
)))


def test_p1_preserves_effective_poles() -> None:
    stage = ComplexScanStage(8, maximum_phase=2.4, output_modes=None)
    expected_phase = tuple(value.detach().clone() for value in stage._effective_phases())
    expected_damping = stage._damping_fields()
    expected_poles = stage.pole_coefficients((2, 7, 9, 8))
    stage.enable_pole_unclamp_()
    assert stage.bound_phase_nyquist
    assert stage.damping_max == 2.0
    for actual, expected in (
        (stage._effective_phases(), expected_phase),
        (stage._damping_fields(), expected_damping),
        *zip(stage.pole_coefficients((2, 7, 9, 8)), expected_poles, strict=True),
    ):
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=1e-6)

    with torch.no_grad():
        stage.phase_x.fill_(1e4)
        stage.phase_y.fill_(-1e4)
    assert all(bool((phase.abs() < math.pi).all()) for phase in stage._effective_phases())
    block = SameResolutionPoleScanBlock(
        8, reader_rank=4, kernel_size=3, pole_template=stage, post_hidden=12,
    )
    assert block.pole_scan.bound_phase_nyquist
    assert block.pole_scan.damping_max == 2.0
    torch.testing.assert_close(block.pole_scan._effective_phases(), stage._effective_phases())

    def coefficients(inputs: Tensor) -> Tensor:
        px, py = stage.pole_coefficients(tuple(inputs.shape))
        return torch.cat((*px, *py), dim=-1)

    result = torch._dynamo.explain(coefficients)(torch.empty(2, 7, 9, 8))
    assert [result.graph_count, result.graph_break_count] == [1, 0]


def _nr4_reference(
    px: PoleTensors, py: PoleTensors, drive: Tensor, *, normalize: bool,
) -> Tensor:
    def recurrent(drive: Tensor, pole: PoleTensors, axis: int, *, reverse: bool) -> tuple:
        ar, ai, gr, gi = (part.float().flatten() for part in pole)
        decay, gain = torch.complex(ar, ai), torch.complex(gr, gi)
        if reverse:
            decay, gain = decay.conj(), gain.conj()
        writes = gain * drive.movedim(axis, 0)
        state = torch.zeros_like(writes[0])
        states = []
        for write in (writes.flip(0) if reverse else writes).unbind():
            state = decay * state + write
            states.append(state)
        if reverse:
            states.reverse()
        return torch.stack(states).movedim(0, axis), writes.movedim(0, axis)

    dr, di = drive.chunk(2, dim=-1)
    source = torch.complex(dr.float(), di.float())
    correlations = []
    for first, second, axis in ((px, py, 2), (py, px, 1)):
        for reverse in (False, True):
            initial, _ = recurrent(source, first, axis, reverse=reverse)
            initial = torch.complex(initial.real.to(drive.dtype).float(),
                                    initial.imag.to(drive.dtype).float())
            state, writes = recurrent(initial, second, 3 - axis, reverse=False)
            energy = writes.real.square() + writes.imag.square()
            corr = (2 * state * writes.conj() - energy).mean((1, 2))
            if normalize:
                corr = corr / energy.mean((1, 2)).clamp_min(1e-6).detach()
            correlations.append(corr)
    return torch.cat(
        [part.real for part in correlations] + [part.imag for part in correlations], dim=-1,
    )


@pytest.mark.parametrize("device", CPU_CUDA)
@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 3), (2, 7, 7, 128), (2, 7, 11, 37), (2, 14, 14, 160), (2, 28, 28, 96)],
)
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nr4_descriptor_and_gradients(
    device: str, shape: tuple[int, int, int, int], normalize: bool, dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LNET_NR4_FUSED", raising=False)
    torch.manual_seed(501)
    batch, height, width, modes = shape

    def pole() -> PoleTensors:
        phase = torch.linspace(-math.pi + 1e-4, math.pi - 1e-4, modes, device=device)
        radius = torch.linspace(0.7, 0.999, modes, device=device)
        return tuple(value.reshape(1, 1, 1, modes).requires_grad_() for value in (
            radius * phase.cos(), radius * phase.sin(),
            0.3 + 0.2 * torch.rand(modes, device=device), 0.15 * torch.rand(modes, device=device),
        ))

    px, py = pole(), pole()
    drive = torch.randn(batch, height, width, 2 * modes, device=device, dtype=dtype)
    drive.requires_grad_()
    actual = complex_correlation_descriptor(
        px, py, drive, non_redundant_four_directions=True, normalize=normalize,
    )
    expected = _nr4_reference(px, py, drive, normalize=normalize)
    upstream = torch.randn_like(expected)
    args = (drive, *px, *py)
    actual_grads = torch.autograd.grad(actual, args, upstream)
    expected_grads = torch.autograd.grad(expected, args, upstream)
    atol, rtol = (2e-5, 2e-5) if dtype == torch.float32 else (2e-2, 1e-2)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    # The first BF16 scan stores rounded states. Its shared pole gradient can
    # lose relative accuracy at individual taps where the two directions cancel.
    torch.testing.assert_close(
        actual_grads, expected_grads, atol=atol if dtype == torch.float32 else 3.5e-2, rtol=rtol,
    )
    if dtype == torch.bfloat16:
        relative_tolerance = torch.finfo(dtype).eps
        for got, want in zip(actual_grads, expected_grads, strict=True):
            assert (
                (got.float() - want.float()).norm()
                <= relative_tolerance * want.float().norm() + 1e-6
            )


@pytest.mark.parametrize(
    ("variant", "parameters", "descriptor_dim"),
    [
        ("17-a2-clean-512d-live-rms-v3", 2_937_060, 512),
        ("17-p1-pole-unclamp-a2clean-v3", 2_937_060, 512),
        ("17-p1-a2-1024d-v3", 2_988_260, 1024),
        (P1_NR4, 2_988_260, 1024),
    ],
)
def test_variant_geometry(variant: str, parameters: int, descriptor_dim: int) -> None:
    model = build_variant(variant)
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert model.classifier.affine.input_dim == descriptor_dim
    assert model.non_redundant_four_directions is (variant == P1_NR4)
    assert all(
        stage.bound_phase_nyquist is variant.startswith("17-p1")
        for stage in model.modules() if isinstance(stage, ComplexScanStage)
    )


@pytest.mark.parametrize("device", CPU_CUDA)
def test_real_descriptor_folding(device: str) -> None:
    torch.manual_seed(501)
    model = build_variant(P1_NR4).to(device)
    inputs = torch.randn(2, 3, 224, 224, device=device)
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with torch.no_grad():
            model(inputs)
            original = model.eval()(inputs)
            folded = fold_terminal_rms_for_inference(deepcopy(model))
            output = folded(inputs)
        assert original[-1].shape == (2, 1024)
        torch.testing.assert_close(original[-1], output[-1], atol=0, rtol=0)
        torch.testing.assert_close(original[0], output[0], atol=1e-5, rtol=0)
        with pytest.raises(RuntimeError, match="cannot run folded StandardizedAffineModalHead"):
            folded.train()(inputs)
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dynamo_single_graph_on_cuda() -> None:
    model = build_variant(P1_NR4).cuda().to(memory_format=torch.channels_last)
    inputs = torch.randn(2, 3, 224, 224, device="cuda").contiguous(
        memory_format=torch.channels_last,
    )
    for mode in ("train", "eval", "folded"):
        model.train(mode == "train")
        if mode == "folded":
            fold_terminal_rms_for_inference(model)
        with torch.set_grad_enabled(mode == "train"), torch.autocast("cuda", dtype=torch.bfloat16):
            explanation = torch._dynamo.explain(model)(inputs)
            assert [explanation.graph_count, explanation.graph_break_count] == [1, 0], (
                mode, explanation.break_reasons,
            )
            logits = torch.compile(model, mode="default", fullgraph=True)(inputs)[0]
        if mode == "train":
            logits.square().mean().backward()
            assert model.backbone.terminal.phase_y.grad.isfinite().all()
        assert logits.isfinite().all()
