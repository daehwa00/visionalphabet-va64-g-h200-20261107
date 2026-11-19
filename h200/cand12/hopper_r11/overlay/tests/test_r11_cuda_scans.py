"""Tests for R11 NVRTC CUDA vertical and horizontal scan kernels and cache isolation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lnet import r6_dispatch, retained_path_dispatch
from lnet import r11_cuda_hscan as rh
from lnet import r11_cuda_vscan as rv
from lnet.path_backward_layout import _LayoutGraphPass
from lnet.product_scan_reference import (
    _product_scan_coarse4_from_tables_reference,
    _product_scan_full16_from_tables_reference,
    bidirectional_product_scan_reference,
    d4_grouped_cell_path_collapse_reference,
    d4_grouped_path_collapse_reference,
    static_variance_tables_reference,
)


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.float(), expected.float()
    return (a - b).norm().item() / max(b.norm().item(), 1e-8)


def test_r11_opt_out_preserves_legacy_and_r10_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When R11 flags are unset, legacy/R6/R10 dispatch hashes remain exact."""
    for name in (
        "LNET_R6_HBACK",
        "LNET_R6_CABWD",
        "LNET_R7_AFWD",
        "LNET_R7_DIRECT",
        "LNET_R8_HFWD",
        "LNET_R8_HBACK",
        "LNET_R9_RMS_DW",
        "LNET_R10_RMS_DW",
        "LNET_R10_VFWD",
        "LNET_R10_HFWD56_SEG4",
        "LNET_R10_RMS_DW56",
        "LNET_R10_TUNED_LAUNCH",
        "LNET_R10_SEG4PACK2",
        "LNET_R11_CUDA_VFWD",
        "LNET_R11_CUDA_VBACK",
        "LNET_R11_CUDA_HFWD",
        "LNET_R11_CUDA_HBACK",
    ):
        monkeypatch.delenv(name, raising=False)

    base = _LayoutGraphPass("sentinel", None)
    assert r6_dispatch.active_r6_flags() == {}
    assert r6_dispatch.r6_graph_pass(base) is base

    rpd_path = Path(retained_path_dispatch.__file__)
    legacy_bytes = r6_dispatch.legacy_dispatch_source(rpd_path)
    assert b"BEGIN R11 DISPATCH" not in legacy_bytes
    assert b"r11_dispatch" not in legacy_bytes

    for filename in (
        "r6_dispatch.py",
        "retained_path_dispatch.py",
        "product_scan_ops.py",
        "triton_bidirectional_product_scan.py",
    ):
        stripped = r6_dispatch._pre_r11_source(Path(r6_dispatch.__file__).with_name(filename))
        assert b"BEGIN R11 DISPATCH" not in stripped
        assert b"r11_dispatch" not in stripped

    # R10-only graph pass must be distinct from opt-out, and R11 must be distinct from R10.
    monkeypatch.setenv("LNET_R10_VFWD", "1")
    r10_uuid = r6_dispatch.r6_graph_pass(base).uuid()
    assert r10_uuid != base.uuid()

    monkeypatch.setenv("LNET_R11_CUDA_VBACK", "1")
    r11_uuid = r6_dispatch.r6_graph_pass(base).uuid()
    assert r11_uuid != r10_uuid
    assert r11_uuid != base.uuid()


def test_r11_guard_rejects_cpu_and_invalid_shapes() -> None:
    cpu_src = torch.zeros((1, 56, 56, 64), dtype=torch.bfloat16)
    cpu_w = torch.zeros((64, 8, 8), dtype=torch.float32)
    assert not rv.can_use_cuda_vscan(cpu_src, cpu_w, coarse=False)
    assert not rh.can_use_cuda_hscan(cpu_src, cpu_src)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("height", [56, 28, 14])
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("force_single_chunk", [False, True])
def test_r11_cuda_vscan_forward_and_backward_parity(
    height: int,
    coarse: bool,
    force_single_chunk: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(17 + height + int(coarse) * 3 + int(force_single_chunk) * 7)
    b, w, m = 2, height, 64
    eps = 1e-6

    ang_x = torch.randn(m, device=device) * 0.5
    rad_x = torch.empty(m, device=device).uniform_(0.75, 0.95)
    pole_x = tuple(
        t.view(1, 1, 1, m)
        for t in (
            rad_x * ang_x.cos(),
            rad_x * ang_x.sin(),
            torch.randn(m, device=device) * 0.2 + 0.5,
            torch.randn(m, device=device) * 0.2,
        )
    )
    ang_y = torch.randn(m, device=device) * 0.5
    rad_y = torch.empty(m, device=device).uniform_(0.75, 0.95)
    dr = (rad_y * ang_y.cos()).view(1, 1, 1, m).clone().requires_grad_()
    di = (rad_y * ang_y.sin()).view(1, 1, 1, m).clone().requires_grad_()
    gr = (torch.randn(m, device=device) * 0.2 + 0.5).view(1, 1, 1, m).clone().requires_grad_()
    gi = (torch.randn(m, device=device) * 0.2).view(1, 1, 1, m).clone().requires_grad_()

    sra = torch.randn(b, height, w, m, device=device, dtype=torch.bfloat16, requires_grad=True)
    sia = torch.randn(b, height, w, m, device=device, dtype=torch.bfloat16, requires_grad=True)
    srb = torch.randn(b, height, w, m, device=device, dtype=torch.bfloat16, requires_grad=True)
    sib = torch.randn(b, height, w, m, device=device, dtype=torch.bfloat16, requires_grad=True)

    wi = (torch.randn(m, 8, 8, device=device) * 0.25).requires_grad_()
    bi = (torch.randn(m, 8, device=device) * 0.1).requires_grad_()
    wo = (torch.randn(m, 2, 8, device=device) * 0.25).requires_grad_()
    bo = (torch.randn(m, 2, device=device) * 0.1).requires_grad_()

    vx, vy = static_variance_tables_reference(pole_x, (dr, di, gr, gi), w, height)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if coarse:
            cr_ref, ci_ref, _ = _product_scan_coarse4_from_tables_reference(
                (dr, di, gr, gi),
                (sra, sia),
                (srb, sib),
                vx,
                vy,
                epsilon=eps,
                gain_normalization="pointwise",
            )
            or_ref, oi_ref = d4_grouped_path_collapse_reference(cr_ref, ci_ref, wi, bi, wo, bo)
        else:
            fr_ref, fi_ref, _ = _product_scan_full16_from_tables_reference(
                (dr, di, gr, gi),
                (sra, sia),
                (srb, sib),
                vx,
                vy,
                epsilon=eps,
                gain_normalization="pointwise",
            )
            or_ref, oi_ref = d4_grouped_cell_path_collapse_reference(
                fr_ref, fi_ref, wi, bi, wo, bo
            )

    g_or = torch.randn(or_ref.shape, device=device, dtype=torch.bfloat16)
    g_oi = torch.randn(oi_ref.shape, device=device, dtype=torch.bfloat16)
    ref_grads = torch.autograd.grad(
        (or_ref, oi_ref),
        (dr, di, gr, gi, sra, sia, srb, sib, wi, bi, wo, bo),
        (g_or.to(or_ref.dtype), g_oi.to(oi_ref.dtype)),
    )

    if force_single_chunk:
        monkeypatch.setattr(rv, "_choose_x_chunks", lambda *_a, **_k: 1)

    squeeze_out = False
    or_cu, oi_cu = rv.r11_cuda_vscan_forward_op(
        dr.detach(),
        di.detach(),
        gr.detach(),
        gi.detach(),
        sra.detach(),
        sia.detach(),
        srb.detach(),
        sib.detach(),
        vx,
        vy,
        wi.detach(),
        bi.detach(),
        wo.detach(),
        bo.detach(),
        coarse,
        squeeze_out,
        eps,
    )
    cu_grads = rv.r11_cuda_vscan_backward(
        dr.detach(),
        di.detach(),
        gr.detach(),
        gi.detach(),
        sra.detach(),
        sia.detach(),
        srb.detach(),
        sib.detach(),
        vx,
        vy,
        wi.detach(),
        bi.detach(),
        wo.detach(),
        bo.detach(),
        g_or,
        g_oi,
        coarse=coarse,
        epsilon=eps,
    )

    assert _rel_l2(or_cu, or_ref) < 0.01
    assert _rel_l2(oi_cu, oi_ref) < 0.01
    for idx, (cu_g, ref_g) in enumerate(zip(cu_grads, ref_grads, strict=True)):
        actual, expected = cu_g, ref_g
        if coarse and idx in (4, 5):
            actual = cu_g[:, :, 1::2, :]
            expected = ref_g[:, :, 1::2, :]
        elif coarse and idx in (6, 7):
            actual = cu_g[:, :, 0::2, :]
            expected = ref_g[:, :, 0::2, :]
        assert _rel_l2(actual, expected) < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [56, 28, 14])
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("force_single_chunk", [False, True])
def test_r11_cuda_hscan_forward_and_backward_parity(
    width: int,
    coarse: bool,
    force_single_chunk: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(31 + width + int(coarse) * 5 + int(force_single_chunk) * 11)
    b, h, m = 2, width, 64

    ang = torch.randn(m, device=device) * 0.5
    rad = torch.empty(m, device=device).uniform_(0.75, 0.95)
    dr = (rad * ang.cos()).view(1, 1, 1, m).clone().requires_grad_()
    di = (rad * ang.sin()).view(1, 1, 1, m).clone().requires_grad_()
    gr = (torch.randn(m, device=device) * 0.2 + 0.5).view(1, 1, 1, m).clone().requires_grad_()
    gi = (torch.randn(m, device=device) * 0.2).view(1, 1, 1, m).clone().requires_grad_()

    packed_src = torch.randn(
        b, h, width, 2 * m, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    sr, si = packed_src[..., :m], packed_src[..., m:]

    pr_ref, pi_ref, nr_ref, ni_ref = bidirectional_product_scan_reference(
        (dr, di, gr, gi), (sr, si)
    )
    st_cu = rh.r11_cuda_hscan_forward_op(
        dr.detach(), di.detach(), gr.detach(), gi.detach(), sr.detach(), si.detach()
    )
    st_ref = torch.stack((pr_ref, pi_ref, nr_ref, ni_ref), dim=0)
    assert _rel_l2(st_cu, st_ref) < 0.005

    g_pr = torch.randn_like(pr_ref)
    g_pi = torch.randn_like(pi_ref)
    g_nr = torch.randn_like(nr_ref)
    g_ni = torch.randn_like(ni_ref)
    if coarse:
        odd = (torch.arange(width, device=device) % 2 == 1).view(1, 1, width, 1)
        even = ~odd
        ref_adjoints = (g_pr * odd, g_pi * odd, g_nr * even, g_ni * even)
    else:
        ref_adjoints = (g_pr, g_pi, g_nr, g_ni)

    ref_grads = torch.autograd.grad(
        (pr_ref, pi_ref, nr_ref, ni_ref),
        (dr, di, gr, gi, packed_src),
        ref_adjoints,
    )

    if force_single_chunk:
        monkeypatch.setattr(rh, "_choose_y_chunks", lambda *_a, **_k: 1)

    dst_packed = torch.empty_like(packed_src)
    g_dr, g_di, g_gr, g_gi = rh.r11_cuda_hscan_backward_into_op(
        dr.detach(),
        di.detach(),
        gr.detach(),
        gi.detach(),
        sr.detach(),
        si.detach(),
        st_cu[0],
        st_cu[1],
        st_cu[2],
        st_cu[3],
        g_pr,
        g_pi,
        g_nr,
        g_ni,
        dst_packed,
        coarse,
    )
    for cu_g, ref_g in zip((g_dr, g_di, g_gr, g_gi, dst_packed), ref_grads, strict=True):
        assert _rel_l2(cu_g, ref_g) < 0.01
