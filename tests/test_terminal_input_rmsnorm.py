"""Tests for terminal input RMSNorm on VA-Ablation-10-dw-terminal-rmsnorm."""

from __future__ import annotations

import run_alphabet_ablations as candidate
import torch
from alphabet_model_builder import model_config


def test_terminal_input_rmsnorm_build_and_scale_invariance() -> None:
    cfg = model_config()
    torch.manual_seed(501)
    model = candidate._build(candidate.VARIANT, cfg)
    state = model.state_dict()
    assert "backbone.terminal.pole_input_projection.input_norm.weight" in state
    reader = model.backbone.terminal.pole_input_projection
    assert not reader.match_input_rms
    assert sum(p.numel() for p in model.parameters()) == 2_389_796
    norm = model.backbone.same_resolution_blocks["7"].reader.input_norm
    assert reader.input_norm is not None
    assert norm is not None
    assert reader.input_norm is not norm
    torch.testing.assert_close(reader.input_norm.weight, norm.weight, rtol=0, atol=0)
    assert reader.input_norm.epsilon == norm.epsilon
    x = torch.randn(2, 7, 7, 256, requires_grad=True)
    y = reader(x)
    scaled = reader(x * 3)
    torch.testing.assert_close(y, scaled, rtol=2e-5, atol=2e-5)
    y.square().mean().backward()
    assert reader.input_norm.weight.grad is not None
    assert torch.isfinite(reader.input_norm.weight.grad).all()
    assert reader.input_norm.weight.grad.abs().sum() > 0
