from __future__ import annotations

# pyright: reportPrivateUsage=false
import torch

from lnet.pac_gated_post_fusion import GatedStrictComplexMagnitudePostFusion
from scripts import a2d_r2k3_runtime as runtime
from scripts import run_a2d_r2k3_same_resolution_all_clmaggate384_imagenet100 as runner


def test_clmaggate384_is_exactly_parameter_matched_and_phase_equivariant() -> None:
    model = runner._build(runner.VARIANT, runtime.model_config())
    runner._assert_model(model)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_772_516
    assert all(
        type(owner.post_fusion) is GatedStrictComplexMagnitudePostFusion
        and owner.post_fusion.hidden_modes == 384
        for owner in runner._postfusions(model)
    )

    module = runner._postfusions(model)[0].post_fusion
    real = torch.randn(2, 3, 4, 128)
    imag = torch.randn_like(real)
    angle = torch.tensor(0.67)
    cosine, sine = angle.cos(), angle.sin()
    rotated = cosine * real - sine * imag, sine * real + cosine * imag
    output = module(real, imag)
    rotated_output = module(*rotated)
    expected = (
        cosine * output[0] - sine * output[1],
        sine * output[0] + cosine * output[1],
    )
    torch.testing.assert_close(rotated_output[0], expected[0], rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(rotated_output[1], expected[1], rtol=2.0e-5, atol=2.0e-5)

