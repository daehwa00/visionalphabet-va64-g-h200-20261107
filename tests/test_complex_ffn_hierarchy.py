# pyright: reportMissingImports=false
from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from lnet.complex_scan import (
    AugmentedComplexTransition,
    ComplexInteractionTransition,
    ComplexModulatedTransition,
    ComplexResidualFFN,
    S2DPostCFFNCarryMainTransition,
    S2DPostFusionCFFNTransition,
)
from lnet.pac_complex_ffn import ComplexFFN
from lnet.pac_path_cffn import FactorizedQuadrantPathModeCFFNCombiner


def test_project_cffn_modules_share_the_common_base() -> None:
    module_types = (
        ComplexModulatedTransition,
        ComplexInteractionTransition,
        AugmentedComplexTransition,
        S2DPostCFFNCarryMainTransition,
        S2DPostFusionCFFNTransition,
        ComplexResidualFFN,
        FactorizedQuadrantPathModeCFFNCombiner,
    )

    assert all(issubclass(module_type, ComplexFFN) for module_type in module_types)


def test_complex_residual_ffn_keeps_its_checkpoint_contract() -> None:
    module = ComplexResidualFFN(4, 7)

    assert tuple(module.state_dict()) == (
        "layer_scale",
        "input.weight_real",
        "input.weight_imag",
        "input.conjugate_real",
        "input.conjugate_imag",
        "input.bias_real",
        "input.bias_imag",
        "output.weight_real",
        "output.weight_imag",
        "output.conjugate_real",
        "output.conjugate_imag",
        "output.bias_real",
        "output.bias_imag",
    )


def test_complex_residual_ffn_keeps_its_residual_equation() -> None:
    torch.manual_seed(501)
    module = ComplexResidualFFN(4, 7)
    module.layer_scale.data.fill_(0.125)
    real = torch.randn(2, 3, 4)
    imag = torch.randn_like(real)

    hidden_real, hidden_imag = module.input(real, imag)
    expected_real, expected_imag = module.output(
        torch.nn.functional.silu(hidden_real),
        torch.nn.functional.silu(hidden_imag),
    )
    expected_real = real + module.layer_scale * expected_real
    expected_imag = imag + module.layer_scale * expected_imag

    actual_real, actual_imag = module(real, imag)

    torch.testing.assert_close(actual_real, expected_real)
    torch.testing.assert_close(actual_imag, expected_imag)


def test_optional_activation_scales_strictly_round_trip() -> None:
    cases = (
        (
            FactorizedQuadrantPathModeCFFNCombiner(4, 8, 8),
            FactorizedQuadrantPathModeCFFNCombiner(4, 8, 8),
            ("mode_activation_scale", "path_activation_scale"),
        ),
        (
            S2DPostCFFNCarryMainTransition(4, 8),
            S2DPostCFFNCarryMainTransition(4, 8),
            ("ffn_activation_scale",),
        ),
        (
            S2DPostFusionCFFNTransition(4, 8),
            S2DPostFusionCFFNTransition(4, 8),
            ("ffn_activation_scale", "post_ffn_activation_scale"),
        ),
    )

    for source, restored, names in cases:
        for index, name in enumerate(names, start=1):
            setattr(source, name, torch.tensor(1.0 + 0.25 * index))

        restored.load_state_dict(source.state_dict(), strict=True)

        for name in names:
            torch.testing.assert_close(getattr(restored, name), getattr(source, name))


def test_optional_activation_scales_follow_the_target_dtype_on_load() -> None:
    source = S2DPostFusionCFFNTransition(4, 8)
    source.ffn_activation_scale = torch.tensor(1.25)
    source.post_ffn_activation_scale = torch.tensor(1.5)
    restored = S2DPostFusionCFFNTransition(4, 8).bfloat16()

    restored.load_state_dict(source.state_dict(), strict=True)

    assert restored.ffn_activation_scale is not None
    assert restored.post_ffn_activation_scale is not None
    assert restored.ffn_activation_scale.dtype is torch.bfloat16
    assert restored.post_ffn_activation_scale.dtype is torch.bfloat16


class _OptionalBufferProbe(ComplexFFN):
    def __init__(self, *, scoped: bool) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.scale: Tensor | None
        if scoped:
            self.register_optional_persistent_buffer("scale")
        else:
            self.register_buffer("scale", None, persistent=True)


def test_optional_buffer_loading_is_limited_to_explicitly_registered_names() -> None:
    checkpoint = {"anchor": torch.tensor(2.0), "scale": torch.tensor(1.25)}
    scoped = _OptionalBufferProbe(scoped=True)
    scoped.load_state_dict(checkpoint, strict=True)

    torch.testing.assert_close(scoped.scale, checkpoint["scale"])
    assert all("optional_persistent" not in key for key in scoped.state_dict())

    ordinary = _OptionalBufferProbe(scoped=False)
    with pytest.raises(RuntimeError, match=r"Unexpected key\(s\) in state_dict"):
        ordinary.load_state_dict(checkpoint, strict=True)
