"""Compatibility checks for the historical Wave 30C probe entry points."""

import pytest
import torch

from lnet.model_w29c import model_w29c_enabled
from lnet.model_w30c import swiglu_tile


def test_retired_policy_does_not_enable_nr4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LNET_MODEL_W29C", "off")
    monkeypatch.setenv("LNET_MODEL_W30C", "combined_v2")
    assert not model_w29c_enabled(torch.device("cuda"), "nr4")


@pytest.mark.parametrize(("hidden", "cols"), [(96, 32), (144, 16), (192, 64), (384, 64)])
def test_divisor_tiles(hidden: int, cols: int) -> None:
    rows, actual_cols = swiglu_tile(hidden)
    assert actual_cols == cols
    assert hidden % actual_cols == 0
    assert rows * cols == 4096
