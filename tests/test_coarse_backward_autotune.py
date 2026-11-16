"""The coarse adjoint must search configurations without changing forward policy."""
import pytest
import torch
from test_coarse_path_fusion import (
    test_coarse_path_fusion_forward_and_all_gradients as check_coarse_gradients,
)

from lnet import pac_product_scan_ops as ops
from lnet.pac_kernel_launch_config import registered_candidates

CONFIGURATIONS = [(modes, warps) for modes in (4, 8, 16) for warps in (4, 8)]


def test_coarse_backward_search_preserves_forward_default() -> None:
    backward = registered_candidates(ops.COARSE_PATH_BACKWARD_LAUNCH_NAME)
    assert {(c.blocks["BLOCK_MODES"], c.num_warps) for c in backward} == set(CONFIGURATIONS)
    assert all(c.blocks["BLOCK_LINES"] == 1 for c in backward)
    forward = registered_candidates(ops.COARSE_PATH_FORWARD_LAUNCH_NAME)
    assert len(forward) == 1
    assert forward[0].blocks == {"BLOCK_LINES": 1, "BLOCK_MODES": 8}
    assert forward[0].num_warps == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("modes", "warps"), CONFIGURATIONS)
@pytest.mark.parametrize("gain", ["global", "pointwise"])
@pytest.mark.parametrize("partials", ["0", "1"])
def test_every_coarse_backward_configuration(
    monkeypatch: pytest.MonkeyPatch, modes: int, warps: int, gain: str, partials: str,
) -> None:
    prefix = "LNET_LAUNCH_" + ops.COARSE_PATH_BACKWARD_LAUNCH_NAME.upper()
    monkeypatch.setenv("LNET_DISABLE_LAUNCH_AUTOTUNE", "1")
    monkeypatch.setenv("VA_REAL_MEMORY_SCAN_BATCH_PARTIALS", partials)
    monkeypatch.setenv(prefix + "_BLOCK_LINES", "1")
    monkeypatch.setenv(prefix + "_BLOCK_MODES", str(modes))
    monkeypatch.setenv(prefix + "_WARPS", str(warps))
    check_coarse_gradients("cuda", gain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coarse_backward_does_not_explicitly_pin_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = []
    original = ops.autotuned

    def capture(*args: object, **kwargs: object):
        if args[1] == ops.COARSE_PATH_BACKWARD_LAUNCH_NAME:
            observed.append(kwargs.get("geometry"))
        return original(*args, **kwargs)

    monkeypatch.setenv("LNET_DISABLE_LAUNCH_AUTOTUNE", "1")
    monkeypatch.setattr(ops, "autotuned", capture)
    check_coarse_gradients("cuda", "pointwise")
    assert observed
    assert all(value is None for value in observed)
