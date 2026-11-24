"""Run actual Triton launch bindings through PyTorch TTIR on CPU."""
# ruff: noqa: FBT003

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton
from torch._higher_order_ops.triton_kernel_wrap import generate_ttir
from triton.backends.compiler import GPUTarget

from lnet import (
    fused_stem_ops as stem,
)
from lnet import (
    model_w28c as w28c,
)
from lnet import (
    model_w29c as w29c,
)
from lnet import (
    product_scan_normalization as norm,
)
from lnet import (
    product_scan_ops as product,
)
from lnet import (
    terminal_imaginary_correlation as terminal,
)

bidirectional = importlib.import_module("lnet.triton_bidirectional_product_scan")


def test_real_ttir_launch_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    target = GPUTarget("cuda", 90, 32)
    monkeypatch.setattr(triton.runtime.driver, "_active",
                        SimpleNamespace(get_current_target=lambda: target))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _: SimpleNamespace(multi_processor_count=132))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setenv("LNET_GENERAL_SCANS", "off")
    sites = set()

    class Launch:
        def __init__(self, kernel: triton.runtime.JITFunction) -> None:
            self.kernel = kernel

        def __getitem__(self, _grid: object) -> Callable[..., None]:
            return self.launch

        def launch(self, *args: object, **kwargs: object) -> None:
            frame = sys._getframe(1)
            named = dict(zip(self.kernel.arg_names[:len(args)], args, strict=True))
            module, _ = generate_ttir(self.kernel, {**named, **kwargs}, {})
            assert "tt.func" in str(module)
            sites.add((Path(frame.f_code.co_filename).name, frame.f_lineno, self.kernel.__name__))

    for mod in (w28c, w29c, terminal, norm, bidirectional, product):
        monkeypatch.setattr(mod, "wrap_triton", Launch)
    for name in ("stem_conv_tensor_stages", "stem_conv_tensor_reduce"):
        monkeypatch.setattr(stem, name, Launch(getattr(stem, name)))
    probe = importlib.import_module("test_product_scan_ops")
    monkeypatch.setattr(probe, "_path_channel_probe", Launch(probe._path_channel_probe))

    x = torch.empty(4, 64, dtype=torch.bfloat16)
    grad = x[:, :32].contiguous()
    w28c.packed_swiglu._init_fn(x)
    w28c._packed_swiglu_backward._init_fn(x, grad)
    w29c._launch_swiglu(x, x, grad, backward=False)
    w29c.channel_rmsnorm._init_fn(x, x[0], 1e-5, True)
    w29c._channel_rmsnorm_backward._init_fn(x, x[0], torch.ones(4), x, need_dx=True, need_dw=True)

    state = torch.empty(2, 8, 8, 32, dtype=torch.bfloat16)
    pole = (torch.empty(1, 1, 1, 32),) * 4
    terminal._vertical_scan(state.float(), state.float(), *pole[:2], False)
    terminal._nr4_launch((state,) * 4, pole, state, state, state, state,
                         normalize=False, backward=False, warp_tile=False)
    norm.static_global_inverse_gain._init_fn(*pole, *pole, 8, 8, 1e-5)
    norm.static_variance_tables._init_fn(*pole, *pole, 8, 8)
    bidirectional._forward_op._init_fn(*pole, state, state)
    bidirectional._forward_planes_op._init_fn(*pole, state, state)
    bidirectional._launch_backward(pole, (state, state), None, (state,) * 4, launch_name="ttir")

    table = torch.empty(2, 8, 32)
    empty = torch.empty(0)
    inputs = (*pole, *([state] * 4), table, table, empty)
    product._launch_product_scan4_forward(*inputs, 1e-5, 0, emit_coarse=False, launch_name="ttir")
    product._launch_product_scan4_backward(
        *inputs, empty, empty, torch.empty(2, 128), 1e-5, 0, emit_coarse=False, launch_name="ttir",
    )

    image = torch.empty(2, 3, 8, 8, dtype=torch.bfloat16).contiguous(
        memory_format=torch.channels_last,
    )
    weights = torch.empty(27, 32, dtype=torch.bfloat16)
    affine = torch.empty(32)
    y = torch.empty(2, 4, 4, 32, dtype=torch.bfloat16)
    stats = torch.empty(2, 4, 4, 2)
    stem._tensor_core_conv_backward(
        image, weights, affine, affine, affine, y, stats, 1e-5, weights, affine, affine, affine,
    )
    probe._path_channel_probe[(1,)](state, state, state, 64, 4)
    assert len(sites) == 18, sorted(sites)
