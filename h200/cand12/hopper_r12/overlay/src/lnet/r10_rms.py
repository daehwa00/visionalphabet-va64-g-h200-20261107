"""Opt-in 56² ChannelRMSNorm scale-gradient reductions for R10."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch.fx import Graph
from torch.library import triton_op, wrap_triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout

from .r9_rms import _rms_operands

HEIGHT = 56
CHANNELS = 64
ROW_TILE = 512
CHANNEL_TILE = 16
VALUE_STRIDE = (200704, 1, 3584, 56)

_HEIGHT = HEIGHT
_CHANNELS = CHANNELS
_VALUE_STRIDE = VALUE_STRIDE
_INVERSE_STRIDE = (3136, 1, 56, 56)


@gluon.jit
def r10_rms_y_partial(DY, X, INV, DW):
    # Load the input along Y and the upstream gradient along C.
    # Convert the gradient to the input layout before the reduction along Y.
    vl: gl.constexpr = BlockedLayout([1, 1], [8, 4], [4, 1], [0, 1])
    dl: gl.constexpr = BlockedLayout([1, 1], [4, 8], [4, 1], [1, 0])
    column = gl.program_id(0)
    batch, x = column // 56, column % 56
    channel_base = gl.program_id(1) * 8
    y = gl.arange(0, 64, layout=gl.constexpr(SliceLayout(1, vl)))[:, None]
    c = channel_base + gl.arange(0, 8, layout=gl.constexpr(SliceLayout(0, vl)))
    dy_y = gl.arange(0, 64, layout=gl.constexpr(SliceLayout(1, dl)))[:, None]
    dy_c = channel_base + gl.arange(0, 8, layout=gl.constexpr(SliceLayout(0, dl)))
    value = gl.load(X + batch * 200704 + x * 3584 + c[None, :] * 56 + y, y < 56, 0)
    gradient = gl.load(DY + ((batch * 56 + dy_y) * 56 + x) * 64 + dy_c[None, :], dy_y < 56, 0)
    gradient = gl.convert_layout(gradient, vl)
    inverse = gl.load(INV + batch * 3136 + x * 56 + y, y < 56, 0)
    product = gradient.to(gl.float32) * (value.to(gl.float32) * inverse)
    gl.store(DW + column * 64 + c, gl.sum(product, 0))


def rms_y_shape_supported(batch: int) -> bool:
    return isinstance(batch, int) and 0 < batch * _HEIGHT * _HEIGHT * _CHANNELS < 2**31


@triton_op("lnet::r10_rms_scale_gradient", mutates_args={})
def _rms_scale_gradient(gradient: Tensor, values: Tensor, inverse: Tensor) -> Tensor:
    columns = values.shape[0] * _HEIGHT
    partial = torch.empty((columns, _CHANNELS), dtype=torch.float32, device=values.device)
    wrap_triton(r10_rms_y_partial)[(columns, _CHANNELS // 8)](
        gradient, values, inverse, partial, num_warps=4,
    )
    return partial.sum(0).view(1, 1, 1, _CHANNELS)


@dataclass(frozen=True)
class RMSYScaleGradientPass(CustomGraphPass):
    previous: object

    def __call__(self, graph: Graph) -> None:
        if callable(self.previous):
            self.previous(graph)
        if config.emulate_precision_casts:
            return
        for node in list(graph.nodes):
            operands = _rms_operands(
                node, shape=(_HEIGHT, _HEIGHT, _CHANNELS), value_stride=_VALUE_STRIDE,
            )
            if operands is None:
                continue
            _, source, inverse = operands
            if (
                not rms_y_shape_supported(source.meta["val"].shape[0])
                or inverse.meta["val"].stride() != _INVERSE_STRIDE
            ):
                continue
            with graph.inserting_before(node):
                replacement = graph.call_function(
                    torch.ops.lnet.r10_rms_scale_gradient.default, operands
                )
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)

    def uuid(self):
        if self.previous is None:
            return "r10_rms_scale_gradient"
        if isinstance(self.previous, CustomGraphPass):
            key = self.previous.uuid()
            if key is not None:
                return ("r10_rms_scale_gradient", key)
        return None


def rms56_partial_rows(batch):
    if not isinstance(batch, int) or not 0 < batch * HEIGHT**2 * CHANNELS < 2**31:
        return None
    rows = triton.cdiv(batch * HEIGHT**2, ROW_TILE)
    return rows if rows <= 65535 else None


@triton.jit
def r10_rms56_partial(
    DY, X, INV, PARTIAL, N, SD: tl.constexpr, SI: tl.constexpr,
    H: tl.constexpr, C: tl.constexpr, ROWS: tl.constexpr, CHANNEL_TILE: tl.constexpr,
):
    # Enumerate y in the inner row coordinate, matching the source's layout.
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)[:, None]
    channel = tl.program_id(0) * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)[None, :]
    batch, y, x = row // (H * H), row % H, (row // H) % H
    valid = row < N * H * H
    index = batch * SD[0] + y * SD[1] + x * SD[2] + channel * SD[3]
    gradient = tl.load(DY + index, valid, 0).to(tl.float32)
    value = tl.load(X + batch * (H * H * C) + y + x * (H * C) + channel * H,
                    valid, 0).to(tl.float32)
    inverse = tl.load(INV + batch * SI[0] + y * SI[1] + x * SI[2], valid, 0)
    scale = tl.sum(gradient * (value * inverse), 0)
    tl.store(PARTIAL + tl.program_id(1) * C + tl.reshape(channel, (CHANNEL_TILE,)), scale)


@triton_op("lnet::r10_rms56_scale_gradient", mutates_args={})
def _rms56_scale_gradient(gradient: Tensor, values: Tensor, inverse: Tensor) -> Tensor:
    rows = rms56_partial_rows(values.shape[0])
    partial = torch.empty((rows, CHANNELS), device=values.device, dtype=torch.float32)
    wrap_triton(r10_rms56_partial)[(CHANNELS // CHANNEL_TILE, rows)](
        gradient, values, inverse, partial, values.shape[0],
        gradient.stride(), inverse.stride(),
        H=HEIGHT, C=CHANNELS, ROWS=ROW_TILE, CHANNEL_TILE=CHANNEL_TILE, num_warps=4,
    )
    return partial.sum(0).view(1, 1, 1, CHANNELS)


@dataclass(frozen=True)
class RMS56ScaleGradientPass(CustomGraphPass):
    previous: object

    def __call__(self, graph: Graph) -> None:
        if callable(self.previous):
            self.previous(graph)
        if config.emulate_precision_casts:
            return
        for node in list(graph.nodes):
            operands = _rms_operands(
                node,
                shape=(HEIGHT, HEIGHT, CHANNELS),
                value_stride=VALUE_STRIDE,
                row_tile=ROW_TILE,
                allow_matching_grad_stride=True,
            )
            if operands is None:
                continue
            with graph.inserting_before(node):
                replacement = graph.call_function(
                    torch.ops.lnet.r10_rms56_scale_gradient.default, operands
                )
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            graph.erase_node(node)

    def uuid(self):
        if self.previous is None:
            return "r10_rms56_scale_gradient"
        if isinstance(self.previous, CustomGraphPass):
            key = self.previous.uuid()
            if key is not None:
                return ("r10_rms56_scale_gradient", key)
        return None
