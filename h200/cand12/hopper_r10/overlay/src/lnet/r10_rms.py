"""FP32 partial reduction of the 56² ChannelRMSNorm scale gradient."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch._inductor import config
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch.fx import Graph
from torch.library import triton_op, wrap_triton

from .r9_rms import _rms_operands

HEIGHT = 56
CHANNELS = 64
ROW_TILE = 512
CHANNEL_TILE = 16
VALUE_STRIDE = (200704, 1, 3584, 56)


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
            operands = _rms_operands(node, shape=(HEIGHT, HEIGHT, CHANNELS),
                                    value_stride=VALUE_STRIDE, row_tile=ROW_TILE,
                                    allow_matching_grad_stride=True)
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
