"""Pythia/GPT-NeoX adapter wiring for LRCP.

This module owns model-specific attention wrapping.  The LRCP branch math stays
in :mod:`lrcp.headwise`.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

try:  # Optional dependency: parse_layers and tests should work without HF installed.
    from transformers.models.gpt_neox.modeling_gpt_neox import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal env import tests
    ALL_ATTENTION_FUNCTIONS = None
    apply_rotary_pos_emb = None
    eager_attention_forward = None

from .headwise import LRCP


__all__ = [
    "GPTNeoXLRCPAttention",
    "attach_lrcp",
    "collect_lrcp_stats",
    "lrcp_modules",
    "parse_layers",
    "set_lrcp_fused_inference",
    "set_lrcp_fused_training",
    "set_lrcp_stats_enabled",
]


def _require_transformers() -> None:
    if apply_rotary_pos_emb is None or eager_attention_forward is None or ALL_ATTENTION_FUNCTIONS is None:
        raise ModuleNotFoundError("lrcp.pythia requires the optional `transformers` package for GPT-NeoX wiring.")


class GPTNeoXLRCPAttention(nn.Module):
    def __init__(
        self,
        base_attention: nn.Module,
        *,
        rank: int,
        gamma: float,
        down_init_std: float,
        up_init_std: float,
    ):
        super().__init__()
        _require_transformers()
        self.config = base_attention.config
        self.head_size = base_attention.head_size
        self.attention_dropout = base_attention.attention_dropout
        self.rotary_ndims = base_attention.rotary_ndims
        self.scaling = base_attention.scaling
        self.is_causal = True
        self.layer_idx = base_attention.layer_idx
        self.query_key_value = base_attention.query_key_value
        self.dense = base_attention.dense
        self.lrcp = LRCP(
            self.config.num_attention_heads,
            self.head_size,
            rank,
            gamma,
            down_init_std,
            up_init_std,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        layer_past=None,
        output_attentions: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)
        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)
        value_current = value_states.transpose(1, 2)
        if position_embeddings is None:
            raise RuntimeError("GPTNeoXLRCPAttention requires position_embeddings.")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if layer_past is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_ndims, "cache_position": cache_position}
            key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs)  # pragma: no cover
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            head_mask=head_mask,
            **kwargs,
        )
        # HF attention returns BSHD. Keep that layout through LRCP so the
        # branch does not pay for two full-size contiguous transpose copies.
        attn_output = self.lrcp.forward_bshd(attn_output, value_current)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    if spec == "late-half":
        return list(range(n_layers // 2, n_layers))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx += n_layers
        out.append(idx)
    return sorted(set(out))


def attach_lrcp(
    model: nn.Module,
    *,
    layers: str,
    rank: int,
    gamma: float,
    down_init_std: float,
    up_init_std: float,
) -> int:
    blocks = model.gpt_neox.layers
    selected = parse_layers(layers, len(blocks))
    for idx in selected:
        attn = blocks[idx].attention
        if isinstance(attn, GPTNeoXLRCPAttention):
            continue
        wrapped = GPTNeoXLRCPAttention(
            attn,
            rank=rank,
            gamma=gamma,
            down_init_std=down_init_std,
            up_init_std=up_init_std,
        )
        wrapped.to(device=attn.query_key_value.weight.device, dtype=attn.query_key_value.weight.dtype)
        blocks[idx].attention = wrapped
    return len(selected)


def lrcp_modules(model: nn.Module) -> list[LRCP]:
    return [m.lrcp for m in model.modules() if isinstance(m, GPTNeoXLRCPAttention)]


def set_lrcp_stats_enabled(model: nn.Module, enabled: bool) -> None:
    for module in lrcp_modules(model):
        module.stats_enabled = enabled
        if enabled:
            module.last_stats = {}


def set_lrcp_fused_inference(model: nn.Module, enabled: bool) -> int:
    """Enable the optional BF16 CUDA fused path for inference-only forwards."""

    modules = lrcp_modules(model)
    for module in modules:
        module.use_fused_inference = bool(enabled)
    return len(modules)


def set_lrcp_fused_training(model: nn.Module, enabled: bool) -> int:
    """Enable the BF16 CUDA fused-autograd training path."""

    modules = lrcp_modules(model)
    for module in modules:
        module.use_fused_training = bool(enabled)
    return len(modules)


def collect_lrcp_stats(model: nn.Module) -> dict[str, float]:
    modules = lrcp_modules(model)
    out: dict[str, float] = {"lrcp_modules": float(len(modules))}
    stats = [module.last_stats for module in modules if module.last_stats]
    if stats:
        for key in ["branch_over_av", "raw_branch_over_av", "residual_over_av", "gamma_mean"]:
            vals = [float(item[key]) for item in stats if key in item]
            if vals:
                out[f"lrcp_{key}"] = sum(vals) / len(vals)
    return out
