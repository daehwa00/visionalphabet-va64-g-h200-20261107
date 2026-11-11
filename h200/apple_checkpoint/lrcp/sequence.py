"""Sequence-transformer wiring for the canonical LRCP branch.

This module is intentionally small and task-agnostic.  It provides a standard
multi-head self-attention block where the ordinary attention output ``AV`` can
optionally be passed through :class:`lrcp.headwise.LRCP` together with the value
states ``V``.  Task scripts own data loading, recipes, and metrics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .headwise import LRCP


__all__ = [
    "SequenceSelfAttention",
    "SequenceTransformerBlock",
    "collect_lrcp_stats",
    "is_lrcp_parameter_name",
    "iter_lrcp_modules",
]


def is_lrcp_parameter_name(name: str) -> bool:
    """Return True for LRCP parameters in sequence models."""

    return name.startswith("lrcp.") or ".lrcp." in name


def iter_lrcp_modules(model: nn.Module) -> list[LRCP]:
    """Collect canonical LRCP modules attached under a model."""

    return [module for module in model.modules() if isinstance(module, LRCP)]


def collect_lrcp_stats(model: nn.Module) -> dict[str, float]:
    """Average last LRCP telemetry from all active modules."""

    modules = iter_lrcp_modules(model)
    out: dict[str, float] = {"lrcp_modules": float(len(modules))}
    stats = [module.last_stats for module in modules if module.last_stats]
    for key in ("branch_over_av", "raw_branch_over_av", "residual_over_av", "gamma_mean"):
        vals = [float(item[key]) for item in stats if key in item]
        if vals:
            out[f"lrcp_{key}"] = sum(vals) / len(vals)
    return out


class SequenceSelfAttention(nn.Module):
    """Multi-head self-attention with an optional canonical LRCP branch.

    Inputs use the common ``[B, N, C]`` layout.  Internally LRCP sees
    ``AV`` and ``V`` as ``[B, H, N, D]``.  ``key_padding_mask`` follows PyTorch
    convention: True entries are ignored as keys.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        causal: bool = False,
        lrcp_rank: int | None = None,
        lrcp_gamma: float = 1.0,
        down_init_std: float = 0.02,
        up_init_std: float = 0.0,
        residual_detach: bool = False,
        record_stats: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.causal = bool(causal)
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(dropout)
        self.lrcp: LRCP | None = None
        if lrcp_rank is not None and int(lrcp_rank) > 0:
            self.lrcp = LRCP(
                self.num_heads,
                self.head_dim,
                int(lrcp_rank),
                float(lrcp_gamma),
                down_init_std=float(down_init_std),
                up_init_std=float(up_init_std),
                residual_detach=bool(residual_detach),
            )
            self.lrcp.stats_enabled = bool(record_stats)

    def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SequenceSelfAttention expects [B,N,C], got {tuple(x.shape)}")
        bsz, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, num_tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if key_padding_mask is not None:
            if key_padding_mask.shape != (bsz, num_tokens):
                raise ValueError(
                    "key_padding_mask must be shaped [B,N], "
                    f"got {tuple(key_padding_mask.shape)} for input {tuple(x.shape)}"
                )
            attn = attn.masked_fill(key_padding_mask[:, None, None, :].to(torch.bool), -torch.inf)
        if self.causal:
            causal_mask = torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool).triu(1)
            attn = attn.masked_fill(causal_mask, -torch.inf)
        attn = attn.softmax(dim=-1)
        # Rows that only saw masked keys can become NaN; this only occurs for
        # fully padded sequences and their losses are masked by task scripts.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)
        av = attn @ v
        if self.lrcp is not None:
            av = self.lrcp(av, v)
        y = av.transpose(1, 2).reshape(bsz, num_tokens, channels)
        return self.proj_drop(self.proj(y))


class SequenceTransformerBlock(nn.Module):
    """Pre-norm Transformer encoder/decoder-style block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal: bool = False,
        lrcp_rank: int | None = None,
        lrcp_gamma: float = 1.0,
        down_init_std: float = 0.02,
        up_init_std: float = 0.0,
        residual_detach: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SequenceSelfAttention(
            dim,
            num_heads,
            dropout=dropout,
            causal=causal,
            lrcp_rank=lrcp_rank,
            lrcp_gamma=lrcp_gamma,
            down_init_std=down_init_std,
            up_init_std=up_init_std,
            residual_detach=residual_detach,
        )
        hidden = int(math.ceil(dim * mlp_ratio))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x
