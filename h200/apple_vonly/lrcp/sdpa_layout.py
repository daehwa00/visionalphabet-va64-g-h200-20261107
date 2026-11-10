"""Experimental, opt-in SDPA output layout adapter for Hugging Face GPT-NeoX.

The stock adapter requests contiguous BSHD output. That request is already a
no-op when the SDPA backend writes BSHD storage, including the efficient backend
profiled on this RTX 3080 Ti. No copy or memory-traffic reduction was observed
there, so this experiment is NOT integrated into the production model wrappers.
If a backend instead returns contiguous BHSD, APPLE can consume its strided view
and fuse the layout conversion into its output write. This adapter changes only
that final layout request, not the SDPA operation.
The model's ``_attn_implementation`` remains ``sdpa``, preserving mask creation.
"""
from __future__ import annotations

from types import MethodType

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    apply_rotary_pos_emb,
)


__all__ = ["sdpa_attention_bshd_view", "set_neox_sdpa_layout"]


def sdpa_attention_bshd_view(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    position_bias=None,
    **kwargs,
):
    """HF-compatible NeoX SDPA, returning BSHD without a layout materialization.

    Generic attention features and other accelerators retain the installed HF
    adapter. Its ordinary kwargs (e.g. ``position_ids``) remain accepted. The
    caller must accept strided output; the final dense projection or APPLE
    fused output provides contiguous merged heads where needed.
    """
    if (
        query.device.type not in ("cpu", "cuda")
        or getattr(module, "num_key_value_groups", 1) > 1
        or position_bias is not None
        or kwargs.get("output_attentions", False)
    ):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal,
            position_bias=position_bias, **kwargs,
        )

    # Keep the installed HF NeoX adapter's causal dispatch, including a single
    # cached query attending its entire prefix and prefill into an empty static
    # cache. A supplied mask already carries cached multi-query alignment.
    q_length = query.shape[2]
    kv_length = key.shape[2]
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = q_length > 1 and attention_mask is None and is_causal
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()
    if is_causal and attention_mask is None and q_length > 1 and kv_length > q_length:
        key = key[:, :, :q_length, :]
        value = value[:, :, :q_length, :]

    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=is_causal,
    )
    return output.transpose(1, 2), None


def _neox_sdpa_view_forward(
    self,
    hidden_states,
    attention_mask,
    layer_past=None,
    position_embeddings=None,
    **kwargs,
):
    # This model-local baseline adapter applies the same layout policy used by
    # APPLE while preserving non-SDPA configurations and the module's weights.
    if self.config._attn_implementation != "sdpa":
        return self._sdpa_layout_original_forward(
            hidden_states, attention_mask, layer_past=layer_past,
            position_embeddings=position_embeddings, **kwargs,
        )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)
    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query, key, value = qkv.chunk(3, dim=-1)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if layer_past is not None:
        key, value = layer_past.update(key, value, self.layer_idx)
    output, weights = sdpa_attention_bshd_view(
        self, query, key, value, attention_mask, scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout, **kwargs,
    )
    output = output.reshape(*input_shape, -1).contiguous()
    return self.dense(output), weights


def set_neox_sdpa_layout(model, enabled=True):
    """Enable/restore the no-copy adapter on this model's baseline NeoX layers.

    Only exact ``GPTNeoXAttention`` instances are changed. APPLE subclasses and
    existing instance-level custom forwards are left under their owner's
    control. No global attention registry or checkpoint entries are modified.
    Returns the number of modules changed; repeating the same setting is a no-op.
    """
    changed = 0
    for module in model.modules():
        if type(module) is not GPTNeoXAttention:
            continue
        active = "_sdpa_layout_original_forward" in module.__dict__
        if enabled and not active and "forward" not in module.__dict__:
            module._sdpa_layout_original_forward = module.forward
            module.forward = MethodType(_neox_sdpa_view_forward, module)
            changed += 1
        elif not enabled and active:
            # Enabling is restricted to modules with a class-defined forward;
            # removing our override restores exactly that original lookup.
            del module.forward
            del module._sdpa_layout_original_forward
            changed += 1
    return changed
