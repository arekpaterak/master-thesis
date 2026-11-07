"""Size-aware scaling of attention logits.

Multiplying the logits by a factor growing with the number of attended nodes ``n`` keeps
the softmax from flattening as ``n`` grows:

- ``ln n``: Chiang & Cholak (2022), "Overcoming a Theoretical Limitation of Self-Attention".
- ``log_{n_ref} n``: Xiao et al. (2024), "Improving Generalization of Neural Vehicle Routing
  Problem Solvers Through the Lens of Model Architecture" (Entropy-based Scaling Factor).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import nn


class AttentionScalingConfig(BaseModel):
    """Scale attention logits by ``ln n`` or, with a reference size, by ``log_{n_ref} n``."""

    reference_size: int | None = Field(
        default=None,
        ge=2,
        title="Reference size",
        description=(
            "Number of attended nodes at which the factor is 1 (Xiao et al., 2024). Counts "
            "every node, e.g. 101 for CVRP100 (depot included). None scales by ln n "
            "(Chiang & Cholak, 2022)."
        ),
    )

    def factor(self, num_nodes: int) -> float:
        if self.reference_size is None:
            return math.log(num_nodes)
        return math.log(num_nodes) / math.log(self.reference_size)


def scaled_multihead_attention(
    mha: nn.MultiheadAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    logit_scale: float,
    attend_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``mha`` with its logits multiplied by ``logit_scale``, reusing its parameters.

    Args:
        mha: A batch-first ``nn.MultiheadAttention`` with equal q/k/v dims.
        q: (B, L_q, E)
        k, v: (B, L_k, E)
        logit_scale: Multiplier on top of the usual 1/sqrt(head_dim).
        attend_mask: Optional (B, L_k) bool, True = attend.

    Returns:
        (B, L_q, E)
    """
    assert mha.batch_first and mha._qkv_same_embed_dim
    assert mha.bias_k is None and not mha.add_zero_attn

    B, L_q, E = q.shape
    L_k = k.shape[-2]
    H = mha.num_heads
    head_dim = E // H

    w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
    b_q, b_k, b_v = (None,) * 3 if mha.in_proj_bias is None else mha.in_proj_bias.chunk(3)
    q_h = F.linear(q, w_q, b_q).view(B, L_q, H, head_dim).transpose(1, 2)
    k_h = F.linear(k, w_k, b_k).view(B, L_k, H, head_dim).transpose(1, 2)
    v_h = F.linear(v, w_v, b_v).view(B, L_k, H, head_dim).transpose(1, 2)

    attn_mask = None if attend_mask is None else attend_mask[:, None, None, :]
    out = F.scaled_dot_product_attention(
        q_h,
        k_h,
        v_h,
        attn_mask=attn_mask,
        dropout_p=mha.dropout if mha.training else 0.0,
        scale=logit_scale / math.sqrt(head_dim),
    )  # (B, H, L_q, head_dim)
    return mha.out_proj(out.transpose(1, 2).reshape(B, L_q, E))
