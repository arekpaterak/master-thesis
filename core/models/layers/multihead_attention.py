import torch
from torch import nn

from core.models.layers.attention_scaling import (
    AttentionScalingConfig,
    scaled_multihead_attention,
)


class MultiheadAttentionWrapper(nn.Module):
    """Thin wrapper around nn.MultiheadAttention that handles arbitrary leading batch dimensions.

    nn.MultiheadAttention (batch_first=True) requires a single batch dimension (batch, seq, dim).
    This wrapper flattens all leading dimensions before the call and restores them after,
    so tensors of shape (..., L, embed_dim) work transparently.

    For self-attention pass the same tensor as q, k, and v.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_scaling: AttentionScalingConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.attention_scaling = attention_scaling
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, **kwargs
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q: (..., L_q, embed_dim)
            k: (..., L_k, embed_dim)
            v: (..., L_k, embed_dim)
            attention_mask: optional bool mask (..., L_k) where True = attend (allowed).

        Returns:
            (..., L_q, embed_dim)
        """
        batch_shape = q.shape[:-2]
        L_q = q.size(-2)

        q_flat = q.reshape(-1, L_q, q.size(-1))
        k_flat = k.reshape(-1, k.size(-2), k.size(-1))
        v_flat = v.reshape(-1, v.size(-2), v.size(-1))

        if self.attention_scaling is not None:
            out_flat = scaled_multihead_attention(
                self.mha,
                q_flat,
                k_flat,
                v_flat,
                logit_scale=self.attention_scaling.factor(k.size(-2)),
                attend_mask=None
                if attention_mask is None
                else attention_mask.reshape(-1, attention_mask.size(-1)),
            )
            return out_flat.view(*batch_shape, L_q, q.size(-1))

        if attention_mask is not None:
            # nn.MultiheadAttention uses True = ignore, so invert
            mask_flat = (~attention_mask).reshape(-1, attention_mask.size(-1))
        else:
            mask_flat = None

        # need_weights=False routes through the fused scaled_dot_product_attention kernel
        # (the averaged attention weights are unused).
        out_flat, _ = self.mha(
            q_flat, k_flat, v_flat, key_padding_mask=mask_flat, need_weights=False
        )
        return out_flat.view(*batch_shape, L_q, q.size(-1))
