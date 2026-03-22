import torch
import torch.nn.functional as F
from torch import nn


class CachedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        add_bias_kv: bool = False,
    ):
        super().__init__()
        self.d_h = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "d_h must be divisible by num_heads"

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=bias)

    def project_kv(self, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute K and V for the cache.

        Args:
            kv: (..., L, d_h)
        Returns:
            Projected (K, V): each (..., L, d_h)
        """
        return self.W_K(kv), self.W_V(kv)

    def forward(
        self,
        q: torch.Tensor,
        k_cached: torch.Tensor,
        v_cached: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor (..., L_q, d_h) or (..., d_h)
            k_cached: Pre-projected Keys (..., L_k, d_h)
            v_cached: Pre-projected Values (..., L_k, d_h)
            attention_mask: Bool mask (B, 1, L_q, L_k)
        """
        # Normalize query shape to (..., Lq, d_h)
        if q.dim() == k_cached.dim() - 1:
            q = q.unsqueeze(-2)

        # Capture the batch shape (everything before the last two dims)
        # For (B, R, L, d_h), batch_shape is (B, R)
        batch_shape = q.shape[:-2]
        L_q = q.size(-2)
        L_k = k_cached.size(-2)

        def split_heads(x: torch.Tensor, L: int) -> torch.Tensor:
            return x.view(*batch_shape, L, self.num_heads, self.head_dim).transpose(-3, -2)

        q_heads = split_heads(self.W_Q(q), L_q)  # (..., H, L_q, d_h)
        k_heads = split_heads(k_cached, L_k)  # (..., H, L_k, d_h)
        v_heads = split_heads(v_cached, L_k)  # (..., H, L_k, d_h)

        if attention_mask is not None and attention_mask.dim() == len(batch_shape) + 1:
            # If (..., Lk), convert to (..., 1, 1, Lk)
            attention_mask = attention_mask.unsqueeze(-2).unsqueeze(-2)

        attn_out = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=attention_mask,
            is_causal=False,
        )

        # Merge heads and project out
        # (..., H, Lq, d) -> (..., Lq, H, d) -> (..., Lq, D)
        attn_out = attn_out.transpose(-3, -2).reshape(*batch_shape, L_q, self.embed_dim)
        out = self.W_O(attn_out)

        # Return (..., d_h) if input was (..., d_h), else (..., Lq, d_h)
        return out.squeeze(-2) if L_q == 1 else out
