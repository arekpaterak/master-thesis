import torch
import torch.nn.functional as F
from torch import nn


class CachedMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        add_bias_kv: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=add_bias_kv)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=bias)

    def project_kv(self, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute K and V for the cache.

        Args:
            kv: (..., L, embed_dim)
        Returns:
            Projected (K, V): each (..., L, embed_dim)
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
            q: Query tensor (..., L_q, embed_dim) or (..., embed_dim)
            k_cached: Pre-projected Keys (..., L_k, embed_dim)
            v_cached: Pre-projected Values (..., L_k, embed_dim)
            attention_mask: optional bool mask (..., L_k) where True = attend (allowed).
                Note: opposite convention to MultiHeadAttention - mirrors F.scaled_dot_product_attention.
        """
        # Normalize query shape to (..., Lq, embed_dim)
        if q.dim() == k_cached.dim() - 1:
            q = q.unsqueeze(-2)

        # Capture the batch shape (everything before the last two dims)
        # For (B, R, L, embed_dim), batch_shape is (B, R)
        batch_shape = q.shape[:-2]
        L_q = q.size(-2)
        L_k = k_cached.size(-2)

        def split_heads(x: torch.Tensor, L: int) -> torch.Tensor:
            return x.view(*batch_shape, L, self.num_heads, self.head_dim).transpose(-3, -2)

        q_heads = split_heads(self.W_Q(q), L_q)  # (..., H, L_q, head_dim)
        k_heads = split_heads(k_cached, L_k)  # (..., H, L_k, head_dim)
        v_heads = split_heads(v_cached, L_k)  # (..., H, L_k, head_dim)

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
        # (..., H, Lq, head_dim) -> (..., Lq, H, head_dim) -> (..., Lq, embed_dim)
        attn_out = attn_out.transpose(-3, -2).reshape(*batch_shape, L_q, self.embed_dim)
        out = self.W_O(attn_out)

        # Return (..., embed_dim) if input was (..., embed_dim), else (..., Lq, embed_dim)
        return out.squeeze(-2) if L_q == 1 else out
