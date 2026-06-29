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
            q: Query tensor (B, R, embed_dim) or (B, R, L_q, embed_dim).
            k_cached: Pre-projected Keys (B, L_k, embed_dim), shared across the R rollouts.
            v_cached: Pre-projected Values (B, L_k, embed_dim), shared across the R rollouts.
            attention_mask: optional bool mask (B, R, L_k) where True = attend (allowed).
                Note: opposite convention to MultiHeadAttention - mirrors F.scaled_dot_product_attention.

        The R rollouts are folded into the query-sequence dimension so the queries and
        the shared keys/values keep the same batch dimension B. This keeps K/V stored
        once per instance (no multiplication by R) and lets the fused
        scaled_dot_product_attention kernel apply, instead of forcing the math path to
        expand K/V over R at every step.
        """
        # Normalize query shape to (B, R, L_q, embed_dim)
        if q.dim() == 3:
            q = q.unsqueeze(-2)
        B, R, L_q, _ = q.shape
        L_k = k_cached.size(-2)
        num_heads, head_dim = self.num_heads, self.head_dim

        # Fold the R rollouts (and their L_q queries) into one query-sequence axis so q
        # shares batch B with k/v: (B, num_heads, R*L_q, head_dim).
        q_heads = self.W_Q(q).reshape(B, R * L_q, num_heads, head_dim).transpose(1, 2)
        k_heads = k_cached.view(B, L_k, num_heads, head_dim).transpose(
            1, 2
        )  # (B, num_heads, L_k, head_dim)
        v_heads = v_cached.view(B, L_k, num_heads, head_dim).transpose(
            1, 2
        )  # (B, num_heads, L_k, head_dim)

        if attention_mask is not None:
            # (B, R, L_k) is the per-rollout mask; broadcast it over the L_q queries of
            # each rollout and over heads -> (B, 1, R*L_q, L_k).
            attention_mask = (
                attention_mask[:, :, None, :].expand(B, R, L_q, L_k).reshape(B, 1, R * L_q, L_k)
            )

        attn_out = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=attention_mask,
            is_causal=False,
        )  # (B, num_heads, R*L_q, head_dim)

        # Merge heads and unfold the rollouts: (B, R, L_q, embed_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, R, L_q, self.embed_dim)
        out = self.W_O(attn_out)

        # Return (B, R, embed_dim) if the query was a single token, else (B, R, L_q, embed_dim)
        return out.squeeze(-2) if L_q == 1 else out
