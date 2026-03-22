from typing import NamedTuple

import torch
import torch.nn as nn

from core.models.layers.cached_multi_head_attention import CachedMultiHeadAttention
from core.models.layers.single_head_scorer import CompatibilityScorer


class DecoderCache(NamedTuple):
    """Precomputed values that remain constant across all M deconstruction steps.

    Attributes:
        k: Projected keys for multi-head attention.
            Shape: (B, rollout_size, N+1, d_h).
        v: Projected values for multi-head attention.
            Shape: (B, rollout_size, N+1, d_h).
        z: Seed vectors for diverse rollout conditioning.
            Shape: (B, rollout_size, z_dim).
        h_nodes_expanded: Node embeddings expanded over rollout dimension.
            Shape: (B, rollout_size, N+1, d_h).
    """

    k: torch.Tensor
    v: torch.Tensor
    z: torch.Tensor
    h_nodes_expanded: torch.Tensor


class DecoderState(NamedTuple):
    """Mutable state updated at each deconstruction step.

    Attributes:
        gru_hidden: GRU hidden state accumulating history of selected customers.
            Shape: (B*rollout, d_h).
        selected_mask: Boolean mask of already selected customers.
            True indicates the customer has been selected for removal.
            Shape: (B, rollout, N+1).
    """

    h_last_node: torch.Tensor
    gru_hidden: torch.Tensor
    selected_mask: torch.Tensor


class NDSDecoder(nn.Module):
    """Autoregressive decoder for sequential customer selection in NDS.

    At each step, a GRU updates the query based on the previously selected
    customer, which is then used to attend over encoded nodes. A polynomial
    network conditions the output on a seed vector z to encourage diverse
    rollouts across the search.

    Args:
        d_h: Node embedding dimension.
        z_dim: Seed vector dimension.
        d_poly: Hidden dimension of the polynomial conditioning network.
        num_heads: Number of attention heads.
        C: Tanh clipping constant for final scorer logits.
    """

    def __init__(
        self,
        *,
        d_h: int = 128,
        z_dim: int = 10,
        d_poly: int = 512,
        num_heads: int = 8,
        C: float = 10.0,
    ) -> None:
        super().__init__()
        assert d_h % num_heads == 0

        self.d_h = d_h
        self.num_heads = num_heads
        self.head_dim = d_h // num_heads

        self.start_token = nn.Parameter(torch.zeros(d_h), requires_grad=True)

        self.MHA = CachedMultiHeadAttention(
            embed_dim=d_h,
            num_heads=num_heads,
        )

        self.GRU = nn.GRUCell(d_h, d_h)

        self.poly_net = nn.Sequential(
            nn.Linear(d_h + z_dim, d_poly),
            nn.ReLU(),
            nn.Linear(d_poly, d_h),
        )

        self.final_scorer = CompatibilityScorer(d_h=d_h, C=C, use_projections=False)

    def build_initial_state(
        self,
        *,
        B: int,
        rollout_size: int,
        N: int,
        device: torch.device | None = None,
    ) -> DecoderState:
        """Build zeroed initial state before the first decoding step.

        Args:
            B: Batch size.
            rollout_size: Number of parallel rollouts per instance.
            N: Number of customers (excluding depot).
            device: Target device (optional).

        Returns:
            DecoderState with zeroed GRU hidden state and empty selection mask.
        """
        if not device:
            device = next(self.parameters()).device

        selected_mask = torch.zeros(B, rollout_size, N + 1, dtype=torch.bool, device=device)
        selected_mask[:, :, 0] = True

        h_start = self.start_token.view(1, 1, -1).expand(B, rollout_size, -1)

        return DecoderState(
            h_last_node=h_start,
            gru_hidden=torch.zeros(B * rollout_size, self.d_h, device=device),
            selected_mask=selected_mask,
        )

    def build_cache(
        self,
        *,
        h_nodes: torch.Tensor,
        z: torch.Tensor,
    ) -> DecoderCache:
        """Precompute keys, values, and seed vectors before the decoding loop.

        Args:
            h_nodes: Encoder output node embeddings. Shape: (B, N+1, d_h).
            z: Seed vectors for rollout diversity. Shape: (B, rollout_size, z_dim).

        Returns:
            DecoderCache containing projected keys, values, seed vectors, and expanded node embeddings.
        """
        B, L, d_h = h_nodes.shape
        _, rollout_size, _ = z.shape

        # Expand nodes for rollout dimension
        h_nodes_expanded = h_nodes.unsqueeze(1).expand(B, rollout_size, L, d_h)

        k, v = self.MHA.project_kv(h_nodes_expanded)

        return DecoderCache(k=k, v=v, z=z, h_nodes_expanded=h_nodes_expanded)

    def forward(
        self,
        *,
        state: DecoderState,
        cache: DecoderCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score nodes for selection. A pure (stateless) function.

        Args:
            state (DecoderState): Current state of the decoder.
            cache: Precomputed keys, values, and seed vectors.

        Returns:
            Tuple of:
                log_p: Log-probabilities over all nodes. Shape: (B, rollout_size, N+1).
                gru_hidden_state: A new GRU hidden state.
        """
        B, R, d_h = state.h_last_node.shape

        gru_hidden = self.GRU(
            state.h_last_node.reshape(B * R, d_h),
            state.gru_hidden,
        )
        context = gru_hidden.reshape(B, R, d_h)

        attention_mask = ~state.selected_mask
        mh_out = self.MHA(
            q=context, k_cached=cache.k, v_cached=cache.v, attention_mask=attention_mask
        ).reshape(B, R, self.d_h)

        context_out = mh_out + self.poly_net(torch.cat([mh_out, cache.z], dim=-1))

        log_p = self.final_scorer(
            h_c=context_out, h_nodes=cache.h_nodes_expanded, mask=state.selected_mask
        )

        return log_p, gru_hidden
