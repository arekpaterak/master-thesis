from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from core.models.layers.cached_multi_head_attention import CachedMultiHeadAttention
from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.states.destroy import DestroyState


class DecoderCache(NamedTuple):
    """Precomputed values that remain constant across all deconstruction steps.

    Attributes:
        k: Projected keys for multi-head attention.
            Shape: (B, rollout_size, N+1, hidden_dim).
        v: Projected values for multi-head attention.
            Shape: (B, rollout_size, N+1, hidden_dim).
        seed: Seed vectors for diverse rollout conditioning.
            Shape: (B, rollout_size, seed_dim).
        h_nodes_expanded: Node embeddings expanded over rollout dimension.
            Shape: (B, rollout_size, N+1, hidden_dim).
    """

    k: torch.Tensor
    v: torch.Tensor
    seed: torch.Tensor
    h_nodes_expanded: torch.Tensor


class DecoderState(NamedTuple):
    """Decoder-only recurrent state updated at each deconstruction step.

    Attributes:
        gru_hidden: GRU hidden state accumulating history of selected customers.
            Shape: (B*rollout, hidden_dim).
    """

    gru_hidden: torch.Tensor

    @staticmethod
    def transition(gru_hidden: torch.Tensor) -> DecoderState:
        return DecoderState(gru_hidden=gru_hidden)


class DecoderOutput(NamedTuple):
    log_p: torch.Tensor
    gru_hidden: torch.Tensor


class NDSDecoder(nn.Module):
    """Autoregressive decoder for sequential customer selection in NDS.

    At each step, a GRU updates the query based on the previously selected
    customer, which is then used to attend over encoded nodes. A polynomial
    network conditions the output on a seed vector z to encourage diverse
    rollouts across the search.

    Args:
        hidden_dim: Node embedding dimension.
        seed_dim: Seed vector dimension.
        poly_dim: Hidden dimension of the polynomial conditioning network.
        num_heads: Number of attention heads.
        C: Tanh clipping constant for final scorer logits.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        seed_dim: int = 10,
        poly_dim: int = 512,
        num_heads: int = 8,
        C: float = 10.0,
        use_hidden_decoding_state: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.start_token = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        self.MHA = CachedMultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
        )

        self.GRU = nn.GRUCell(hidden_dim, hidden_dim) if use_hidden_decoding_state else None

        self.poly_net = nn.Sequential(
            nn.Linear(hidden_dim + seed_dim, poly_dim),
            nn.ReLU(),
            nn.Linear(poly_dim, hidden_dim),
        )

        self.final_scorer = CompatibilityScorer(hidden_dim=hidden_dim, C=C, use_projections=False)

    def build_initial_state(
        self,
        *,
        B: int,
        rollout_size: int,
        device: torch.device | None = None,
    ) -> DecoderState:
        """Build zeroed initial state before the first decoding step.

        Args:
            B: Batch size.
            rollout_size: Number of parallel rollouts per instance.
            device: Target device (optional).

        Returns:
            DecoderState with zeroed GRU hidden state.
        """
        if not device:
            device = next(self.parameters()).device

        return DecoderState(
            gru_hidden=torch.zeros(B * rollout_size, self.hidden_dim, device=device),
        )

    def build_cache(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
    ) -> DecoderCache:
        """Precompute keys, values, and seed vectors before the decoding loop.

        Args:
            h_nodes: Encoder output node embeddings. Shape: (B, N+1, hidden_dim).
            seed: Seed vectors for rollout diversity. Shape: (B, rollout_size, seed_dim).

        Returns:
            DecoderCache containing projected keys, values, seed vectors, and expanded node embeddings.
        """
        B, L, hidden_dim = h_nodes.shape
        _, rollout_size, _ = seed.shape

        # Expand nodes for rollout dimension
        h_nodes_expanded = h_nodes.unsqueeze(1).expand(B, rollout_size, L, hidden_dim)

        k, v = self.MHA.project_kv(h_nodes_expanded)

        return DecoderCache(k=k, v=v, seed=seed, h_nodes_expanded=h_nodes_expanded)

    def forward(
        self,
        *,
        problem_state: DestroyState,
        decoder_state: DecoderState,
        cache: DecoderCache,
    ) -> DecoderOutput:
        """Score nodes for selection. A pure (stateless) function.

        Args:
            problem_state: Problem state that provides action_mask and last_node_idx.
            decoder_state (DecoderState): Current state of the decoder.
            cache: Precomputed keys, values, and seed vectors.

        Returns:
            DecoderOutput:
                log_p: Log-probabilities over all nodes. Shape: (B, rollout_size, N+1).
                gru_hidden_state: A new GRU hidden state.
        """
        B, R, _, _ = cache.h_nodes_expanded.shape
        hidden_dim = self.hidden_dim

        last_node_idx = problem_state.last_node_idx
        if last_node_idx is None:
            h_last_node = self.start_token.view(1, 1, -1).expand(B, R, -1)
        else:
            h_last_node = cache.h_nodes_expanded.gather(
                dim=-2,
                index=last_node_idx.unsqueeze(-1).expand(B, R, 1, hidden_dim),
            ).squeeze(-2)

        gru_hidden = decoder_state.gru_hidden
        if self.GRU is not None:
            gru_hidden = self.GRU(
                h_last_node.reshape(B * R, hidden_dim),
                gru_hidden,
            )
            context = gru_hidden.reshape(B, R, hidden_dim)
        else:
            context = h_last_node.reshape(B, R, hidden_dim)

        attention_mask = ~problem_state.action_mask
        mh_out = self.MHA(
            q=context, k_cached=cache.k, v_cached=cache.v, attention_mask=attention_mask
        ).reshape(B, R, self.hidden_dim)

        context_out = mh_out + self.poly_net(torch.cat([mh_out, cache.seed], dim=-1))

        log_p = self.final_scorer(
            h_c=context_out,
            h_nodes=cache.h_nodes_expanded,
            mask=problem_state.action_mask,
        )

        return DecoderOutput(
            log_p=log_p,
            gru_hidden=gru_hidden,
        )
