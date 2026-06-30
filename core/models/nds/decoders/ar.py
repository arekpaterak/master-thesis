from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from core.models.entropy import entropy
from core.models.layers.batched_gru_cell import BatchedGRUCell
from core.models.layers.cached_multihead_attention import CachedMultiheadAttention
from core.models.layers.compatibility_scorer import CompatibilityScorer
from core.models.nds.decoders.base import DestroyDecoder
from core.models.rollout import Rollout
from core.states.destroy import DestroyState


class DecoderCache(NamedTuple):
    """Precomputed values that remain constant across all deconstruction steps.

    Keys, values, and node embeddings carry no rollout dimension: they are projected
    once per instance and broadcast across rollouts inside attention, so the cache size
    is independent of the number of rollouts.

    Here N is the number of nodes the decoder selects over (including the depot for
    problems that have one, e.g. CVRP).

    Attributes:
        k: Projected keys for multi-head attention, shared across rollouts.
            Shape: (B, N, hidden_dim).
        v: Projected values for multi-head attention, shared across rollouts.
            Shape: (B, N, hidden_dim).
        seed: Seed vectors for diverse rollout conditioning.
            Shape: (B, rollout_size, seed_dim).
        h_nodes: Node embeddings.
            Shape: (B, N, hidden_dim).
    """

    k: torch.Tensor
    v: torch.Tensor
    seed: torch.Tensor
    h_nodes: torch.Tensor


class DecoderState(NamedTuple):
    """Decoder-only recurrent state updated at each deconstruction step.

    Attributes:
        gru_hidden: GRU hidden state accumulating history of selected customers.
            Shape: (B, rollout_size, hidden_dim).
    """

    gru_hidden: torch.Tensor

    # TODO: Refactor, it is stupid
    @staticmethod
    def transition(gru_hidden: torch.Tensor) -> DecoderState:
        return DecoderState(gru_hidden=gru_hidden)


class DecoderOutput(NamedTuple):
    logits: torch.Tensor
    gru_hidden: torch.Tensor


class AutoregressiveNDSDecoder(DestroyDecoder):
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
        with_poly: bool = True,
        C: float = 10.0,
        use_hidden_decoding_state: bool = True,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self._seed_dim = seed_dim
        self.with_poly = with_poly

        self.start_token = nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        self.MHA = CachedMultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
        )

        self.GRU = BatchedGRUCell(hidden_dim, hidden_dim) if use_hidden_decoding_state else None

        self.poly_net = (
            nn.Sequential(
                nn.Linear(hidden_dim + seed_dim, poly_dim),
                nn.ReLU(),
                nn.Linear(poly_dim, hidden_dim),
            )
            if with_poly
            else None
        )

        self.final_scorer = CompatibilityScorer(hidden_dim=hidden_dim, C=C, use_projections=False)

    @property
    def seed_dim(self) -> int:
        return self._seed_dim

    def rollout(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
        action_mask: torch.Tensor | None,
        deconstruction_size: int,
        greedy: bool,
        **kwargs,
    ) -> Rollout:
        """Autoregressively select ``deconstruction_size`` nodes from ``h_nodes``.

        ``action_mask`` (B, N) is the initial forbidden-node mask (e.g. the CVRP depot);
        it seeds the transitionable problem state stepped through the decoding loop.
        """
        if action_mask is None:
            raise ValueError("AutoregressiveNDSDecoder.rollout requires an action_mask.")

        B, N, _ = h_nodes.shape
        R = seed.shape[1]

        cache = self.build_cache(h_nodes=h_nodes, seed=seed)
        problem_state = DestroyState.from_mask(action_mask.unsqueeze(1).expand(B, R, N))
        decoder_state = self.build_initial_state(
            batch_size=B, rollout_size=R, device=h_nodes.device
        )

        selected = []
        log_prob_of_selected = []
        entropies = []
        entropies_norm = []

        for _ in range(deconstruction_size):
            logits, gru_hidden = self.forward(
                problem_state=problem_state, decoder_state=decoder_state, cache=cache
            )

            distribution = torch.distributions.Categorical(logits=logits)
            idx = logits.argmax(dim=-1) if greedy else distribution.sample()
            selected.append(idx)
            log_prob_of_selected.append(distribution.log_prob(idx))

            step_entropy = entropy(logits)
            entropies.append(step_entropy.entropy.detach())
            entropies_norm.append(step_entropy.entropy_norm.detach())

            problem_state = problem_state.transition(action=idx.unsqueeze(-1))
            decoder_state = decoder_state.transition(gru_hidden)

        return Rollout.create(
            actions=torch.stack(selected, dim=-1).detach(),
            log_prob=torch.stack(log_prob_of_selected, dim=-1).sum(dim=-1),
            entropy=torch.stack(entropies, dim=-1).mean(dim=-1),
            entropy_norm=torch.stack(entropies_norm, dim=-1).mean(dim=-1),
        )

    def build_initial_state(
        self,
        *,
        batch_size: int,
        rollout_size: int,
        device: torch.device | None = None,
    ) -> DecoderState:
        """Build zeroed initial state before the first decoding step.

        Args:
            batch_size: Batch size.
            rollout_size: Number of parallel rollouts per instance.
            device: Target device (optional).

        Returns:
            DecoderState with zeroed GRU hidden state.
        """
        if not device:
            device = next(self.parameters()).device

        return DecoderState(
            gru_hidden=torch.zeros(batch_size, rollout_size, self.hidden_dim, device=device),
        )

    def build_cache(
        self,
        *,
        h_nodes: torch.Tensor,
        seed: torch.Tensor,
    ) -> DecoderCache:
        """Precompute keys, values, and seed vectors before the decoding loop.

        Keys and values are projected once at the per-instance shape (B, N, hidden_dim);
        they are broadcast across rollouts inside attention, so the cache does not grow
        with the number of rollouts.

        Args:
            h_nodes: Encoder output node embeddings. Shape: (B, N, hidden_dim).
            seed: Seed vectors for rollout diversity. Shape: (B, rollout_size, seed_dim).

        Returns:
            DecoderCache containing projected keys, values, seed vectors, and node embeddings.
        """
        k, v = self.MHA.project_kv(h_nodes)

        return DecoderCache(k=k, v=v, seed=seed, h_nodes=h_nodes)

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
                logits: Logits over all nodes. Shape: (B, rollout_size, N).
                gru_hidden_state: A new GRU hidden state.
        """
        B, _, hidden_dim = cache.h_nodes.shape
        R = cache.seed.shape[1]

        last_node_idx = problem_state.last_node_idx
        if last_node_idx is None:
            h_last_node = self.start_token.view(1, 1, -1).expand(B, R, -1)
        else:
            # last_node_idx: (B, R, 1). Gather the per-rollout last-node embedding from the
            # shared (B, N, hidden_dim) node embeddings.
            h_last_node = cache.h_nodes.gather(
                dim=1,
                index=last_node_idx.expand(B, R, hidden_dim),
            )

        gru_hidden = decoder_state.gru_hidden
        if self.GRU is not None:
            gru_hidden = self.GRU(h_last_node, gru_hidden)
            context = gru_hidden
        else:
            context = h_last_node

        attention_mask = ~problem_state.action_mask
        mha_out = self.MHA(
            q=context, k_cached=cache.k, v_cached=cache.v, attention_mask=attention_mask
        ).reshape(B, R, self.hidden_dim)

        if self.poly_net is None:
            context = mha_out
        else:
            context = mha_out + self.poly_net(torch.cat([mha_out, cache.seed], dim=-1))

        # Shared-keys scoring: the R rollouts in context (B, R, D) all score against
        # the same per-instance nodes (B, N, D), avoiding any broadcast over R.
        logits = self.final_scorer(
            h_context=context,
            h_nodes=cache.h_nodes,
            mask=problem_state.action_mask,
        )

        return DecoderOutput(
            logits=logits,
            gru_hidden=gru_hidden,
        )
