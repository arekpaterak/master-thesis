from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.embeddings.cvrp import CVRPInitialEmbedding
from core.models.embeddings.tsp import TSPInitialEmbedding
from core.models.nds.config import EncoderConfig


class NDSEncoder(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        encoder_config: EncoderConfig | None = None,
        with_neighbours_encoding: bool = True,
        with_route_encoding: bool = True,
        initial_embedding: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.problem_name = problem_name
        if initial_embedding is not None:
            if problem_name == "cvrp" and not isinstance(initial_embedding, CVRPInitialEmbedding):
                raise TypeError(
                    f"Expected CVRPInitialEmbedding for problem_name='cvrp', got {type(initial_embedding).__name__}"
                )
            if problem_name == "tsp" and not isinstance(initial_embedding, TSPInitialEmbedding):
                raise TypeError(
                    f"Expected TSPInitialEmbedding for problem_name='tsp', got {type(initial_embedding).__name__}"
                )
            self.initial_embedding = initial_embedding
        elif problem_name == "cvrp":
            self.initial_embedding = CVRPInitialEmbedding(hidden_dim=hidden_dim)
        elif problem_name == "tsp":
            self.initial_embedding = TSPInitialEmbedding(hidden_dim=hidden_dim)
        else:
            raise ValueError(f"Unknown problem_name: {problem_name}")

        encoder_config = encoder_config or EncoderConfig()
        attention_layer = lambda: AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            norm_type=encoder_config.norm_type,
            pre_norm=encoder_config.pre_norm,
            ff_activation=encoder_config.ff_activation,
        )
        self.pre_attention = nn.Sequential(
            attention_layer(),
            attention_layer(),
        )
        self.neighbours_encoding = (
            NeighboursEncodingLayer(hidden_dim=hidden_dim) if with_neighbours_encoding else None
        )
        self.route_encoding = (
            RouteEncodingLayer(hidden_dim=hidden_dim) if with_route_encoding else None
        )
        self.post_attention = nn.Sequential(
            attention_layer(),
            attention_layer(),
        )

    def forward(self, **features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            **features: Problem-specific feature dictionary.

        Returns:
            (B, N, hidden_dim) or (B, N+1, hidden_dim)
        """
        h = self.initial_embedding(**features)
        h = self.pre_attention(h)

        if self.neighbours_encoding is not None and "neighbours_index" in features:
            h = self.neighbours_encoding(h, neighbours_index=features["neighbours_index"])

        if self.route_encoding is not None and "route_index" in features:
            h = self.route_encoding(h, route_index=features["route_index"])

        h = self.post_attention(h)
        return h


class InstanceNorm(nn.Module):
    """InstanceNorm1d for (..., N, D) tensors without manual transpose at call sites."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm1d(hidden_dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(-1, -2)).transpose(-1, -2)


class SwiGLUFeedForward(nn.Module):
    """Swish-Gated Linear Unit feed-forward block (Shazeer, 2020).

    FF(x) = W_out (SiLU(W_gate x) * W_value x)

    The SiLU(W_gate x) acts as a soft, per-feature multiplicative gate on the
    linear W_value x: each scalar decides how much of its value to let through.
    A ReLU MLP can only zero out or pass; this can modulate. Standard in modern
    LLMs (LLaMA, Mistral).
    Biases are omitted by convention -- with three
    linears any constant offset is absorbed by the next layer.

    Note: keeps the same ff_dim as the ReLU variant, so
    parameter count is ~1.5x larger.
    """

    def __init__(self, hidden_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.out_proj = nn.Linear(ff_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


def _build_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "instance":
        return InstanceNorm(hidden_dim)
    if norm_type == "rms":
        return nn.RMSNorm(hidden_dim)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


def _build_feedforward(ff_activation: str, hidden_dim: int, ff_dim: int) -> nn.Module:
    if ff_activation == "relu":
        return nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
    if ff_activation == "swiglu":
        return SwiGLUFeedForward(hidden_dim, ff_dim)
    raise ValueError(f"Unknown ff_activation: {ff_activation!r}")


class AttentionLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        norm_type: str = "instance",
        pre_norm: bool = False,
        ff_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.MHA = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = _build_norm(norm_type, hidden_dim)
        self.norm2 = _build_norm(norm_type, hidden_dim)
        self.FF = _build_feedforward(ff_activation, hidden_dim, ff_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._residual(x, self._self_attention, self.norm1)
        x = self._residual(x, self.FF, self.norm2)
        return x

    def _residual(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
        norm: nn.Module,
    ) -> torch.Tensor:
        if self.pre_norm:
            return x + sublayer(norm(x))
        return norm(x + sublayer(x))

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.MHA(x, x, x)
        return out


class NeighboursEncodingLayer(nn.Module):
    """
    Named Message Passing Layer in the original paper (Hottung et al. 2025).
    """

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.neighbour_projector = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbour_combiner = nn.Linear(2 * hidden_dim, hidden_dim)
        self.feedforward_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = InstanceNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        *,
        neighbours_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, N+offset, hidden_dim) node embeddings. For problems with a depot (e.g. CVRP),
               offset=1 and h[:, 0] is the depot embedding with h[:, 1:] as customer embeddings.
               For problems without a depot (e.g. TSP), offset=0 and all nodes are treated equally.
            neighbours_index: (B, N, 2) integer indices of the previous and next nodes for each non-depot node.

        Returns:
            (B, N+offset, hidden_dim) updated node embeddings after message passing.
        """
        # offset=1 for problems with a depot (CVRP), offset=0 when all nodes are customers (TSP)
        offset = h.shape[-2] - neighbours_index.shape[-2]
        customer_embeddings = h[:, offset:]  # (B, N, hidden_dim)

        previous = self._gather(h, neighbours_index[:, :, 0])  # (B, N, hidden_dim)
        next = self._gather(h, neighbours_index[:, :, 1])  # (B, N, hidden_dim)

        neighbour_info = self.neighbour_projector(previous) + self.neighbour_projector(next)
        combined = torch.cat([customer_embeddings, neighbour_info], dim=-1)
        update = self.feedforward_layer(F.relu(self.neighbour_combiner(combined)))
        updated = self.norm(customer_embeddings + update)

        return torch.cat([h[:, :offset], updated], dim=1)

    def _gather(self, embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, N+1, hidden_dim)
            indices: (B, N)

        Returns:
            (B, N, hidden_dim)
        """
        B, N = indices.shape
        hidden_dim = embeddings.shape[2]
        return embeddings.gather(1, indices[:, :, None].expand(B, N, hidden_dim))


class RouteEncodingLayer(nn.Module):
    """
    Named Tour Encoding Layer in the original paper (Hottung et al. 2025).
    """

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        self.tour_combiner = nn.Linear(2 * hidden_dim, hidden_dim)
        self.feedforward_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm = InstanceNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        *,
        route_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: (B, N+offset, hidden_dim)
            route_index: (B, N) integer route assignment per non-depot node.

        Returns:
            (B, N+offset, hidden_dim)
        """
        B, L, hidden_dim = node_embeddings.shape
        N = route_index.shape[-1]
        offset = L - N

        route_index = route_index.reshape(B, N)
        customer_embeddings = node_embeddings[:, offset:]

        # Static upper bound on route count: at most one route per customer.
        # Sized to N+1 to cover both 0-indexed and 1-indexed (head-node) schemes
        # without a host sync on max(route_index).
        max_routes = N + 1
        idx = route_index.unsqueeze(-1).expand(B, N, hidden_dim)
        route_embeddings = torch.zeros(
            B, max_routes, hidden_dim, device=node_embeddings.device, dtype=node_embeddings.dtype
        )
        route_embeddings = route_embeddings.scatter_reduce(
            1, idx, customer_embeddings, reduce="mean", include_self=False
        )
        route_mean_per_customer = route_embeddings.gather(1, idx)  # (B, N, hidden_dim)

        combined = F.relu(
            self.tour_combiner(torch.cat([customer_embeddings, route_mean_per_customer], dim=-1))
        )
        update = self.feedforward_layer(combined)
        updated_customer_embeddings = self.norm(customer_embeddings + update)

        return torch.cat([node_embeddings[:, :offset], updated_customer_embeddings], dim=1)
