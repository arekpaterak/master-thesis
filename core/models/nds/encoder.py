import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.embeddings.cvrp import CVRPInitialEmbedding
from core.models.embeddings.tsp import TSPInitialEmbedding


class NDSEncoder(nn.Module):
    def __init__(
        self,
        *,
        problem_name: str,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        with_neighbours_encoding: bool = True,
        with_tour_encoding: bool = True,
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

        attention_layer = lambda: AttentionLayer(
            hidden_dim=hidden_dim, num_heads=num_heads, ff_dim=ff_dim
        )
        self.pre_attention = nn.Sequential(
            attention_layer(),
            attention_layer(),
        )
        self.neighbours_encoding = (
            NeighboursEncodingLayer(hidden_dim=hidden_dim, ff_dim=ff_dim)
            if with_neighbours_encoding
            else None
        )
        self.tour_encoding = (
            TourEncodingLayer(hidden_dim=hidden_dim, ff_dim=ff_dim) if with_tour_encoding else None
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

        if self.tour_encoding is not None and "tour_index" in features:
            h = self.tour_encoding(h, tour_index=features["tour_index"])

        h = self.post_attention(h)
        return h


class AttentionLayer(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
    ) -> None:
        super().__init__()
        self.MHA = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.FF = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, num_nodes, hidden_dim = x.shape
        attn_out, _ = self.MHA(x, x, x)
        x = self.norm1((x + attn_out).reshape(B * num_nodes, hidden_dim)).reshape(
            B, num_nodes, hidden_dim
        )
        x = self.norm2((x + self.FF(x)).reshape(B * num_nodes, hidden_dim)).reshape(
            B, num_nodes, hidden_dim
        )
        return x


class NeighboursEncodingLayer(nn.Module):
    """
    Named Message Passing Layer in the original paper (Hottung et al. 2025).
    """

    def __init__(self, *, hidden_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # previous neighbour projection
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # next neighbour projection
        self.W3 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.FF = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

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

        prev_emb = self.W1(previous)
        next_emb = self.W2(next)
        agg = prev_emb + next_emb

        update = self.FF(F.relu(self.W3(torch.cat([customer_embeddings, agg], dim=-1))))
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


# TODO: Rename to RouteEncodingLayer to unify naming
class TourEncodingLayer(nn.Module):
    def __init__(self, *, hidden_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.tour_combiner = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.FF = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        *,
        tour_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: (B, N+1, hidden_dim)
            tour_index: (B, N)

        Returns:
            (B, N+1, hidden_dim)
        """
        B, L, hidden_dim = node_embeddings.shape
        N = L - 1

        tour_index = tour_index.reshape(B, N)

        customer_embeddings = node_embeddings[:, 1:]

        max_tours = int(tour_index.max().item()) + 1
        _, _, hidden_dim = node_embeddings.shape
        idx = tour_index.unsqueeze(-1).expand(B, N, hidden_dim)
        tour_embeddings = torch.zeros(
            B, max_tours, hidden_dim, device=node_embeddings.device, dtype=node_embeddings.dtype
        )
        tour_embeddings = tour_embeddings.scatter_reduce(
            1, idx, customer_embeddings, reduce="mean", include_self=False
        )
        tour_mean_per_customer = tour_embeddings.gather(1, idx)  # (B, N, hidden_dim)

        update = self.FF(
            F.relu(
                self.tour_combiner(torch.cat([customer_embeddings, tour_mean_per_customer], dim=-1))
            )
        )
        updated_customer_embeddings = self.norm(customer_embeddings + update)

        return torch.cat(
            [node_embeddings[:, :1], updated_customer_embeddings], dim=1
        )  # (B, N+1, hidden_dim)
