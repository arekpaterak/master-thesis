import torch
import torch.nn as nn
import torch.nn.functional as F


class NDSEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ff_dim: int = 512,
        with_neighbours_encoding: bool = True,
        with_tour_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.initial_embedding = CVRPInitialNodeEmbedding(d_h=hidden_dim)
        attention_layer = lambda: AttentionLayer(d_h=hidden_dim, num_heads=num_heads, d_ff=ff_dim)
        self.pre_attention = nn.Sequential(
            attention_layer(),
            attention_layer(),
        )
        self.neighbours_encoding = (
            NeighboursEncodingLayer(d_h=hidden_dim, d_ff=ff_dim)
            if with_neighbours_encoding
            else nn.Identity()
        )
        self.tour_encoding = (
            TourEncodingLayer(d_h=hidden_dim, d_ff=ff_dim) if with_tour_encoding else nn.Identity()
        )
        self.post_attention = nn.Sequential(
            attention_layer(),
            attention_layer(),
        )

    def forward(
        self,
        *,
        depot_location: torch.Tensor,
        customer_features: torch.Tensor,
        neighbours_index: torch.Tensor,
        tour_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depot_location: (B, 2)
            customer_features: (B, N, 3)
            neighbours_index: (B, N, 2)
            tour_index: (B, N)

        Returns:
            (B, N+1, d_h)
        """
        h = self.initial_embedding(
            depot_location=depot_location, customer_features=customer_features
        )
        h = self.pre_attention(h)
        h = self.neighbours_encoding(h, neighbours_index=neighbours_index)
        h = self.tour_encoding(h, tour_index=tour_index)
        h = self.post_attention(h)
        return h


class CVRPInitialNodeEmbedding(nn.Module):
    """
    Separate linear projections for depot and customer nodes.

    Default:
    - Depot: (x, y) -> d_h
    - Customer: (x, y, demand) -> d_h
    """

    def __init__(
        self,
        *,
        d_h: int,
        depot_features_dim: int = 2,
        customer_features_dim: int = 3,
    ) -> None:
        super().__init__()
        assert depot_features_dim > 0
        assert customer_features_dim > 0
        self.depot_projection = nn.Linear(depot_features_dim, d_h)  # default: [x, y]
        self.customer_projection = nn.Linear(customer_features_dim, d_h)  # default: [x, y, demand]

    def forward(
        self,
        *,
        depot_location: torch.Tensor,
        customer_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depot_location: (B, 1, depot_features_dim)
            customer_features: (B, N, customer_features_dim)

        Returns:
            (B, N+1, d_h)
        """
        depot_embedding = self.depot_projection(depot_location)
        customer_embeddings = self.customer_projection(customer_features)
        return torch.cat([depot_embedding, customer_embeddings], dim=1)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        *,
        d_h: int,
        num_heads: int,
        d_ff: int,
    ) -> None:
        super().__init__()
        self.MHA = nn.MultiheadAttention(d_h, num_heads, batch_first=True)
        self.norm1 = nn.BatchNorm1d(d_h)
        self.FF = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_h),
        )
        self.norm2 = nn.BatchNorm1d(d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        attn_out, _ = self.MHA(x, x, x)
        x = self.norm1((x + attn_out).reshape(B * L, d)).reshape(B, L, d)
        x = self.norm2((x + self.FF(x)).reshape(B * L, d)).reshape(B, L, d)
        return x


class NeighboursEncodingLayer(nn.Module):
    """
    Named Message Passing Layer in the original paper.
    """

    def __init__(self, *, d_h: int, d_ff: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(d_h, d_h, bias=False)  # previous neighbour projection
        self.W2 = nn.Linear(d_h, d_h, bias=False)  # next neighbour projection
        self.W3 = nn.Linear(2 * d_h, d_h, bias=False)
        self.FF = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_h),
        )
        self.norm = nn.LayerNorm(d_h)

    def forward(
        self,
        h: torch.Tensor,
        *,
        neighbours_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, N+1, d_h) node embeddings, where h[:, 0] is the depot embedding and h[:, 1:] are customer embeddings
            neighbours_index: (B, N, 2) integer indices of the previous and next nodes for each customer

        Returns:
            (B, N+1, d_h) updated node embeddings after message passing
        """
        customer_embeddings = h[:, 1:]  # (B, N, d_h)

        previous = self._gather(h, neighbours_index[:, :, 0])  # (B, N, d_h)
        next = self._gather(h, neighbours_index[:, :, 1])  # (B, N, d_h)

        prev_emb = self.W1(previous)
        next_emb = self.W2(next)
        agg = prev_emb + next_emb

        update = self.FF(F.relu(self.W3(torch.cat([customer_embeddings, agg], dim=-1))))
        updated = self.norm(customer_embeddings + update)

        return torch.cat([h[:, :1], updated], dim=1)

    def _gather(self, embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, N+1, d_h)
            indices: (B, N)

        Returns:
            (B, N, d_h)
        """
        B, N = indices.shape
        d_h = embeddings.shape[2]
        return embeddings.gather(1, indices[:, :, None].expand(B, N, d_h))


# TODO: Rename to RouteEncodingLayer to unify naming
class TourEncodingLayer(nn.Module):
    def __init__(self, *, d_h: int, d_ff: int) -> None:
        super().__init__()
        self.tour_combiner = nn.Linear(2 * d_h, d_h, bias=False)
        self.FF = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_h),
        )
        self.norm = nn.LayerNorm(d_h)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        *,
        tour_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings: (B, N+1, d_h)
            tour_index: (B, N)

        Returns:
            (B, N+1, d_h)
        """
        B, L, d_h = node_embeddings.shape
        N = L - 1

        tour_index = tour_index.reshape(B, N)

        customer_embeddings = node_embeddings[:, 1:]

        max_tours = int(tour_index.max().item()) + 1
        idx = tour_index.unsqueeze(-1).expand(B, N, d_h)
        tour_embeddings = torch.zeros(
            B, max_tours, d_h, device=node_embeddings.device, dtype=node_embeddings.dtype
        )
        tour_embeddings = tour_embeddings.scatter_reduce(
            1, idx, customer_embeddings, reduce="mean", include_self=False
        )
        tour_mean_per_customer = tour_embeddings.gather(1, idx)  # (B, N, d_h)

        update = self.FF(
            F.relu(
                self.tour_combiner(torch.cat([customer_embeddings, tour_mean_per_customer], dim=-1))
            )
        )
        updated_customer_embeddings = self.norm(customer_embeddings + update)

        return torch.cat(
            [node_embeddings[:, :1], updated_customer_embeddings], dim=1
        )  # (B, N+1, d_h)
