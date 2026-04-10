import torch
import torch.nn as nn


class CVRPInitialEmbedding(nn.Module):
    """
    Separate linear projections for depot and customer nodes.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        depot_features_dim: int = 2,
        customer_features_dim: int = 3,
    ) -> None:
        super().__init__()
        assert depot_features_dim > 0
        assert customer_features_dim > 0
        self.depot_projection = nn.Linear(depot_features_dim, hidden_dim)
        self.customer_projection = nn.Linear(customer_features_dim, hidden_dim)

    def forward(self, **features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            **features: dictionary containing:
                - `depot_features` (B, 1, depot_dim)
                - `customer_features` (B, N, customer_dim)

        Returns:
            (B, N+1, H)
        """
        depot = features["depot_features"]
        customer = features["customer_features"]

        depot_embedding = self.depot_projection(depot)
        customer_embeddings = self.customer_projection(customer)
        return torch.cat([depot_embedding, customer_embeddings], dim=-2)
