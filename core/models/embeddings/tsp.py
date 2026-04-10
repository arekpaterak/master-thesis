import torch
import torch.nn as nn


class TSPInitialEmbedding(nn.Module):
    """
    Linear projection for TSP nodes.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        node_features_dim: int = 2,
    ) -> None:
        super().__init__()
        assert node_features_dim > 0
        self.node_projection = nn.Linear(node_features_dim, hidden_dim)

    def forward(self, **features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            **features: dictionary containing:
                - `node_features` (B, N, input_dim)

        Returns:
            (B, N, H)
        """
        nodes = features["node_features"]

        return self.node_projection(nodes)
