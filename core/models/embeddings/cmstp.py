from __future__ import annotations

import torch
import torch.nn as nn

from core.problems.cmstp.features import CMSTPInstanceFeatures, CMSTPSolutionFeatures


class CMSTPInitialEmbedding(nn.Module):
    """Separate linear projections for the depot and customer nodes.

    The CMSTP analog of ``CVRPInitialEmbedding`` without the positional path: the first
    CMSTP runs use content + structural (neighbours/route) signal only, so no in-tree or
    cross-tree positional encoding is injected here.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        depot_features_dim: int = 2,
        customer_features_dim: int = 3,
        positional: object | None = None,
    ) -> None:
        super().__init__()
        assert depot_features_dim > 0
        assert customer_features_dim > 0
        if positional is not None:
            raise NotImplementedError("CMSTP positional encoding is not implemented yet.")
        self.depot_projection = nn.Linear(depot_features_dim, hidden_dim)
        self.customer_projection = nn.Linear(customer_features_dim, hidden_dim)

    def forward(
        self,
        instance: CMSTPInstanceFeatures,
        solution: CMSTPSolutionFeatures | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            (..., N+1, H)
        """
        depot_embedding = self.depot_projection(instance.depot_features)
        customer_embeddings = self.customer_projection(instance.customer_features)
        return torch.cat([depot_embedding, customer_embeddings], dim=-2)
