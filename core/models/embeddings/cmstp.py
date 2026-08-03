from __future__ import annotations

import torch
import torch.nn as nn

from core.problems.cmstp.features import CMSTPInstanceFeatures, CMSTPSolutionFeatures


class CMSTPInitialEmbedding(nn.Module):
    """Separate linear projections for the root and terminal nodes.

    The CMSTP analog of ``CVRPInitialEmbedding`` without the positional path: the first
    CMSTP runs use content + structural (neighbours/route) signal only, so no in-tree or
    cross-tree positional encoding is injected here.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        root_features_dim: int = 2,
        terminal_features_dim: int = 3,
        positional: object | None = None,
    ) -> None:
        super().__init__()
        assert root_features_dim > 0
        assert terminal_features_dim > 0
        if positional is not None:
            raise NotImplementedError("CMSTP positional encoding is not implemented yet.")
        self.root_projection = nn.Linear(root_features_dim, hidden_dim)
        self.terminal_projection = nn.Linear(terminal_features_dim, hidden_dim)

    def forward(
        self,
        instance: CMSTPInstanceFeatures,
        solution: CMSTPSolutionFeatures | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            (..., N+1, H)
        """
        root_embedding = self.root_projection(instance.root_features)
        terminal_embeddings = self.terminal_projection(instance.terminal_features)
        return torch.cat([root_embedding, terminal_embeddings], dim=-2)
