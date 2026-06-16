from __future__ import annotations

import torch
import torch.nn as nn

from core.models.embeddings.positional import circular_sinusoidal_encoding
from core.models.nds.config import PositionalEncodingConfig
from core.problems.tsp.features import TSPInstanceFeatures, TSPSolutionFeatures


class TSPInitialEmbedding(nn.Module):
    """Linear projection for TSP nodes.

    When positional encoding is enabled with ``mode="concat"``, the in-route (IPE)
    sinusoidal encoding is concatenated to the node content features before projection.
    TSP has no depot, so there is no cross-route (XPE) component.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        node_features_dim: int = 2,
        positional: PositionalEncodingConfig | None = None,
    ) -> None:
        super().__init__()
        assert node_features_dim > 0
        self.positional = positional

        extra = positional.dim if self._concat_enabled else 0  # IPE only
        self.node_projection = nn.Linear(node_features_dim + extra, hidden_dim)

    @property
    def _concat_enabled(self) -> bool:
        return self.positional is not None and self.positional.mode == "concat"

    def forward(
        self,
        instance: TSPInstanceFeatures,
        solution: TSPSolutionFeatures | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            (..., N, H)
        """
        nodes = instance.node_features
        if self._concat_enabled:
            nodes = torch.cat([nodes, self._positional_encoding(solution)], dim=-1)
        return self.node_projection(nodes)

    def _positional_encoding(self, solution: TSPSolutionFeatures | None) -> torch.Tensor:
        assert solution is not None and solution.tour_phase is not None, (
            "Positional encoding requires solution tour_phase."
        )
        config = self.positional
        return circular_sinusoidal_encoding(
            solution.tour_phase,
            dim=config.dim,
            direction_aware=config.direction_aware,
            schedule=config.frequency_schedule,
        )
