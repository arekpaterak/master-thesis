from __future__ import annotations

import torch
import torch.nn as nn

from core.models.embeddings.positional import circular_sinusoidal_encoding
from core.models.nds.config import PositionalEncodingConfig
from core.problems.cvrp.features import CVRPInstanceFeatures, CVRPSolutionFeatures


class CVRPInitialEmbedding(nn.Module):
    """Separate linear projections for depot and customer nodes.

    When a positional encoding config is given, the in-route (IPE) and cross-route
    (XPE) sinusoidal encodings are concatenated to the customer content features before
    the customer projection. The depot is the XPE anchor and keeps its own PE-free
    projection.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        depot_features_dim: int = 2,
        customer_features_dim: int = 3,
        positional: PositionalEncodingConfig | None = None,
    ) -> None:
        super().__init__()
        assert depot_features_dim > 0
        assert customer_features_dim > 0
        self.positional = positional

        extra = 2 * positional.dim if self._inject_at_input else 0  # IPE + XPE
        self.depot_projection = nn.Linear(depot_features_dim, hidden_dim)
        self.customer_projection = nn.Linear(customer_features_dim + extra, hidden_dim)

    @property
    def _inject_at_input(self) -> bool:
        return self.positional is not None

    def forward(
        self,
        instance: CVRPInstanceFeatures,
        solution: CVRPSolutionFeatures | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            (..., N+1, H)
        """
        customer = instance.customer_features
        if self._inject_at_input:
            customer = torch.cat([customer, self._positional_encoding(instance, solution)], dim=-1)

        depot_embedding = self.depot_projection(instance.depot_features)
        customer_embeddings = self.customer_projection(customer)
        return torch.cat([depot_embedding, customer_embeddings], dim=-2)

    def _positional_encoding(
        self,
        instance: CVRPInstanceFeatures,
        solution: CVRPSolutionFeatures | None,
    ) -> torch.Tensor:
        assert solution is not None and solution.route_phase is not None, (
            "Positional encoding requires solution route_phase."
        )
        assert instance.depot_angle is not None, "Positional encoding requires depot_angle."
        config = self.positional
        ipe = circular_sinusoidal_encoding(
            solution.route_phase,
            dim=config.dim,
            direction_aware=config.direction_aware,
            schedule=config.frequency_schedule,
        )
        xpe = circular_sinusoidal_encoding(
            instance.depot_angle,
            dim=config.dim,
            direction_aware=config.direction_aware,
            schedule=config.frequency_schedule,
        )
        return torch.cat([ipe, xpe], dim=-1)
