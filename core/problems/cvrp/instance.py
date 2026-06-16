from __future__ import annotations

import torch
from matplotlib.axes import Axes

from core.problems.augmentations import apply_d4_transforms
from core.problems.instance import BaseProblemInstance


# TODO: Enable viewing also unnormalized demand and capacity
class CVRPInstance(BaseProblemInstance):
    """Canonical, generator-agnostic representation of a single or batched CVRP instance."""

    locations: torch.Tensor
    """(B, N+1, 2) or (N+1, 2)"""
    demand: torch.Tensor
    """(B, N+1) or (N+1,)"""
    capacity: torch.Tensor
    """(B, 1) or (1,)"""

    DEPOT = 0

    @classmethod
    def create(
        cls, locations: torch.Tensor, demand: torch.Tensor, capacity: torch.Tensor
    ) -> CVRPInstance:
        batch_size = locations.shape[0] if locations.dim() == 3 else []
        return cls(
            locations=locations,
            demand=demand,
            capacity=capacity,
            batch_size=batch_size,
            device=locations.device,
        )

    def __post_init__(self):
        b_ndim = len(self.batch_size)

        assert self.locations.ndim == b_ndim + 2, (
            f"Expected nodes to have {b_ndim + 2} dims (batch + N+1 + 2), got {self.locations.ndim}"
        )
        assert self.demand.ndim == b_ndim + 1, (
            f"Expected demand to have {b_ndim + 1} dims (batch + N+1), got {self.demand.ndim}"
        )
        assert self.capacity.ndim == b_ndim + 1, (
            f"Expected capacity to have {b_ndim + 1} dims (batch + 1), got {self.capacity.ndim}"
        )

        assert self.locations.shape[-1] == 2, "Expected nodes last dim to be 2 (x, y)."
        assert self.capacity.shape[-1] == 1, "Expected capacity last dim to be 1."
        assert self.locations.shape[-2] == self.demand.shape[-1], (
            "Number of nodes mismatch between nodes and demands."
        )

    @property
    def num_nodes(self) -> int:
        return self.locations.shape[-2]

    @property
    def num_customers(self) -> int:
        return self.num_nodes - 1

    @property
    def depot_location(self) -> torch.Tensor:
        return self.locations[..., :1, :]  # (..., 1, 2)

    @property
    def customer_locations(self) -> torch.Tensor:
        return self.locations[..., 1:, :]

    @property
    def customer_demands(self) -> torch.Tensor:
        return self.demand[..., 1:]

    def distances(self) -> torch.Tensor:
        """(..., N+1, N+1) pairwise Euclidean distances between all nodes (depot + customers)."""
        return torch.cdist(self.locations, self.locations)

    def augment(self, n_augmentations: int) -> CVRPInstance:
        augmented_locations = apply_d4_transforms(self.locations, n_augmentations)
        expanded_demand = self.demand.unsqueeze(-2).expand(
            *self.batch_size, n_augmentations, self.demand.shape[-1]
        )
        expanded_capacity = self.capacity.unsqueeze(-2).expand(
            *self.batch_size, n_augmentations, self.capacity.shape[-1]
        )
        return CVRPInstance(
            locations=augmented_locations,
            demand=expanded_demand.clone(),
            capacity=expanded_capacity.clone(),
            batch_size=(*self.batch_size, n_augmentations),
            device=augmented_locations.device,
        )

    @property
    def customer_features(self) -> torch.Tensor:
        """
        (B, N, 3) or (N, 3)
        """
        customer_locations = self.locations[..., 1:, :]
        customer_demands = self.demand[..., 1:].unsqueeze(-1)
        return torch.cat([customer_locations, customer_demands], dim=-1)

    def visualize(
        self,
        *,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
    ) -> None:
        from .render import render_instance

        render_instance(self, min_coordinate=min_coordinate, max_coordinate=max_coordinate, ax=ax)
