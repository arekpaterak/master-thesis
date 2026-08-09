from __future__ import annotations

from typing import Protocol

import torch
from tensordict.tensorclass import TensorClass

from core.constants import EPS
from core.problems.cvrp.instance import CVRPInstance
from core.problems.features import InstanceFeatureExtractor, SolutionFeatureExtractor


class CVRPInstanceFeatures(TensorClass):
    depot_location: torch.Tensor
    """(..., 1, 2)"""
    customer_locations: torch.Tensor
    """(..., N, 2)"""
    customer_demand: torch.Tensor
    """(..., N, 1)"""
    depot_angle: torch.Tensor | None = None
    """(..., N) raw depot-anchored polar angle (XPE). None when PE is disabled."""

    @property
    def customer_features(self) -> torch.Tensor:
        """(..., N, 3) content features [x, y, demand]."""
        return torch.cat([self.customer_locations, self.customer_demand], dim=-1)

    @property
    def depot_features(self) -> torch.Tensor:
        """(..., 1, 2) depot content features."""
        return self.depot_location

    @property
    def locations(self) -> torch.Tensor:
        """(..., N+1, 2) all node coordinates (depot first), matching encoder node order."""
        return torch.cat([self.depot_location, self.customer_locations], dim=-2)


class CVRPSolutionFeatures(TensorClass):
    neighbours_index: torch.Tensor
    """(..., N, 2) [prev, next] node index per customer."""
    route_index: torch.Tensor
    """(..., N) route assignment per customer."""
    route_phase: torch.Tensor | None = None
    """(..., N) raw in-route phase (IPE) in [0, 2*pi]. None when PE is disabled."""


class CVRPSolutionLike(Protocol):
    """Structural contract the CVRP solution extractor relies on.

    Satisfied by both CVRPSolution (sequence) and CVRPGraphSolution. ``route_phase`` is
    only implemented by the graph representation; the sequence one raises.
    """

    @property
    def instance(self) -> CVRPInstance: ...
    def neighbours_per_customer(self) -> torch.Tensor: ...
    def route_per_customer(self) -> torch.Tensor: ...
    def route_phase(self) -> torch.Tensor: ...


class CVRPInstanceFeatureExtractor(InstanceFeatureExtractor[CVRPInstance, CVRPInstanceFeatures]):
    def __init__(self, *, with_positional: bool = False) -> None:
        self._with_positional = with_positional

    def __call__(self, instance: CVRPInstance) -> CVRPInstanceFeatures:
        normalized_customer_demand = instance.normalized_customer_demands.unsqueeze(-1)
        assert bool((normalized_customer_demand <= 1.0 + EPS).all()), (
            "CVRP encoder demand must be capacity-normalised (<= 1); got raw demand."
        )
        return CVRPInstanceFeatures(
            depot_location=instance.depot_location,
            customer_locations=instance.customer_locations,
            customer_demand=normalized_customer_demand,
            depot_angle=instance.depot_anchored_polar_angle() if self._with_positional else None,
            batch_size=instance.batch_size,
        )


class CVRPSolutionFeatureExtractor(
    SolutionFeatureExtractor[CVRPSolutionLike, CVRPSolutionFeatures]
):
    def __init__(self, *, with_positional: bool = False) -> None:
        self._with_positional = with_positional

    def __call__(self, solution: CVRPSolutionLike) -> CVRPSolutionFeatures:
        return CVRPSolutionFeatures(
            neighbours_index=solution.neighbours_per_customer(),
            route_index=solution.route_per_customer(),
            route_phase=solution.route_phase() if self._with_positional else None,
            batch_size=solution.instance.batch_size,
        )
