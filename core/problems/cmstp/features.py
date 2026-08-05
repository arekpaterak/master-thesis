from __future__ import annotations

from typing import Protocol

import torch
from tensordict.tensorclass import TensorClass

from core.constants import EPS
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.features import InstanceFeatureExtractor, SolutionFeatureExtractor


class CMSTPInstanceFeatures(TensorClass):
    """Instance-derived inputs for a CMSTP encoder. Fields are raw signals."""

    depot_location: torch.Tensor
    """(..., 1, 2)"""
    customer_locations: torch.Tensor
    """(..., N, 2)"""
    customer_demand: torch.Tensor
    """(..., N, 1)"""

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


class CMSTPSolutionFeatures(TensorClass):
    """Solution-derived inputs for a CMSTP encoder. Fields are raw signals.

    ``parent_index`` alone drives both message directions in the tree neighbours layer:
    the parent by a gather, the children by scatter-inverting the same array.
    """

    parent_index: torch.Tensor
    """(..., N) parent node index per customer. 0 = depot, 1..N = customer."""
    route_index: torch.Tensor
    """(..., N) subtree (depot-branch) assignment per customer."""


class CMSTPSolutionLike(Protocol):
    """Structural contract the CMSTP solution extractor relies on."""

    @property
    def instance(self) -> CMSTPInstance: ...
    def parent_per_customer(self) -> torch.Tensor: ...
    def subtree_per_customer(self) -> torch.Tensor: ...


class CMSTPInstanceFeatureExtractor(InstanceFeatureExtractor[CMSTPInstance, CMSTPInstanceFeatures]):
    def __call__(self, instance: CMSTPInstance) -> CMSTPInstanceFeatures:
        normalized_customer_demand = instance.normalized_customer_demands.unsqueeze(-1)
        assert bool((normalized_customer_demand <= 1.0 + EPS).all()), (
            "CMSTP encoder demand must be capacity-normalised (<= 1); got raw demand."
        )
        return CMSTPInstanceFeatures(
            depot_location=instance.depot_location,
            customer_locations=instance.customer_locations,
            customer_demand=normalized_customer_demand,
            batch_size=instance.batch_size,
        )


class CMSTPSolutionFeatureExtractor(
    SolutionFeatureExtractor[CMSTPSolutionLike, CMSTPSolutionFeatures]
):
    def __call__(self, solution: CMSTPSolutionLike) -> CMSTPSolutionFeatures:
        return CMSTPSolutionFeatures(
            parent_index=solution.parent_per_customer(),
            route_index=solution.subtree_per_customer(),
            batch_size=solution.instance.batch_size,
        )
