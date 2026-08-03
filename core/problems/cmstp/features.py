from __future__ import annotations

from typing import Protocol

import torch
from tensordict.tensorclass import TensorClass

from core.constants import EPS
from core.problems.cmstp.instance import CMSTPInstance
from core.problems.features import InstanceFeatureExtractor, SolutionFeatureExtractor


class CMSTPInstanceFeatures(TensorClass):
    """Instance-derived inputs for a CMSTP encoder. Fields are raw signals."""

    root_location: torch.Tensor
    """(..., 1, 2)"""
    terminal_locations: torch.Tensor
    """(..., N, 2)"""
    terminal_demand: torch.Tensor
    """(..., N, 1)"""

    @property
    def terminal_features(self) -> torch.Tensor:
        """(..., N, 3) content features [x, y, demand]."""
        return torch.cat([self.terminal_locations, self.terminal_demand], dim=-1)

    @property
    def root_features(self) -> torch.Tensor:
        """(..., 1, 2) root content features."""
        return self.root_location

    @property
    def locations(self) -> torch.Tensor:
        """(..., N+1, 2) all node coordinates (root first), matching encoder node order."""
        return torch.cat([self.root_location, self.terminal_locations], dim=-2)


class CMSTPSolutionFeatures(TensorClass):
    """Solution-derived inputs for a CMSTP encoder. Fields are raw signals.

    ``parent_index`` alone drives both message directions in the tree neighbours layer:
    the parent by a gather, the children by scatter-inverting the same array.
    """

    parent_index: torch.Tensor
    """(..., N) parent node index per terminal. 0 = root, 1..N = terminal."""
    route_index: torch.Tensor
    """(..., N) subtree (root-branch) assignment per terminal."""


class CMSTPSolutionLike(Protocol):
    """Structural contract the CMSTP solution extractor relies on."""

    @property
    def instance(self) -> CMSTPInstance: ...
    def parent_per_terminal(self) -> torch.Tensor: ...
    def subtree_per_terminal(self) -> torch.Tensor: ...


class CMSTPInstanceFeatureExtractor(InstanceFeatureExtractor[CMSTPInstance, CMSTPInstanceFeatures]):
    def __call__(self, instance: CMSTPInstance) -> CMSTPInstanceFeatures:
        normalized_terminal_demand = instance.normalized_terminal_demands.unsqueeze(-1)
        assert bool((normalized_terminal_demand <= 1.0 + EPS).all()), (
            "CMSTP encoder demand must be capacity-normalised (<= 1); got raw demand."
        )
        return CMSTPInstanceFeatures(
            root_location=instance.root_location,
            terminal_locations=instance.terminal_locations,
            terminal_demand=normalized_terminal_demand,
            batch_size=instance.batch_size,
        )


class CMSTPSolutionFeatureExtractor(
    SolutionFeatureExtractor[CMSTPSolutionLike, CMSTPSolutionFeatures]
):
    def __call__(self, solution: CMSTPSolutionLike) -> CMSTPSolutionFeatures:
        return CMSTPSolutionFeatures(
            parent_index=solution.parent_per_terminal(),
            route_index=solution.subtree_per_terminal(),
            batch_size=solution.instance.batch_size,
        )
