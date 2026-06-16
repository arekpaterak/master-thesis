from __future__ import annotations

from typing import Protocol

import torch
from tensordict.tensorclass import TensorClass

from core.problems.features import InstanceFeatureExtractor, SolutionFeatureExtractor
from core.problems.tsp.instance import TSPInstance


class TSPInstanceFeatures(TensorClass):
    """Instance-derived inputs for a TSP encoder. Fields are raw signals."""

    node_locations: torch.Tensor
    """(..., N, 2)"""

    @property
    def node_features(self) -> torch.Tensor:
        """(..., N, 2) content features."""
        return self.node_locations

    @property
    def locations(self) -> torch.Tensor:
        """(..., N, 2) all node coordinates, matching encoder node order."""
        return self.node_locations


class TSPSolutionFeatures(TensorClass):
    """Solution-derived inputs for a TSP encoder. Fields are raw signals."""

    neighbours_index: torch.Tensor
    """(..., N, 2) [prev, next] node index per node."""
    tour_phase: torch.Tensor | None = None
    """(..., N) raw in-tour phase (IPE) in [0, 2*pi]. None when PE is disabled."""


class TSPSolutionLike(Protocol):
    """Structural contract the TSP solution extractor relies on.

    Satisfied by both TSPSolution (sequence) and TSPGraphSolution. ``tour_phase`` is only
    implemented by the graph representation; the sequence one raises.
    """

    @property
    def instance(self) -> TSPInstance: ...
    def neighbours_index(self) -> torch.Tensor: ...
    def tour_phase(self) -> torch.Tensor: ...


class TSPInstanceFeatureExtractor(InstanceFeatureExtractor[TSPInstance, TSPInstanceFeatures]):
    def __call__(self, instance: TSPInstance) -> TSPInstanceFeatures:
        return TSPInstanceFeatures(
            node_locations=instance.locations,
            batch_size=instance.batch_size,
        )


class TSPSolutionFeatureExtractor(SolutionFeatureExtractor[TSPSolutionLike, TSPSolutionFeatures]):
    def __init__(self, *, with_positional: bool = False) -> None:
        self._with_positional = with_positional

    def __call__(self, solution: TSPSolutionLike) -> TSPSolutionFeatures:
        return TSPSolutionFeatures(
            neighbours_index=solution.neighbours_index(),
            tour_phase=solution.tour_phase() if self._with_positional else None,
            batch_size=solution.instance.batch_size,
        )
