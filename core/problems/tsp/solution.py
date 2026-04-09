from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.problems.solution import BaseSolution
from core.problems.tsp import TSPInstance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# TODO: Use solution method instead
def tour_length(instances, solutions) -> torch.Tensor:
    """
    Args:
        instances: (batch, n, 2)
        solutions: (batch, n) - permutation of node indices
    Returns:
        lengths: (batch,)
    """
    batch_size, n = solutions.shape
    # gather coordinates in tour order
    ordered = instances[torch.arange(batch_size).unsqueeze(1), solutions]  # (batch, n, 2)
    # distance from each node to the next, wrapping around
    rolled = torch.roll(ordered, -1, dims=1)
    return (ordered - rolled).norm(dim=-1).sum(dim=-1)  # (batch,)


class TSPSolution(BaseSolution):
    """
    Canonical representation of single or batched TSP solution.
    """

    _instance: TSPInstance
    tour: torch.Tensor
    """(..., n), -1 for unvisited slots."""

    PADDING = -1

    @property
    def instance(self) -> TSPInstance:
        return self._instance

    @classmethod
    def create(cls, instance: TSPInstance, tour: torch.Tensor) -> TSPSolution:
        return cls(_instance=instance, tour=tour, batch_size=instance.batch_size)

    def __post_init__(self):
        # Validate that tour indices are within bounds of the instance
        mask = self.tour != self.PADDING
        if mask.any():
            assert self.tour[mask].max() < self.instance.num_locations, "Tour index out of bounds."
            assert self.tour[mask].min() >= 0, "Tour index (excluding -1) must be non-negative."

    @property
    def num_visited(self) -> torch.Tensor:
        return (self.tour != self.PADDING).sum(dim=-1)

    @property
    def is_complete(self) -> torch.Tensor:
        return self.num_visited == self.instance.num_locations

    @property
    def is_feasible(self) -> torch.Tensor:
        return self.is_complete  # All permutations are feasible

    def objective(self) -> torch.Tensor:
        if (self.is_complete == False).any():
            raise ValueError("Cannot calculate objective for incomplete TSP solutions.")
        return self.tour_length()

    def tour_length(self) -> torch.Tensor:
        """
        Calculates tour length for single or batched instances.
        Handles both open paths (partial) and closed loops (complete).
        """
        locs = self.instance.locations

        safe_tour = self.tour.clamp(min=0)

        # (..., N, 2) coordinates in tour order
        ordered_locs = locs.gather(
            dim=-2, index=safe_tour.unsqueeze(-1).expand(*safe_tour.shape, 2)
        )

        # 1. Calculate distances between consecutive nodes in the sequence
        p1 = ordered_locs[..., :-1, :]
        p2 = ordered_locs[..., 1:, :]
        dists = (p1 - p2).norm(dim=-1)

        # Mask out transitions involving PADDING
        mask = (self.tour[..., :-1] != self.PADDING) & (self.tour[..., 1:] != self.PADDING)
        total_dist = (dists * mask.float()).sum(dim=-1)

        # 2. Add the return trip to the start only for complete tours (the Hamiltonian Cycle)
        complete_mask = self.is_complete
        if complete_mask.any():
            start_loc = ordered_locs[..., 0, :]
            end_loc = ordered_locs[..., -1, :]
            return_dist = (end_loc - start_loc).norm(dim=-1)
            total_dist = torch.where(complete_mask, total_dist + return_dist, total_dist)

        return total_dist

    def visualize(
        self,
        *,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
        with_problem_size: bool = True,
        with_objective: bool = True,
    ) -> None:
        from .render import render_solution

        render_solution(
            self,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            ax=ax,
            with_problem_size=with_problem_size,
            with_objective=with_objective,
        )
