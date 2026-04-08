from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from core.problems.solution import BaseSolution
from core.problems.tsp import TSPInstance

if TYPE_CHECKING:
    from matplotlib.axes import Axes


# TODO: Adjust to work on better defined representation and handle discontinuous tours (for partial solutions)
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

    instance: TSPInstance
    tour: torch.Tensor
    """(..., n), -1 for unvisited slots."""

    def __post_init__(self):
        # Validate that tour indices are within bounds of the instance
        mask = self.tour != -1
        if mask.any():
            assert self.tour[mask].max() < self.instance.num_locations, "Tour index out of bounds."
            assert self.tour[mask].min() >= 0, "Tour index (excluding -1) must be non-negative."

    @property
    def num_locations(self) -> int:
        return self.instance.num_locations

    @property
    def num_visited(self) -> torch.Tensor:
        return (self.tour != -1).sum(dim=-1)

    @property
    def is_complete(self) -> torch.Tensor:
        return self.num_visited == self.instance.num_locations

    def objective(self) -> torch.Tensor:
        return self.tour_length

    @property
    def tour_length(self) -> torch.Tensor:
        """
        Calculates tour length for single or batched instances.
        Handles both open paths (partial) and closed loops (complete).
        """
        # (..., N, 2)
        locs = self.instance.locations

        # Gather coordinates in tour order
        # We use a dummy index for -1 to avoid errors during gather, then mask later
        safe_tour = self.tour.clamp(min=0)
        # Advanced indexing to handle arbitrary batch dims
        ordered = locs.gather(dim=-2, index=safe_tour.unsqueeze(-1).expand(*safe_tour.shape, 2))

        # Calculate distances between consecutive nodes
        # (..., N-1, 2)
        p1 = ordered[..., :-1, :]
        p2 = ordered[..., 1:, :]
        dists = (p1 - p2).norm(dim=-1)

        # Mask out transitions involving -1
        mask = (self.tour[..., :-1] != -1) & (self.tour[..., 1:] != -1)
        total_dist = (dists * mask.float()).sum(dim=-1)

        # If complete, add the return trip to the start
        # Only add for batch elements that are actually complete
        complete_mask = self.is_complete
        if complete_mask.any():
            start_loc = ordered[..., 0, :]
            end_loc = ordered[..., -1, :]
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
