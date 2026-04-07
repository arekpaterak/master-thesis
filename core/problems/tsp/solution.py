from __future__ import annotations

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from core.problems.solution import BaseSolution
from core.problems.tsp import TSPInstance


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
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
        with_problem_size: bool = True,
        with_objective: bool = True,
    ) -> None:
        if self.batch_size.numel() > 1:
            raise ValueError(f"visualize() only works on a single instance. Got {self.batch_size}")

        unbatched_self = self.squeeze(0) if self.batch_size else self

        if ax is None:
            ax = plt.gca()

        locations = unbatched_self.instance.locations.cpu().numpy()
        tour = unbatched_self.tour.cpu().numpy()

        if min_coordinate is None or max_coordinate is None:
            min_coordinate = locations.min()
            max_coordinate = locations.max()

        cmap = plt.get_cmap("Blues")

        # -- unvisited nodes --
        visited_set = set(tour.tolist())
        unvisited = [i for i in range(self.instance.num_locations) if i not in visited_set]
        if unvisited:
            ax.scatter(
                locations[unvisited, 0],
                locations[unvisited, 1],
                s=100,
                color=cmap(0.2),
                edgecolors="black",
                linewidths=1,
                zorder=2,
            )

        # -- visited nodes --
        ax.scatter(
            locations[tour, 0],
            locations[tour, 1],
            s=100,
            color=cmap(0.6),
            edgecolors="black",
            linewidths=1,
            zorder=3,
        )

        # -- tour edges --
        pairs = (
            zip(tour, np.roll(tour, -1))  # close the loop if complete
            if self.is_complete
            else zip(tour[:-1], tour[1:])
        )
        for i, j in pairs:
            ax.annotate(
                "",
                xy=locations[j],
                xytext=locations[i],
                arrowprops=dict(arrowstyle="->", color=cmap(0.8), lw=2),
                zorder=1,
            )

        ax.set_xlim(min_coordinate - 0.1, max_coordinate + 0.1)
        ax.set_ylim(min_coordinate - 0.1, max_coordinate + 0.1)
        ax.set_aspect("equal", adjustable="box")

        title = ""
        if with_problem_size:
            title += f"TSP({self.instance.num_locations}) | "

        if with_objective:
            title += f"distance = {self.tour_length.item():.4f}"

        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
