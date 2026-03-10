from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from core.problems.tsp import TSPInstance


@dataclass(frozen=True)
class TSPSolution:
    """A (partial or complete) solution to a TSP instance."""

    instance: TSPInstance
    tour: torch.Tensor  # indices into instance.locations

    def __post_init__(self):
        assert self.tour.ndim == 1, "Expected tour to be a 1D tensor of node indices."
        assert self.tour.max() < self.instance.num_locations, (
            "Tour contains node index out of bounds."
        )

    @property
    def num_visited(self) -> int:
        return self.tour.shape[0]

    @property
    def is_complete(self) -> bool:
        return self.num_visited == self.instance.num_locations

    @property
    def tour_length(self) -> float:
        ordered = self.instance.locations[self.tour]  # (num_visited, 2)
        if self.is_complete:
            rolled = torch.roll(ordered, -1, dims=0)
        else:
            rolled = ordered[1:]
            ordered = ordered[:-1]
        return (ordered - rolled).norm(dim=-1).sum().item()

    def visualize(
        self,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
        with_problem_size: bool = True,
        with_objective: bool = True,
    ) -> None:
        if ax is None:
            ax = plt.gca()

        locations = self.instance.locations.cpu().numpy()
        tour = self.tour.cpu().numpy()

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
            title += f"distance = {self.tour_length:.4f}"

        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)


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


@dataclass(frozen=True)
class TSPBatchSolution:
    """
    Batched solution representation using tensors throughout.

    tour holds full (batch, n) tensor, with -1 for unvisited positions.
    """

    locations: torch.Tensor  # (batch, n, 2)
    tour: torch.Tensor  # (batch, n), -1 for unvisited slots

    @property
    def batch_size(self) -> int:
        return self.locations.shape[0]

    @property
    def n(self) -> int:
        return self.locations.shape[1]

    @property
    def tour_lengths(self) -> torch.Tensor:
        return tour_length(self.locations, self.tour)  # (batch,)

    def to_list(self) -> list[TSPSolution]:
        return [
            TSPSolution(
                instance=TSPInstance(self.locations[b_idx]),
                tour=self.tour[b_idx],
            )
            for b_idx in range(self.batch_size)
        ]
