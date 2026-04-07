from __future__ import annotations

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from core.problems.instance import BaseProblemInstance


class TSPInstance(BaseProblemInstance):
    """Canonical, generator-agnostic representation of a single or batched TSP instance."""

    locations: torch.Tensor
    """(B, N, 2) or (N, 2)"""

    def __post_init__(self):
        assert self.locations.shape[-1] == 2, (
            "Expected locations to have shape (batch_size, num_locations, 2) or (num_locations, 2)."
        )

    @property
    def num_locations(self) -> int:
        return self.locations.shape[-2]

    @property
    def num_nodes(self) -> int:
        return self.num_locations

    def visualize(
        self,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
    ) -> None:
        if self.batch_size.numel() > 1:
            raise ValueError("visualize() only works on a single instance.")

        instance = self.squeeze(0) if self.batch_size else self

        if min_coordinate is not None and max_coordinate is not None:
            assert min_coordinate < max_coordinate, (
                "min_coordinate must be less than max_coordinate."
            )

        if ax is None:
            ax = plt.gca()

        locations = instance.locations.cpu().numpy()

        if min_coordinate is None or max_coordinate is None:
            min_coordinate = locations.min()
            max_coordinate = locations.max()

        ax.scatter(
            locations[:, 0],
            locations[:, 1],
            s=100,
            color=plt.get_cmap("Blues")(0.5),
            edgecolors="black",
            linewidths=1,
        )

        ax.set_xlim(min_coordinate - 0.1, max_coordinate + 0.1)
        ax.set_ylim(min_coordinate - 0.1, max_coordinate + 0.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"TSP({self.num_locations})")
        ax.grid(True, linestyle="--", alpha=0.4)
