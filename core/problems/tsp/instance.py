from __future__ import annotations

from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tensordict import TensorDict


@dataclass(frozen=True)
class TSPInstance:
    """Canonical, generator-agnostic representation of a single TSP instance."""

    locations: torch.Tensor  # (n, 2)

    def __post_init__(self):
        assert self.locations.ndim == 2 and self.locations.shape[1] == 2, (
            "Expected locations to be of shape (num_locations, 2)."
        )

    @property
    def num_locations(self) -> int:
        return self.locations.shape[0]

    @classmethod
    def from_tensor(cls, locations: torch.Tensor) -> TSPInstance:
        """Convert from a raw tensor of locations."""
        assert locations.ndim == 2 and locations.shape[1] == 2, (
            "Expected locations to be of shape (num_locations, 2)."
        )
        return cls(locations=locations)

    def to_tensor(self) -> torch.Tensor:
        """Convert to a raw tensor of locations."""
        return self.locations

    @classmethod
    def from_rl4co(cls, td: TensorDict) -> TSPInstance:
        """Convert from a single (unbatched) RL4CO TensorDict."""
        assert td.batch_dims == 0, "Expected an unbatched TensorDict (single instance)."
        return cls(locations=td["locs"])

    def to_rl4co(self) -> TensorDict:
        """Convert to the RL4CO TensorDict format."""
        return TensorDict({"locs": self.locations}, batch_size=[])

    def visualize(
        self,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
    ) -> None:
        if min_coordinate is not None and max_coordinate is not None:
            assert min_coordinate < max_coordinate, (
                "min_coordinate must be less than max_coordinate."
            )

        if ax is None:
            ax = plt.gca()

        locations = self.locations.cpu().numpy()

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


@dataclass(frozen=True)
class TSPBatchInstance:
    locations: torch.Tensor  # (batch, n, 2)

    @property
    def batch_size(self) -> int:
        return self.locations.shape[0]

    @property
    def n(self) -> int:
        return self.locations.shape[1]

    def to_list(self) -> list[TSPInstance]:
        return [
            TSPInstance(
                self.locations[b_idx],
            )
            for b_idx in range(self.batch_size)
        ]
