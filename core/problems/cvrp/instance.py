from __future__ import annotations

from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tensordict import TensorDict


@dataclass(frozen=True)
class CVRPInstance:
    """Canonical, generator-agnostic representation of a single CVRP instance."""

    nodes: torch.Tensor  # shape: (n+1, 2)
    demands: torch.Tensor  # shape: (n+1,) - normalised to [0, 1]
    capacity: float  # vehicle capacity (normalised)

    @property
    def depot(self) -> torch.Tensor:
        return self.nodes[0]

    @property
    def customer_nodes(self) -> torch.Tensor:
        return self.nodes[1:]

    @property
    def customer_demands(self) -> torch.Tensor:
        return self.demands[1:]

    def __post_init__(self):
        assert self.nodes.ndim == 2 and self.nodes.shape[1] == 2, (
            "Expected locations to be of shape (num_nodes, 2)."
        )
        assert self.demands.ndim == 1 and self.demands.shape[0] == self.nodes.shape[0], (
            "Expected demands to be of shape (num_nodes,) matching locations."
        )
        assert self.depot.shape == (2,), "Expected depot to be of shape (2,)."

    @property
    def num_customers(self) -> int:
        return self.nodes.shape[0] - 1  # excluding depot

    @classmethod
    def from_td(cls, td: TensorDict) -> CVRPInstance:
        """Convert from a single (unbatched) RL4CO TensorDict."""
        assert td.batch_dims == 0, "Expected an unbatched TensorDict (single instance)."
        return cls(
            nodes=td["nodes"],
            demands=td["demand"],
            capacity=td["capacity"].item(),
        )

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

        customer_locations = self.customer_nodes.cpu().numpy()
        customer_demands = self.customer_demands.cpu().numpy()
        depot = self.depot.cpu().numpy()

        if min_coordinate is None or max_coordinate is None:
            all_coords = customer_locations
            min_coordinate = float(all_coords.min())
            max_coordinate = float(all_coords.max())

        scatter = ax.scatter(
            customer_locations[:, 0],
            customer_locations[:, 1],
            c=customer_demands,
            cmap="Reds",
            s=100,
            vmin=0,
            vmax=customer_demands.max(),
            edgecolors="black",
            linewidths=1,
            zorder=2,
        )

        ax.scatter(
            depot[0],
            depot[1],
            c="orange",
            s=150,
            marker="s",
            label="depot",
            edgecolors="black",
            linewidths=1,
            zorder=3,
        )

        plt.colorbar(scatter, ax=ax, label="Demand (normalised)")
        ax.set_xlim(min_coordinate - 0.1, max_coordinate + 0.1)
        ax.set_ylim(min_coordinate - 0.1, max_coordinate + 0.1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right")
        ax.set_title(f"CVRP({self.num_customers})")
        ax.grid(True, linestyle="--", alpha=0.4)
