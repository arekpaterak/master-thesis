from __future__ import annotations

from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tensordict import TensorClass, TensorDict


@dataclass(frozen=True)
class CVRPInstance:
    """Canonical, generator-agnostic representation of a single CVRP instance."""

    locations: torch.Tensor  # shape: (n+1, 2)
    demand: torch.Tensor  # shape: (n+1,) - normalised to [0, 1]
    capacity: float  # vehicle capacity (normalised)

    @property
    def num_customers(self) -> int:
        return self.locations.shape[0] - 1  # excluding depot

    @property
    def depot_location(self) -> torch.Tensor:
        return self.locations[0]

    @property
    def customer_locations(self) -> torch.Tensor:
        return self.locations[1:]

    @property
    def customer_demands(self) -> torch.Tensor:
        return self.demand[1:]

    def __post_init__(self):
        assert self.locations.ndim == 2 and self.locations.shape[1] == 2, (
            "Expected locations to be of shape (num_nodes, 2)."
        )
        assert self.demand.ndim == 1 and self.demand.shape[0] == self.locations.shape[0], (
            "Expected demands to be of shape (num_nodes,) matching locations."
        )
        assert self.depot_location.shape == (2,), "Expected depot to be of shape (2,)."

    @classmethod
    def from_td(cls, td: TensorDict) -> CVRPInstance:
        """Convert from a single (unbatched) RL4CO TensorDict."""
        assert td.batch_dims == 0, "Expected an unbatched TensorDict (single instance)."
        return cls(
            locations=td["nodes"],
            demand=td["demand"],
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

        customer_locations = self.customer_locations.cpu().numpy()
        customer_demands = self.customer_demands.cpu().numpy()
        depot = self.depot_location.cpu().numpy()

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


class CVRPBatchInstance(TensorClass):
    locations: torch.Tensor  # shape: (B, N+1, 2)
    demand: torch.Tensor  # shape: (B, N+1)
    capacity: torch.Tensor  # shape: (B, 1)

    def __post_init__(self):
        b_ndim = len(self.batch_size)

        assert self.locations.ndim == b_ndim + 2, (
            f"Expected nodes to have {b_ndim + 2} dims (batch + N+1 + 2), got {self.locations.ndim}"
        )
        assert self.demand.ndim == b_ndim + 1, (
            f"Expected demand to have {b_ndim + 1} dims (batch + N+1), got {self.demand.ndim}"
        )
        assert self.capacity.ndim == b_ndim + 1, (
            f"Expected capacity to have {b_ndim + 1} dims (batch + 1), got {self.capacity.ndim}"
        )

        assert self.locations.shape[-1] == 2, (
            f"Expected nodes last dim to be 2 (x, y), got {self.locations.shape[-1]}"
        )
        assert self.capacity.shape[-1] == 1, (
            f"Expected capacity last dim to be 1, got {self.capacity.shape[-1]}"
        )
        assert self.locations.shape[-2] == self.demand.shape[-1], (
            "Number of nodes mismatch between nodes and demands."
        )

    @property
    def num_nodes(self) -> int:
        return self.locations.shape[1]

    @property
    def num_customers(self) -> int:
        return self.num_nodes - 1

    @property
    def depot_location(self) -> torch.Tensor:
        return self.locations[:, :1, :]  # (B, 1, 2)

    @property
    def customer_features(self) -> torch.Tensor:
        customer_locations = self.locations[:, 1:, :]
        customer_demands = self.demand[:, 1:].unsqueeze(-1)
        return torch.cat([customer_locations, customer_demands], dim=-1)

    def get_single(self, idx: int) -> CVRPInstance:
        item = self[idx]
        return CVRPInstance(
            locations=item.locations, demand=item.demand, capacity=item.capacity.item()
        )
