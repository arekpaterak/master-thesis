from __future__ import annotations

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tensordict import TensorClass


class CVRPInstance(TensorClass):
    """Canonical, generator-agnostic representation of a single or batched CVRP instance."""

    locations: torch.Tensor
    """(B, N+1, 2) or (N+1, 2)"""
    demand: torch.Tensor
    """(B, N+1) or (N+1,)"""
    capacity: torch.Tensor
    """(B, 1) or (1,)"""

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

        assert self.locations.shape[-1] == 2, "Expected nodes last dim to be 2 (x, y)."
        assert self.capacity.shape[-1] == 1, "Expected capacity last dim to be 1."
        assert self.locations.shape[-2] == self.demand.shape[-1], (
            "Number of nodes mismatch between nodes and demands."
        )

    @property
    def num_nodes(self) -> int:
        return self.locations.shape[-2]

    @property
    def num_customers(self) -> int:
        return self.num_nodes - 1

    @property
    def depot_location(self) -> torch.Tensor:
        return self.locations[..., :1, :]  # (..., 1, 2)

    @property
    def customer_locations(self) -> torch.Tensor:
        return self.locations[..., 1:, :]

    @property
    def customer_demands(self) -> torch.Tensor:
        return self.demand[..., 1:]

    @property
    def customer_features(self) -> torch.Tensor:
        customer_locations = self.locations[..., 1:, :]
        customer_demands = self.demand[..., 1:].unsqueeze(-1)
        return torch.cat([customer_locations, customer_demands], dim=-1)

    def visualize(
        self,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
    ) -> None:
        if self.batch_size.numel() > 1:
            raise ValueError(
                f"visualize() can only be called on a single instance (batch_size=[]). "
                f"Current batch_size is {self.batch_size}. Try calling .visualize() on a slice, "
                f"e.g., instance[0].visualize()"
            )

        instance = self.squeeze(0) if self.batch_size else self

        if min_coordinate is not None and max_coordinate is not None:
            assert min_coordinate < max_coordinate, (
                "min_coordinate must be less than max_coordinate."
            )

        if ax is None:
            ax = plt.gca()

        customer_locations = instance.customer_locations.cpu().numpy()
        customer_demands = instance.customer_demands.cpu().numpy()
        depot = instance.depot_location.cpu().numpy()

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
            depot[..., 0],
            depot[..., 1],
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
