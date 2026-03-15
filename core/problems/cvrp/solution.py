from __future__ import annotations

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tensordict.tensorclass import TensorClass

from .instance import CVRPInstance


class CVRPSolution(TensorClass):
    """Canonical representation of single or batched CVRP solutions."""

    instance: CVRPInstance
    sequence: torch.Tensor
    """(..., L) where L is the max sequence length in the batch. Starts and ends with 0 (depot). Padded with 0s if necessary."""

    def __post_init__(self):
        # Validate that sequence indices are within bounds
        if self.sequence.numel() > 0:
            assert (self.sequence >= 0).all(), "Node indices must be non-negative."
            assert (self.sequence <= self.instance.num_customers).all(), "Node index out of bounds."

        assert (self.sequence[..., 0] == 0).all() and (self.sequence[..., -1] == 0).all(), (
            "Sequence must start and end at depot (0)."
        )

    @property
    def total_distance(self) -> torch.Tensor:
        """
        Calculates total distance for single or batched instances.
        Computes the norm between consecutive nodes in the sequence.
        """
        # (..., L, 2) - Gather locations based on sequence indices
        # We need to expand sequence to match the (x, y) dimension
        idx = self.sequence.unsqueeze(-1).expand(*self.sequence.shape, 2)
        pts = self.instance.locations.gather(dim=-2, index=idx)

        # Distance between consecutive points in the sequence
        dists = (pts[..., 1:, :] - pts[..., :-1, :]).norm(dim=-1)
        return dists.sum(dim=-1)

    @property
    def is_complete(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def is_feasible(self) -> torch.Tensor:
        raise NotImplementedError()

    def routes(self) -> list[torch.Tensor]:
        """Splits the single-instance sequence into a list of routes (excluding depot)."""
        if self.batch_size.numel() > 1:
            raise ValueError("routes() only works on a single instance.")

        seq = self.sequence.cpu()
        depot_indices = (seq == 0).nonzero(as_tuple=True)[0]
        routes = []
        for start, end in zip(depot_indices[:-1], depot_indices[1:]):
            route = seq[start + 1 : end]
            if route.numel() > 0:
                routes.append(route)
        return routes

    def visualize(
        self,
        min_coordinate: float | None = None,
        max_coordinate: float | None = None,
        ax: Axes | None = None,
        with_problem_size: bool = True,
        with_objective: bool = True,
        show_depot_edges: bool = False,
        depot_in_legend: bool = True,
        with_route_labels: bool = True,
        legend_loc: str | None = "upper right",
    ) -> None:
        if self.batch_size.numel() > 1:
            raise ValueError("visualize() only works on a single instance.")

        if ax is None:
            ax = plt.gca()

        instance = self.instance.squeeze(0) if self.batch_size else self.instance

        all_nodes = instance.locations.cpu().numpy()
        # customer_locs = instance.customer_locations.cpu().numpy()
        depot = instance.depot_location.squeeze(0).cpu().numpy()
        routes = self.routes()

        if min_coordinate is None or max_coordinate is None:
            min_coordinate = float(all_nodes.min())
            max_coordinate = float(all_nodes.max())

        cmap = plt.get_cmap("tab10")

        # Plot depot
        ax.scatter(
            depot[0],
            depot[1],
            c="orange",
            s=120,
            marker="s",
            edgecolors="black",
            linewidths=1,
            label="depot" if depot_in_legend else None,
            zorder=5,
        )

        # Unvisited customers in grey
        # visited = self.all_visited_nodes
        # unvisited_mask = [i not in visited for i in range(1, self.instance.num_customers + 1)]
        # if any(unvisited_mask):
        #     unvisited_locs = customer_locs[[i for i, m in enumerate(unvisited_mask) if m]]
        #     ax.scatter(
        #         unvisited_locs[:, 0],
        #         unvisited_locs[:, 1],
        #         c="lightgrey",
        #         s=40,
        #         edgecolors="black",
        #         linewidths=0.5,
        #         zorder=3,
        #     )

        num_colours = max(len(routes), 1)
        for r_idx, route in enumerate(routes):
            colour = cmap(r_idx % num_colours)
            indices = torch.cat([torch.tensor([0]), route, torch.tensor([0])])
            waypoints = all_nodes[indices.numpy()]

            route_demand = self.instance.demand[route].sum().item()
            route_dist = (
                (torch.tensor(waypoints[1:]) - torch.tensor(waypoints[:-1]))
                .norm(dim=-1)
                .sum()
                .item()
            )
            route_label = (
                f"R{r_idx + 1}, # {route.numel()}, c={route_demand:.2f}, d={route_dist:.4f}"
            )

            # Nodes coloured by route
            route_locs = all_nodes[route.numpy()]
            ax.scatter(
                route_locs[:, 0],
                route_locs[:, 1],
                c=[colour],
                s=40,
                edgecolors="black",
                linewidths=0.5,
                label=route_label if with_route_labels else None,
                zorder=4,
            )

            # Edges
            for k in range(len(waypoints) - 1):
                is_depot_edge = k == 0 or k == len(waypoints) - 2
                if is_depot_edge and not show_depot_edges:
                    continue
                ax.annotate(
                    "",
                    xy=waypoints[k + 1],
                    xytext=waypoints[k],
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.5),
                    zorder=2,
                )

        ax.set_xlim(min_coordinate - 0.1, max_coordinate + 0.1)
        ax.set_ylim(min_coordinate - 0.1, max_coordinate + 0.1)
        ax.set_aspect("equal", adjustable="box")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc)

        def resolve_routes_num_desc(n: int) -> str:
            return "no routes" if n == 0 else f"{n} route{'s' if n > 1 else ''}"

        title = f"{resolve_routes_num_desc(len(routes))}"
        if with_problem_size:
            title = f"CVRP({self.instance.num_customers}) | " + title
        if with_objective:
            title += f" | total distance = {self.total_distance.item():.4f}"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
