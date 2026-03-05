from __future__ import annotations

from dataclasses import dataclass

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .instance import CVRPInstance


@dataclass(frozen=True)
class CVRPSolution:
    """A (partial or complete) solution to a CVRP instance."""

    instance: CVRPInstance
    sequence: torch.Tensor  # e.g. [0, 3, 7, 2, 0, 5, 1, 0], starts and ends with 0

    def __post_init__(self):
        for i, route in enumerate(self.routes):
            assert route.ndim == 1, f"Route {i} must be a 1-D tensor."
            assert route.numel() > 0, f"Route {i} is empty."
            assert int(route.min()) >= 1 and int(route.max()) <= self.instance.num_customers, (
                f"Route {i} contains an index outside [1, num_customers]."
            )

    @property
    def routes(self) -> list[torch.Tensor]:
        """
        Split ``sequence`` on depot visits into individual routes.

        Routes store *global* node indices (1..n). Index 0 is the depot and is
        implicit at the start and end of every route.
        """
        depot_positions = (self.sequence == 0).nonzero(as_tuple=True)[0]
        routes = []
        for i in range(len(depot_positions) - 1):
            segment = self.sequence[depot_positions[i] + 1 : depot_positions[i + 1]]
            if segment.numel() > 0:
                routes.append(segment)
        return routes

    @property
    def num_routes(self) -> int:
        return len(self.routes)

    @property
    def all_visited_nodes(self) -> set[int]:
        visited: set[int] = set()
        for route in self.routes:
            visited.update(route.tolist())
        return visited

    @property
    def total_distance(self) -> float:
        pts = self.instance.nodes[self.sequence]
        return (pts[1:] - pts[:-1]).norm(dim=-1).sum().item()

    @property
    def is_feasible(self) -> bool:
        if not self.is_complete:
            return False
        return all(
            self.instance.demands[route].sum().item() <= self.instance.capacity
            for route in self.routes
        )

    @property
    def is_complete(self) -> bool:
        customers = self.sequence[self.sequence != 0]
        return customers.sort().values.equal(torch.arange(1, self.instance.num_customers + 1))

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
        if ax is None:
            ax = plt.gca()

        all_nodes = self.instance.nodes.cpu().numpy()
        customer_locs = self.instance.customer_nodes.cpu().numpy()
        depot = self.instance.depot.cpu().numpy()

        if min_coordinate is None or max_coordinate is None:
            min_coordinate = float(all_nodes.min())
            max_coordinate = float(all_nodes.max())

        cmap = plt.get_cmap("tab10")
        num_colours = max(self.num_routes, 1)

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
        visited = self.all_visited_nodes
        unvisited_mask = [i not in visited for i in range(1, self.instance.num_customers + 1)]
        if any(unvisited_mask):
            unvisited_locs = customer_locs[[i for i, m in enumerate(unvisited_mask) if m]]
            ax.scatter(
                unvisited_locs[:, 0],
                unvisited_locs[:, 1],
                c="lightgrey",
                s=40,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

        for r_idx, route in enumerate(self.routes):
            colour = cmap(r_idx % num_colours)
            indices = torch.cat([torch.tensor([0]), route, torch.tensor([0])])
            waypoints = all_nodes[indices.numpy()]

            route_demand = self.instance.demands[route].sum().item()
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

        title = f"{resolve_routes_num_desc(self.num_routes)}"
        if with_problem_size:
            title = f"CVRP({self.instance.num_customers}) | " + title
        if with_objective:
            title += f" | total distance = {self.total_distance:.4f}"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
