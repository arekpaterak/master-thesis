from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from core.problems.cvrp import CVRPInstance

if TYPE_CHECKING:
    from core.problems.cvrp.solution import CVRPSolution


def render_instance(
    instance: CVRPInstance,
    *,
    min_coordinate: float | None = None,
    max_coordinate: float | None = None,
    ax: Axes | None = None,
) -> None:
    if instance.batch_size.numel() > 1:
        raise ValueError(
            f"visualize() can only be called on a single instance (batch_size=[]). "
            f"Current batch_size is {instance.batch_size}. Try calling .visualize() on a slice, "
            f"e.g., instance[0].visualize()"
        )

    instance = instance.squeeze(0) if instance.batch_size else instance

    if min_coordinate is not None and max_coordinate is not None:
        assert min_coordinate < max_coordinate, "min_coordinate must be less than max_coordinate."

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
    ax.set_title(f"CVRP({instance.num_customers})")
    ax.grid(True, linestyle="--", alpha=0.4)


def render_solution(
    solution: CVRPSolution,
    *,
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
    if solution.batch_size.numel() > 1:
        raise ValueError("visualize() only works on a single instance.")

    unbatched_solution = solution.squeeze(0) if solution.batch_size else solution

    if ax is None:
        ax = plt.gca()

    instance = (
        unbatched_solution.instance.squeeze(0)
        if unbatched_solution.instance.batch_size
        else unbatched_solution.instance
    )

    all_nodes = instance.locations.cpu().numpy()
    depot = instance.depot_location.squeeze(0).cpu().numpy()
    routes = unbatched_solution.routes()

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

    num_colours = max(len(routes), 1)
    for r_idx, route in enumerate(routes):
        colour = cmap(r_idx % num_colours)
        indices = torch.cat([torch.tensor([0]), route.cpu(), torch.tensor([0])])
        waypoints = all_nodes[indices.numpy()]

        route_demand = instance.demand[route].sum().item()
        route_dist = (
            (torch.tensor(waypoints[1:]) - torch.tensor(waypoints[:-1])).norm(dim=-1).sum().item()
        )
        route_label = f"R{r_idx + 1}, # {route.numel()}, c={route_demand:.2f}, d={route_dist:.4f}"

        # Nodes coloured by route
        route_locs = all_nodes[route.cpu().numpy()]
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

    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc=legend_loc)

    def resolve_routes_num_desc(n: int) -> str:
        return "no routes" if n == 0 else f"{n} route{'s' if n > 1 else ''}"

    title = f"{resolve_routes_num_desc(len(routes))}"
    if with_problem_size:
        title = f"CVRP({instance.num_customers}) | " + title
    if with_objective:
        title += f" | total distance = {unbatched_solution.total_distance().item():.4f}"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
