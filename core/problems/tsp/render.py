from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from core.problems.tsp import TSPInstance

if TYPE_CHECKING:
    from core.problems.tsp.graph_solution import TSPGraphSolution
    from core.problems.tsp.solution import TSPSolution


def render_instance(
    instance: TSPInstance,
    *,
    min_coordinate: float | None = None,
    max_coordinate: float | None = None,
    ax: Axes | None = None,
) -> None:
    if instance.batch_size.numel() > 1:
        raise ValueError("visualize() only works on a single instance.")

    instance = instance.squeeze(0) if instance.batch_size else instance

    if min_coordinate is not None and max_coordinate is not None:
        assert min_coordinate < max_coordinate, "min_coordinate must be less than max_coordinate."

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
    ax.set_title(f"TSP({instance.num_locations})")
    ax.grid(True, linestyle="--", alpha=0.4)


def render_solution(
    solution: TSPSolution,
    *,
    min_coordinate: float | None = None,
    max_coordinate: float | None = None,
    ax: Axes | None = None,
    with_problem_size: bool = True,
    with_objective: bool = True,
) -> None:
    if solution.batch_size.numel() > 1:
        raise ValueError(f"visualize() only works on a single instance. Got {solution.batch_size}")

    unbatched_solution = solution.squeeze(0) if solution.batch_size else solution

    if ax is None:
        ax = plt.gca()

    locations = unbatched_solution.instance.locations.cpu().numpy()
    tour = unbatched_solution.tour.cpu().numpy()

    if min_coordinate is None or max_coordinate is None:
        min_coordinate = locations.min()
        max_coordinate = locations.max()

    cmap = plt.get_cmap("Blues")

    # -- unvisited nodes --
    visited_set = set(tour.tolist())
    unvisited = [
        i for i in range(unbatched_solution.instance.num_locations) if i not in visited_set
    ]
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
        zip(tour, np.roll(tour, -1)) if unbatched_solution.is_complete else zip(tour[:-1], tour[1:])
    )
    for i, j in pairs:
        ax.annotate(
            "",
            xy=tuple(locations[j].tolist()),
            xytext=tuple(locations[i].tolist()),
            arrowprops=dict(arrowstyle="->", color=cmap(0.8), lw=2),
            zorder=1,
        )

    ax.set_xlim(min_coordinate - 0.1, max_coordinate + 0.1)
    ax.set_ylim(min_coordinate - 0.1, max_coordinate + 0.1)
    ax.set_aspect("equal", adjustable="box")

    title = ""
    if with_problem_size:
        title += f"TSP({unbatched_solution.instance.num_locations}) | "

    if with_objective:
        title += f"distance = {unbatched_solution._tour_length().item():.4f}"

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)


def render_graph_solution(
    solution: TSPGraphSolution,
    *,
    min_coordinate: float | None = None,
    max_coordinate: float | None = None,
    ax: Axes | None = None,
    with_problem_size: bool = True,
    with_objective: bool = True,
) -> None:
    if solution.batch_size.numel() > 1:
        raise ValueError(f"render only works on a single instance. Got {solution.batch_size}")

    unbatched = solution.squeeze(0) if solution.batch_size else solution

    if ax is None:
        ax = plt.gca()

    locations = unbatched.instance.locations.cpu().numpy()
    next_nodes = unbatched.next_nodes.cpu().numpy()
    prev_nodes = unbatched.prev_nodes.cpu().numpy()

    if min_coordinate is None or max_coordinate is None:
        min_coordinate = locations.min()
        max_coordinate = locations.max()

    UNASSIGNED = unbatched.UNASSIGNED
    next_unassigned = next_nodes == UNASSIGNED
    prev_unassigned = prev_nodes == UNASSIGNED
    orphaned = next_unassigned & prev_unassigned
    chain_endpoint = (next_unassigned | prev_unassigned) & ~orphaned
    connected = ~next_unassigned & ~prev_unassigned

    cmap = plt.get_cmap("Blues")

    for mask, color in (
        (orphaned, cmap(0.6)),
        (chain_endpoint, cmap(0.6)),
        (connected, cmap(0.6)),
    ):
        if mask.any():
            ax.scatter(
                locations[mask, 0],
                locations[mask, 1],
                s=100,
                color=color,
                edgecolors="black",
                linewidths=1,
                zorder=3,
            )

    for i, j in enumerate(next_nodes):
        if j != UNASSIGNED:
            ax.annotate(
                "",
                xy=tuple(locations[j].tolist()),
                xytext=tuple(locations[i].tolist()),
                arrowprops=dict(arrowstyle="->", color=cmap(0.8), lw=2),
                zorder=1,
            )

    ax.set_xlim(min_coordinate - 0.1, max_coordinate + 0.1)
    ax.set_ylim(min_coordinate - 0.1, max_coordinate + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)

    title_parts = []
    if with_problem_size:
        title_parts.append(f"TSP({unbatched.instance.num_nodes})")
    is_complete = bool(unbatched.is_complete.item())
    if with_objective and is_complete:
        title_parts.append(f"distance = {unbatched.objective().item():.4f}")
    ax.set_title(" | ".join(title_parts))
