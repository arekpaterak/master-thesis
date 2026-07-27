from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from core.problems.tsp import TSPInstance

if TYPE_CHECKING:
    from core.problems.tsp.graph_solution import TSPGraphSolution
    from core.problems.tsp.solution import TSPSolution

    AnyTSPSolution = TSPSolution | TSPGraphSolution


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
    solution: AnyTSPSolution,
    *,
    min_coordinate: float | None = None,
    max_coordinate: float | None = None,
    ax: Axes | None = None,
    with_problem_size: bool = True,
    with_objective: bool = True,
    with_direction: bool = True,
    separator: str = " | ",
    scale: float = 1.0,
) -> None:
    if solution.batch_size.numel() > 1:
        raise ValueError(f"visualize() only works on a single instance. Got {solution.batch_size}")

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
    arrowstyle = "->" if with_direction else "-"

    for mask, color in (
        (orphaned, cmap(0.2)),
        (chain_endpoint, cmap(0.6)),
        (connected, cmap(0.6)),
    ):
        if mask.any():
            ax.scatter(
                locations[mask, 0],
                locations[mask, 1],
                s=100 * scale,
                color=color,
                edgecolors="black",
                linewidths=1.0 * scale,
                zorder=3,
            )

    for i, j in enumerate(next_nodes):
        if j != UNASSIGNED:
            ax.annotate(
                "",
                xy=tuple(locations[j].tolist()),
                xytext=tuple(locations[i].tolist()),
                arrowprops=dict(arrowstyle=arrowstyle, color=cmap(0.8), lw=2 * scale),
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
    ax.set_title(separator.join(title_parts))
