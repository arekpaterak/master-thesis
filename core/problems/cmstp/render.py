from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from core.problems.cmstp import CMSTPInstance
from core.problems.palette import resolve_palette

if TYPE_CHECKING:
    from core.problems.cmstp.solution import CMSTPSolution


def render_instance(
    instance: CMSTPInstance,
    *,
    min_coordinate: float | None = None,
    max_coordinate: float | None = None,
    ax: Axes | None = None,
    with_normalized_demand: bool = True,
    depot_label: str = "depot",
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
    customer_demands = (
        (
            instance.normalized_customer_demands
            if with_normalized_demand
            else instance.customer_demands
        )
        .cpu()
        .numpy()
    )
    depot = instance.depot_location.squeeze(0).cpu().numpy()

    if min_coordinate is None or max_coordinate is None:
        min_coordinate = float(customer_locations.min())
        max_coordinate = float(customer_locations.max())

    scatter = ax.scatter(
        customer_locations[:, 0],
        customer_locations[:, 1],
        c=customer_demands,
        cmap="Reds",
        s=100,
        vmin=0,
        vmax=1.0 if with_normalized_demand else float(customer_demands.max()),
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
        label=depot_label,
        edgecolors="black",
        linewidths=1,
        zorder=3,
    )

    plt.colorbar(
        scatter, ax=ax, label="Demand (normalised)" if with_normalized_demand else "Demand"
    )
    ax.set_xlim(min_coordinate - 0.1, max_coordinate + 0.1)
    ax.set_ylim(min_coordinate - 0.1, max_coordinate + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    ax.set_title(f"CMSTP({instance.num_customers})")
    ax.grid(True, linestyle="--", alpha=0.4)


def render_solution(
    solution: CMSTPSolution,
    *,
    min_coordinate: float | None = None,
    max_coordinate: float | None = None,
    ax: Axes | None = None,
    with_problem_size: bool = True,
    with_objective: bool = True,
    separator: str = " | ",
    palette: str = "tab20",
    scale: float = 1.0,
    depot_label: str = "depot",
) -> None:
    if solution.batch_size.numel() > 1:
        raise ValueError(
            f"visualize() only works on a single instance. "
            f"Current batch_size is {solution.batch_size}. Try solution[0].visualize()."
        )

    solution = solution.squeeze(0) if solution.batch_size else solution
    instance = solution.instance.squeeze(0) if solution.instance.batch_size else solution.instance

    if ax is None:
        ax = plt.gca()

    all_nodes = instance.locations.cpu().numpy()
    depot = instance.depot_location.squeeze(0).cpu().numpy()
    parent = solution.parent_nodes.cpu().numpy()  # (N+1,), 0 = depot, -1 = unassigned
    subtree_id = solution.subtree_ids.cpu().numpy()  # (N+1,), -1 for depot

    if min_coordinate is None or max_coordinate is None:
        min_coordinate = float(all_nodes.min())
        max_coordinate = float(all_nodes.max())

    subtree_colours = resolve_palette(palette)
    # Stable color per subtree head so the same subtree keeps its colour across nodes.
    subtree_heads = sorted({int(s) for s in subtree_id[1:]})
    head_to_colour = {
        head: subtree_colours[i % len(subtree_colours)] for i, head in enumerate(subtree_heads)
    }

    ax.scatter(
        depot[0],
        depot[1],
        c="orange",
        s=120 * scale,
        marker="s",
        edgecolors="black",
        linewidths=1.0 * scale,
        label=depot_label,
        zorder=5,
    )

    for i in range(1, instance.num_customers + 1):
        if parent[i] == solution.UNASSIGNED:
            ax.scatter(
                all_nodes[i, 0],
                all_nodes[i, 1],
                c="lightgrey",
                s=100 * scale,
                edgecolors="black",
                linewidths=0.5 * scale,
                zorder=4,
            )
            continue

        colour = head_to_colour[int(subtree_id[i])]
        ax.scatter(
            all_nodes[i, 0],
            all_nodes[i, 1],
            c=[colour],
            s=100 * scale,
            edgecolors="black",
            linewidths=0.5 * scale,
            zorder=4,
        )
        ax.plot(
            [all_nodes[i, 0], all_nodes[parent[i], 0]],
            [all_nodes[i, 1], all_nodes[parent[i], 1]],
            color=colour,
            lw=1.5 * scale,
            zorder=2,
        )

    ax.set_xlim(min_coordinate - 0.1, max_coordinate + 0.1)
    ax.set_ylim(min_coordinate - 0.1, max_coordinate + 0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    title_parts = []
    if with_problem_size:
        title_parts.append(f"CMSTP({instance.num_customers})")
    num_unassigned = int(solution.num_unassigned.item())
    if num_unassigned > 0:
        title_parts.append(f"{num_unassigned} unassigned")
    if with_objective and bool(solution.is_complete.item()):
        title_parts.append(f"total distance = {solution.objective().item():.4f}")
    ax.set_title(separator.join(title_parts))
