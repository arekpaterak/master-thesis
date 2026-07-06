from __future__ import annotations

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.problems.solution import BaseSolution


def plot_objective_curve(
    step_labels: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    *,
    title: str,
) -> Figure:
    """Mean objective vs improvement step, with a +/- 1 std band."""
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = list(step_labels)
    mean = list(mean)
    std = list(std)
    ax.plot(steps, mean, label="mean")
    ax.fill_between(
        steps,
        [m - s for m, s in zip(mean, std)],
        [m + s for m, s in zip(mean, std)],
        alpha=0.2,
        label="+/- 1 std",
    )
    ax.set_xlabel("Improvement step")
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_search_progress(
    snapshots: list[BaseSolution],
    step_labels: list[int],
    *,
    epoch: int,
) -> Figure:
    n_snapshots = len(snapshots)
    n_cols = min(n_snapshots, 5)
    n_rows = math.ceil(n_snapshots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i, (solution, step) in enumerate(zip(snapshots, step_labels)):
        solution.visualize(ax=axes_flat[i])
        axes_flat[i].set_title(
            axes_flat[i].get_title() + f"\nStep {step}",
            fontsize=9,
        )

    for i in range(n_snapshots, n_rows * n_cols):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"Epoch {epoch}", fontsize=12)
    fig.tight_layout()
    return fig
