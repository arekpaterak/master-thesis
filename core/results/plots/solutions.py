from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.problems.solution import BaseSolution
from core.results.plots.utils import use_white_background


def plot_solutions_from_search(
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
    use_white_background(fig)
    return fig
