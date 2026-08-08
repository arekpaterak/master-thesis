from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from core.results.plots.utils import use_white_background


def plot_objective(
    curves: dict[str, pl.DataFrame],
    *,
    title: str,
    x_column: str,
    x_label: str,
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    markevery: int,
    skip_first: int = 0,
    x_divisor: float = 1.0,
    markersize: float = 6,
    figsize: tuple[float, float] | None = None,
    y_column: str = "obj_mean",
) -> Figure:
    """Objective vs ``x_column``, one line per run keyed by label.

    Each frame holds the per-step objective (``y_column``) against ``x_column``. Colors and
    markers are looked up by label, so a run keeps the same style across figures. The y-range
    is autoscaled from the post-``skip_first`` tail only, so the steep early descent runs off
    the top instead of flattening the interesting tail.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for frame in curves.values():
        tail = frame[skip_first:]
        ax.plot(tail[x_column] / x_divisor, tail[y_column])
    ylim = ax.get_ylim()
    ax.cla()

    # Redraw the full curves and pin the tail-based y-range back.
    for label, frame in curves.items():
        n = len(frame)
        # Mark every N steps, but always mark the final point too.
        marks = list(range(0, n, markevery))
        if n and (n - 1) not in marks:
            marks.append(n - 1)
        ax.plot(
            frame[x_column] / x_divisor,
            frame[y_column],
            marker=markers.get(label, "o"),
            markersize=markersize,
            markevery=marks,
            label=label,
            color=colors.get(label),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Objective")
    ax.set_title(title)
    ax.set_xlim(left=0)
    ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    use_white_background(fig)
    return fig
