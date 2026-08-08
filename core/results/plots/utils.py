from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(fig: Figure, path: str | Path, *, dpi: int | None = None) -> None:
    """Save a figure with a white background and tight bounding box.

    Passing ``dpi`` selects a raster format (e.g. PNG); leave it ``None`` for vector
    formats such as PDF.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        dpi=dpi,
    )


def use_white_background(fig: Figure) -> None:
    """Force white figure and axes backgrounds so figures display and export white
    regardless of the notebook theme."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
