from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import polars as pl
from matplotlib.axes import Axes

# Thesis \the\textwidth in points (1 pt = 1/72.27 in). Authoring each figure at its final
# printed width keeps the LaTeX \includegraphics scale factor at 1, so a matplotlib point
# equals a document point and the font size is consistent across figures of any width.
TEXT_WIDTH_PT = 455.24411


def figsize(fraction: float = 1.0, aspect: float = 0.62) -> tuple[float, float]:
    """Figure size in inches for a given fraction of the text width and height/width ratio."""
    width_in = fraction * TEXT_WIDTH_PT / 72.27
    return (width_in, width_in * aspect)


def robust_mean(values: Sequence[float], iqr_factor: float = 1.5) -> float:
    """Mean of ``values`` after dropping IQR outliers (e.g. warm-up spikes)."""
    series = pl.Series(values)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    kept = series.filter((series >= low) & (series <= high))
    return float(kept.mean())


# Not applicable to all figures.
def set_thesis_style() -> None:
    """Set consistent font sizes (in points)."""
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 9,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            # "font.family": "serif",
        }
    )


def hide_ticks(ax: Axes) -> None:
    """Hide tick marks, coordinate labels, and gridlines, keeping the frame. For spatial
    node plots whose coordinate values carry no meaning."""
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(False)


def project_root() -> Path:
    """Repo root, resolved by walking up to the directory holding pyproject.toml.

    Anchoring save paths here keeps them correct regardless of the working
    directory the notebook is launched from.
    """
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate project root (no pyproject.toml found).")
