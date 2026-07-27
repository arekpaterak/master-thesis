from __future__ import annotations

from matplotlib import pyplot as plt

_TAB20_FAMILY = (
    plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors + plt.get_cmap("tab20c").colors
)


def resolve_palette(name: str) -> tuple:
    if name == "tab10":
        return plt.get_cmap("tab10").colors
    if name == "tab20":
        return _TAB20_FAMILY
    raise ValueError(f"Unknown palette {name!r}.")
