import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Positional encodings in CVRP

    Two geometric signals encode where each customer sits before the initial projection to a hidden embedding in a model:

    - **IPE** – in-route positional encoding: phase $\phi_i \in [0, 2\pi]$; solution geometry
    - **XPE** – cross-route positional encoding: depot-anchored polar angle $\theta_i \in [0, 2\pi]$; instance geometry only
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from core.problems.cvrp.graph_solution import CVRPGraphSolution
    from core.problems.cvrp.sampler import CVRPSampler
    from core.problems.cvrp.solve import index_sequential

    return CVRPGraphSolution, CVRPSampler, index_sequential, mo, np, plt, torch


@app.cell
def _(CVRPGraphSolution, CVRPSampler, index_sequential, torch):
    torch.manual_seed(42)
    sampler = CVRPSampler.for_size(10)
    instance = sampler(batch_size=1)

    solution_seq = index_sequential(instance)
    solution = CVRPGraphSolution.create_from_sequence(instance, solution_seq.sequence)

    # Split sequence into ordered routes for visualization
    routes = []
    _current_route = []
    for _node in solution_seq.sequence[0].tolist():
        if _node == 0:
            if _current_route:
                routes.append(_current_route)
                _current_route = []
        else:
            _current_route.append(_node)
    return instance, routes, solution


@app.cell
def _(instance, solution):
    locs = instance.locations[0].detach().numpy()
    angles = instance.depot_anchored_polar_angle()[0].detach().numpy()
    acc, total = solution.cumulative_route_length()
    acc = acc[0].detach().numpy()
    total = total[0].detach().numpy()
    phase = solution.route_phase()[0].detach().numpy()
    return acc, angles, locs, phase, total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Instance and routes
    """)
    return


@app.cell
def _(locs, plt, routes):
    _route_colors = plt.cm.tab10.colors

    fig_map, _ax = plt.subplots(figsize=(5, 5))

    for _route_idx, _route in enumerate(routes):
        _color = _route_colors[_route_idx]
        _path = [0] + _route + [0]
        for _i in range(len(_path) - 1):
            _p0, _p1 = locs[_path[_i]], locs[_path[_i + 1]]
            _ax.annotate(
                "",
                xy=_p1,
                xytext=_p0,
                arrowprops=dict(arrowstyle="-|>", color=_color, lw=1.4, mutation_scale=10),
                zorder=1,
            )
        _ax.plot([], [], color=_color, lw=1.5, label=f"Route {_route_idx + 1}")

    _customers = locs[1:]
    _ax.scatter(
        _customers[:, 0],
        _customers[:, 1],
        s=70,
        c="white",
        edgecolors="black",
        linewidths=1,
        zorder=3,
    )
    for _i, (_x, _y) in enumerate(_customers):
        _ax.text(_x + 0.015, _y + 0.015, str(_i + 1), fontsize=9)

    _ax.scatter(
        *locs[0], s=150, marker="s", c="orange", edgecolors="black", zorder=4, label="Depot"
    )
    _ax.set_aspect("equal")
    _ax.set_xlim(-0.02, 1.02)
    _ax.set_ylim(-0.02, 1.02)
    _ax.legend(loc="lower right", fontsize=9)
    _ax.set_title("CVRP instance (index_sequential)")
    plt.tight_layout()
    fig_map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## IPE

    The in-route phase $\phi_i$ is the cumulative arc length from the depot to customer $i$,
    normalized so the full closed route maps to $2\pi$:

    $$\phi_i = \frac{2\pi \cdot \operatorname{acc}_i}{L}, \qquad L = \text{total route length}$$

    The normalization makes the encoding invariant to route length: progress matters, not distance.
    """)
    return


@app.cell
def _(acc, np, phase, plt, routes, total):
    _route_colors = plt.cm.tab10.colors
    _N = sum(len(_r) for _r in routes)

    fig_ipe, (ax_ipe_acc, ax_ipe_phase) = plt.subplots(1, 2, figsize=(9, 4))

    for _route_idx, _route in enumerate(routes):
        _color = _route_colors[_route_idx]
        _label = f"Route {_route_idx + 1}"
        ax_ipe_acc.bar(
            _route, [acc[_c - 1] for _c in _route], color=_color, edgecolor="black", linewidth=0.5
        )
        _t = total[_route[0] - 1]
        ax_ipe_acc.hlines(
            _t,
            _route[0] - 0.45,
            _route[-1] + 0.45,
            colors=_color,
            linestyles="--",
            lw=1.5,
            label=f"{_label} (L={_t:.3f})",
        )
        ax_ipe_phase.bar(
            _route,
            [phase[_c - 1] for _c in _route],
            color=_color,
            edgecolor="black",
            linewidth=0.5,
            label=_label,
        )

    ax_ipe_acc.set_xlabel("Customer")
    ax_ipe_acc.set_ylabel("Cumulative route length")
    ax_ipe_acc.set_xticks(range(1, _N + 1))
    ax_ipe_acc.set_title("Cumulative route length acc$_i$  (-- = route total L)")
    ax_ipe_acc.legend(fontsize=8)

    ax_ipe_phase.axhline(np.pi, color="gray", ls="--", lw=0.8)
    ax_ipe_phase.text(_N + 0.6, np.pi, "π", va="center", fontsize=9, color="gray")
    ax_ipe_phase.set_xlabel("Customer")
    ax_ipe_phase.set_ylabel("φ (radians)")
    ax_ipe_phase.set_xticks(range(1, _N + 1))
    ax_ipe_phase.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax_ipe_phase.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax_ipe_phase.set_ylim(0, 2 * np.pi * 1.08)
    ax_ipe_phase.set_title("In-route phase φ$_i$")
    ax_ipe_phase.legend(fontsize=8)

    plt.tight_layout()
    fig_ipe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XPE

    $$\theta_i = \operatorname{atan2}(y_i - y_d,\; x_i - x_d) \bmod 2\pi$$

    A purely geometric, solution-independent signal. Customers in the same angular sector
    around the depot get similar encodings regardless of which route they are assigned to.
    """)
    return


@app.cell
def _(angles, locs, np, plt):
    _cmap_hsv = plt.cm.twilight
    _customers = locs[1:]
    _depot = locs[0]
    _N = len(_customers)

    fig_xpe, (ax_xpe_map, ax_xpe_bar) = plt.subplots(1, 2, figsize=(9, 4))

    for _i in range(_N):
        ax_xpe_map.plot(
            [_depot[0], _customers[_i, 0]],
            [_depot[1], _customers[_i, 1]],
            color=_cmap_hsv(angles[_i] / (2 * np.pi)),
            lw=1.5,
            alpha=0.6,
        )

    _sc = ax_xpe_map.scatter(
        _customers[:, 0],
        _customers[:, 1],
        c=angles,
        cmap="twilight",
        vmin=0,
        vmax=2 * np.pi,
        s=80,
        edgecolors="black",
        linewidths=1,
        zorder=3,
    )
    ax_xpe_map.scatter(*_depot, s=150, marker="s", c="orange", edgecolors="black", zorder=4)
    for _i, (_x, _y) in enumerate(_customers):
        ax_xpe_map.text(_x + 0.015, _y + 0.015, str(_i + 1), fontsize=9)

    _cb = plt.colorbar(_sc, ax=ax_xpe_map, label="θ [radians]")
    _cb.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    _cb.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax_xpe_map.set_aspect("equal")
    ax_xpe_map.set_xlim(-0.02, 1.02)
    ax_xpe_map.set_ylim(-0.02, 1.02)
    ax_xpe_map.set_title("Depot angle")

    ax_xpe_bar.bar(
        range(1, _N + 1),
        angles,
        color=_cmap_hsv(angles / (2 * np.pi)),
        edgecolor="black",
        linewidth=0.5,
    )
    ax_xpe_bar.set_xlabel("Customer")
    ax_xpe_bar.set_ylabel("θ [radians]")
    ax_xpe_bar.set_xticks(range(1, _N + 1))
    ax_xpe_bar.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax_xpe_bar.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax_xpe_bar.set_ylim(0, 2 * np.pi * 1.08)
    ax_xpe_bar.set_title("Depot angle per customer")

    plt.tight_layout()
    fig_xpe
    return


if __name__ == "__main__":
    app.run()
