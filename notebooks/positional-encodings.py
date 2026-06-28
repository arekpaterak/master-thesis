import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Positional encodings in CVRP

    Two geometric signals encode where each customer sits before the initial projection to a hidden embedding in a model:

    - **IPE** – in-route positional encoding: phase $\phi_i \in [0, 2\pi]$; solution geometry
    - **XPE** – cross-route positional encoding: depot-anchored polar angle $\theta_i \in [-\pi, \pi]$; instance geometry only
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from core.models.embeddings.positional import circular_sinusoidal_encoding
    from core.problems.cvrp.graph_solution import CVRPGraphSolution
    from core.problems.cvrp.sampler import CVRPSampler
    from core.problems.cvrp.solve import index_sequential

    return (
        CVRPGraphSolution,
        CVRPSampler,
        circular_sinusoidal_encoding,
        index_sequential,
        mo,
        np,
        plt,
        torch,
    )


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

    $$\theta_i = \operatorname{atan2}(y_i - y_d,\; x_i - x_d) \in [-\pi, \pi]$$

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Integer vs geometric: why one is reversible and the other is not

    The raw signals above ($\phi$, $\theta$) are just angles on a circle. They become model
    inputs only after `circular_sinusoidal_encoding` maps each phase to a vector. That step
    has a `schedule` knob - **integer** or **geometric** - and this is where reversibility
    is won or lost.

    **Route reversal.** A CVRP route is a closed cycle depot $\to \dots \to$ depot. Driving
    it backwards is the *same* route with the *same* cost, so the in-route phase of a
    customer flips: $\phi \mapsto 2\pi - \phi$. For the cosine-only variant (the project
    default, `direction_aware=False`) each component is $\cos(\omega_k \phi)$, and under
    reversal:

    $$\cos\!\big(\omega_k (2\pi - \phi)\big)
      = \cos(\omega_k 2\pi)\cos(\omega_k \phi) + \sin(\omega_k 2\pi)\sin(\omega_k \phi)$$

    - **integer** $\omega_k = k + 1$: $\omega_k 2\pi$ is a whole number of turns, so
      $\cos(\omega_k 2\pi) = 1$ and $\sin(\omega_k 2\pi) = 0$. The result collapses back to
      $\cos(\omega_k \phi)$: the encoding of a reversed route is **identical**. The harmonics
      *close up* on the circle - $\phi = 0$ and $\phi = 2\pi$ (the same depot point) land on
      the same value.
    - **geometric** $\omega_k = \text{base}^{-2k/D}$ (the Vaswani line schedule): $\omega_k$
      is not an integer, the $\sin(\omega_k 2\pi)$ term survives, and the encoding moves. The
      basis functions do **not** close on the circle - there is a jump between $\phi = 0$ and
      $\phi = 2\pi$, so a reversed route gets a different encoding.

    (The `direction_aware=True` variant adds the $\sin$ components on purpose, to *keep*
    travel direction; reversibility is only a desideratum for the cosine-only variant.)
    """)
    return


@app.cell
def _(circular_sinusoidal_encoding, np, plt, torch):
    _dim = 8
    _phi = torch.linspace(0, 2 * np.pi, 400)
    _enc_int = circular_sinusoidal_encoding(_phi, dim=_dim, direction_aware=False).numpy()
    _enc_geo = circular_sinusoidal_encoding(
        _phi, dim=_dim, direction_aware=False, schedule="geometric"
    ).numpy()
    _phi = _phi.numpy()

    fig_basis, (ax_basis_int, ax_basis_geo) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    _components = [0, 2, 5, 7]  # a spread of low- and high-frequency components
    _colors = plt.cm.viridis(np.linspace(0, 0.9, len(_components)))

    for _ax, _enc, _title in (
        (ax_basis_int, _enc_int, "integer (omega_k = k + 1)"),
        (ax_basis_geo, _enc_geo, "geometric (omega_k = base^(-2k/D))"),
    ):
        for _color, _k in zip(_colors, _components):
            _ax.plot(_phi, _enc[:, _k], color=_color, lw=1.6, label=f"k={_k}")
            # Mark where the curve starts (phi=0) vs where it ends (phi=2*pi).
            _ax.scatter([0], [_enc[0, _k]], color=_color, s=30, zorder=3)
            _ax.scatter([2 * np.pi], [_enc[-1, _k]], color=_color, s=30, marker="X", zorder=3)
        _ax.axvline(0, color="gray", ls="--", lw=0.8)
        _ax.axvline(2 * np.pi, color="gray", ls="--", lw=0.8)
        _ax.set_xlabel("phi (radians)")
        _ax.set_xticks([0, np.pi, 2 * np.pi])
        _ax.set_xticklabels(["0\n(depot)", "pi", "2pi\n(depot)"])
        _ax.set_title(_title)
        _ax.legend(fontsize=8, loc="lower left")

    ax_basis_int.set_ylabel("cos(omega_k * phi)")
    ax_basis_int.text(
        np.pi, 1.15, "start (o) and end (X) coincide", ha="center", fontsize=9, color="green"
    )
    ax_basis_geo.text(
        np.pi, 1.15, "start (o) and end (X) jump apart", ha="center", fontsize=9, color="firebrick"
    )
    plt.tight_layout()
    fig_basis
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Left: every integer-harmonic component returns to exactly where it started after one full
    turn (the dot and cross overlap), so the circle is closed. Right: the higher-frequency
    geometric components end somewhere else than they started - the encoding has a seam at the
    depot. That seam is precisely what reversal exposes.

    Now the direct test on the actual route phases from this instance: encode each customer
    forward ($\phi$) and reversed ($2\pi - \phi$), and measure how far the two encodings sit
    apart.
    """)
    return


@app.cell
def _(circular_sinusoidal_encoding, np, phase, plt, torch):
    _dim = 16
    _phase = torch.from_numpy(phase).float()
    _phase_rev = (2 * np.pi - _phase) % (2 * np.pi)

    _diffs = {}
    for _schedule in ("integer", "geometric"):
        _enc = circular_sinusoidal_encoding(
            _phase, dim=_dim, direction_aware=False, schedule=_schedule
        )
        _enc_rev = circular_sinusoidal_encoding(
            _phase_rev, dim=_dim, direction_aware=False, schedule=_schedule
        )
        _diffs[_schedule] = (_enc.numpy(), _enc_rev.numpy())

    _N = len(_phase)
    fig_reversal, (ax_rev_vec, ax_rev_bar) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: one representative customer, forward vs reversed encoding vector.
    _c = int(np.argmax(np.abs(phase - np.pi)))  # the customer furthest from the symmetric point
    for _schedule, _color in (("integer", "tab:blue"), ("geometric", "tab:red")):
        _enc, _enc_rev = _diffs[_schedule]
        ax_rev_vec.plot(_enc[_c], color=_color, lw=1.6, label=f"{_schedule} forward")
        ax_rev_vec.plot(_enc_rev[_c], color=_color, lw=1.6, ls="--", label=f"{_schedule} reversed")
    ax_rev_vec.set_xlabel("encoding component")
    ax_rev_vec.set_ylabel("value")
    ax_rev_vec.set_title(f"Customer {_c + 1}: forward (solid) vs reversed (dashed)")
    ax_rev_vec.legend(fontsize=8)

    # Right: per-customer max abs difference between forward and reversed encoding.
    _x = np.arange(1, _N + 1)
    _w = 0.4
    for _offset, (_schedule, _color) in zip(
        (-_w / 2, _w / 2), (("integer", "tab:blue"), ("geometric", "tab:red"))
    ):
        _enc, _enc_rev = _diffs[_schedule]
        _max_diff = np.abs(_enc - _enc_rev).max(axis=1)
        ax_rev_bar.bar(
            _x + _offset,
            _max_diff,
            width=_w,
            color=_color,
            edgecolor="black",
            linewidth=0.5,
            label=f"{_schedule} (max={_max_diff.max():.2e})",
        )
    ax_rev_bar.set_xlabel("Customer")
    ax_rev_bar.set_ylabel("max |enc - enc_reversed|")
    ax_rev_bar.set_xticks(_x)
    ax_rev_bar.set_title("Reversal error per customer")
    ax_rev_bar.legend(fontsize=8)

    plt.tight_layout()
    fig_reversal
    return


if __name__ == "__main__":
    app.run()
