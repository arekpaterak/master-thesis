import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from core.lns.operators.destroy.random_node import RandomNodeDestroy
    from core.lns.operators.repair.greedy_reconnect import CVRPGreedyReconnectRepair
    from core.problems.cvrp import CVRPConfig, CVRPSampler
    from core.problems.cvrp.solve import one_customer_per_route_graph

    return (
        CVRPConfig,
        CVRPGreedyReconnectRepair,
        CVRPSampler,
        RandomNodeDestroy,
        mo,
        one_customer_per_route_graph,
        plt,
    )


@app.cell
def _(mo):
    n_dropdown = mo.ui.dropdown(
        {"10": 10, "20": 20, "50": 50, "100": 100},
        value="20",
        label="Customers (N)",
    )
    destruction_slider = mo.ui.slider(1, 25, value=5, label="Destruction size")
    seed_slider = mo.ui.slider(0, 99, value=42, label="Seed")
    num_steps_slider = mo.ui.slider(1, 100, value=50, label="LNS steps")
    mo.hstack([n_dropdown, destruction_slider, seed_slider, num_steps_slider], gap=2)
    return destruction_slider, n_dropdown, num_steps_slider, seed_slider


@app.cell
def _(CVRPConfig, CVRPSampler, n_dropdown, seed_slider):
    instance = CVRPSampler(CVRPConfig.for_n(n_dropdown.value), seed=seed_slider.value)(1)[0]
    return (instance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## One destroy / repair
    """)
    return


@app.cell
def _(
    CVRPGreedyReconnectRepair,
    RandomNodeDestroy,
    destruction_slider,
    instance,
    one_customer_per_route_graph,
    seed_slider,
):
    initial = one_customer_per_route_graph(instance)
    partial = RandomNodeDestroy(destruction_size=destruction_slider.value, seed=seed_slider.value)(
        initial
    )
    repaired = CVRPGreedyReconnectRepair()(partial)
    return initial, partial, repaired


@app.cell
def _(initial, partial, plt, repaired):
    fig0, axes0 = plt.subplots(1, 3, figsize=(17, 5))

    initial.visualize(
        ax=axes0[0],
        with_objective=False,
        with_route_labels=False,
        show_depot_edges=True,
        legend_loc=None,
    )
    axes0[0].set_title(
        f"Initial (one per route)  |  obj: {initial.objective().item():.4f}",
        fontsize=11,
    )

    partial.visualize(
        ax=axes0[1],
        with_objective=False,
        with_route_labels=False,
        show_depot_edges=True,
        legend_loc=None,
    )
    axes0[1].set_title(
        f"After destroy  |  unassigned: {partial.num_unassigned.item()}",
        fontsize=11,
    )

    repaired.visualize(
        ax=axes0[2],
        with_route_labels=False,
        show_depot_edges=True,
        legend_loc="upper right",
    )
    axes0[2].set_title(
        f"After repair | obj: {repaired.objective().item():.4f}",
        fontsize=11,
    )

    plt.tight_layout()
    fig0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## LNS loop (greedy accept)
    """)
    return


@app.cell
def _(
    CVRPGreedyReconnectRepair,
    RandomNodeDestroy,
    destruction_slider,
    initial,
    num_steps_slider,
    repaired,
    seed_slider,
):
    repair = CVRPGreedyReconnectRepair()
    current = repaired
    best_solution = repaired
    # Index 0 = initial, index 1 = after first repair, then one entry per LNS step
    objectives = [initial.objective().item(), repaired.objective().item()]

    for lns_step in range(1, num_steps_slider.value):
        destroyed = RandomNodeDestroy(
            destruction_size=destruction_slider.value,
            seed=seed_slider.value + lns_step,
        )(current)
        candidate = repair(destroyed)
        current = candidate
        if current.objective().item() < best_solution.objective().item():
            best_solution = current
        objectives.append(current.objective().item())
    return best_solution, objectives


@app.cell
def _(objectives, plt):
    fig_curve, ax_curve = plt.subplots(figsize=(9, 4))
    ax_curve.plot(objectives, color="steelblue", linewidth=1.5)
    ax_curve.scatter(range(len(objectives)), objectives, s=18, color="steelblue", zorder=3)
    ax_curve.set_xlabel("Step  (0 = initial, 1 = after first repair)")
    ax_curve.set_ylabel("Objective (total distance)")
    ax_curve.set_title("Objective over LNS iterations")
    ax_curve.grid(True, linestyle="--", alpha=0.4)
    fig_curve
    return


@app.cell
def _(best_solution, plt, repaired):
    fig_best, axes_best = plt.subplots(1, 2, figsize=(13, 5))

    repaired.visualize(
        ax=axes_best[0],
        with_route_labels=False,
        show_depot_edges=True,
        legend_loc=None,
    )
    axes_best[0].set_title(
        f"After first repair  |  obj: {repaired.objective().item():.4f}",
        fontsize=11,
    )

    best_solution.visualize(
        ax=axes_best[1],
        with_route_labels=True,
        show_depot_edges=True,
        legend_loc="upper right",
    )
    axes_best[1].set_title(
        f"Best found  |  obj: {best_solution.objective().item():.4f}",
        fontsize=11,
    )

    plt.tight_layout()
    fig_best
    return


@app.cell
def _(best_solution, initial, mo, objectives, repaired):
    improvement_repair = (
        (initial.objective().item() - repaired.objective().item())
        / initial.objective().item()
        * 100
    )
    improvement_lns = (
        (repaired.objective().item() - best_solution.objective().item())
        / repaired.objective().item()
        * 100
    )
    mo.md(f"""
    | | Initial | After first repair | Best found |
    |---|---|---|---|
    | Routes | {initial.num_routes.item()} | {repaired.num_routes.item()} | {best_solution.num_routes.item()} |
    | Objective | {initial.objective().item():.4f} | {repaired.objective().item():.4f} | {best_solution.objective().item():.4f} |
    | Improvement | | **{improvement_repair:.1f}%** vs initial | **{improvement_lns:.1f}%** vs first repair |
    | Feasible | {bool(initial.is_feasible.item())} | {bool(repaired.is_feasible.item())} | {bool(best_solution.is_feasible.item())} |
    | LNS steps | | | {len(objectives) - 2} |
    """)
    return


if __name__ == "__main__":
    app.run()
