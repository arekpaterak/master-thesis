import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Large Neighbourhood Search

    This notebook enables to check how the LNS work with different destroy and repair operators.

    Available pairs of operators:

    1. random node destroy + greedy reconnection repair
    """)
    return


@app.cell(hide_code=True)
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from core.lns.operators.destroy.random_node import RandomNodeDestroy
    from core.lns.operators.repair.greedy_reconnect import (
        CVRPGreedyReconnectRepair,
        TSPGreedyReconnectRepair,
    )
    from core.problems.cvrp import CVRPConfig, CVRPSampler
    from core.problems.cvrp.solve import one_customer_per_route_graph
    from core.problems.tsp import TSPConfig, TSPSampler
    from core.problems.tsp.solve import random_permutation_graph

    return (
        CVRPConfig,
        CVRPGreedyReconnectRepair,
        CVRPSampler,
        RandomNodeDestroy,
        TSPConfig,
        TSPGreedyReconnectRepair,
        TSPSampler,
        mo,
        one_customer_per_route_graph,
        plt,
        random_permutation_graph,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    problem_dropdown = mo.ui.dropdown(
        {"CVRP": "cvrp", "TSP": "tsp"},
        value="CVRP",
        label="Problem",
    )
    n_dropdown = mo.ui.dropdown(
        {"10": 10, "20": 20, "50": 50, "100": 100},
        value="20",
        label="N",
    )
    batch_size_slider = mo.ui.slider(1, 64, value=8, label="Batch size")
    num_steps_widget = mo.ui.slider(10, 200, step=10, value=50, label="LNS steps")
    destruction_widget = mo.ui.slider(5, 25, value=5, step=5, label="Destruction size")
    seed_widget = mo.ui.number(0, 100, value=42, label="Seed")
    mo.hstack(
        [
            problem_dropdown,
            n_dropdown,
            batch_size_slider,
            num_steps_widget,
            destruction_widget,
            seed_widget,
        ],
        gap=2,
    )
    return (
        batch_size_slider,
        destruction_widget,
        n_dropdown,
        num_steps_widget,
        problem_dropdown,
        seed_widget,
    )


@app.cell
def _(
    CVRPConfig,
    CVRPSampler,
    TSPConfig,
    TSPSampler,
    batch_size_slider,
    n_dropdown,
    problem_dropdown,
    seed_widget,
):
    if problem_dropdown.value == "cvrp":
        instance = CVRPSampler(CVRPConfig.for_n(n_dropdown.value), seed=seed_widget.value)(
            batch_size_slider.value
        )[0]
    else:
        instance = TSPSampler(TSPConfig(num_locations=n_dropdown.value), seed=seed_widget.value)(
            batch_size_slider.value
        )[0]
    return (instance,)


@app.cell
def _(
    CVRPGreedyReconnectRepair,
    RandomNodeDestroy,
    TSPGreedyReconnectRepair,
    destruction_widget,
    instance,
    one_customer_per_route_graph,
    problem_dropdown,
    random_permutation_graph,
    seed_widget,
):
    if problem_dropdown.value == "cvrp":
        initial = one_customer_per_route_graph(instance)
        repair_op = CVRPGreedyReconnectRepair()
    else:
        initial = random_permutation_graph(instance, seed=seed_widget.value)
        repair_op = TSPGreedyReconnectRepair()

    destroy_op = RandomNodeDestroy(
        destruction_size=destruction_widget.value, seed=seed_widget.value
    )
    return destroy_op, initial, repair_op


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Single LNS step: destroy and repair
    """)
    return


@app.cell
def _(destroy_op, initial, repair_op):
    deconstructed_first = destroy_op(initial)
    repaired_first = repair_op(deconstructed_first)
    return deconstructed_first, repaired_first


@app.cell(hide_code=True)
def _(deconstructed_first, initial, plt, problem_dropdown, repaired_first):
    _is_cvrp = problem_dropdown.value == "cvrp"
    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))

    if _is_cvrp:
        initial.visualize(
            ax=_axes[0],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=True,
            legend_loc=None,
        )
        deconstructed_first.visualize(
            ax=_axes[1],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=True,
            legend_loc=None,
        )
        repaired_first.visualize(
            ax=_axes[2],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=True,
            legend_loc="upper right",
        )
    else:
        initial.visualize(ax=_axes[0], with_objective=False)
        deconstructed_first.visualize(ax=_axes[1], with_objective=False)
        repaired_first.visualize(ax=_axes[2], with_objective=False)

    _axes[0].set_title(f"Initial | obj: {initial.objective().item():.4f}", fontsize=11)
    _axes[1].set_title(
        f"After destroy | unassigned: {deconstructed_first.num_unassigned.item()}", fontsize=11
    )
    _axes[2].set_title(f"After repair | obj: {repaired_first.objective().item():.4f}", fontsize=11)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## LNS loop

    Continuation of the LNS after the first step.
    """)
    return


@app.cell
def _(
    deconstructed_first,
    destroy_op,
    initial,
    num_steps_widget,
    repair_op,
    repaired_first,
):
    _current = repaired_first
    _best = repaired_first
    _best_obj = repaired_first.objective().mean().item()
    _objectives = [
        initial.objective().mean().item(),
        repaired_first.objective().mean().item(),
    ]
    # Initialise to the first step so num_steps == 1 works without special-casing.
    _before_last = initial
    _deconstructed_last = deconstructed_first

    for _step in range(1, num_steps_widget.value):
        _before_last = _current
        _deconstructed_last = destroy_op(_current)
        _current = repair_op(_deconstructed_last)
        _obj = _current.objective().mean().item()
        if _obj < _best_obj:
            _best = _current
            _best_obj = _obj
        _objectives.append(_obj)

    best_solution = _best
    last_solution = _current
    before_last = _before_last
    deconstructed_last = _deconstructed_last
    objectives = _objectives
    return (
        before_last,
        best_solution,
        deconstructed_last,
        last_solution,
        objectives,
    )


@app.cell(hide_code=True)
def _(objectives, plt):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.plot(objectives, color="steelblue", linewidth=1.5)
    _ax.scatter(range(len(objectives)), objectives, s=18, color="steelblue", zorder=3)
    _ax.set_xlabel("Step (0 = initial, 1 = after first repair)")
    _ax.set_ylabel("Objective")
    _ax.set_title("Objective over LNS iterations")
    _ax.grid(True, linestyle="--", alpha=0.4)
    _fig
    return


@app.cell(hide_code=True)
def _(
    best_solution,
    initial,
    mo,
    objectives,
    problem_dropdown,
    repaired_first,
):
    _is_cvrp = problem_dropdown.value == "cvrp"
    _imp_one = (
        (initial.objective().mean() - repaired_first.objective().mean())
        / initial.objective().mean()
        * 100
    )
    _imp_all = (
        (repaired_first.objective().mean() - best_solution.objective().mean())
        / repaired_first.objective().mean()
        * 100
    )
    _extra_rows = (
        f"| Routes | {initial.num_routes.item()} | {repaired_first.num_routes.item()} | {best_solution.num_routes.item()} |\n"
        f"| Feasible | {bool(initial.is_feasible.item())} | {bool(repaired_first.is_feasible.item())} | {bool(best_solution.is_feasible.item())} |"
        if _is_cvrp
        else ""
    )
    mo.md(f"""
    | | Initial | After first repair | Best found |
    |---|---|---|---|
    | LNS steps | 0 | 1 | {len(objectives) - 1} |
    | Objective | {initial.objective().item():.2f} | {repaired_first.objective().item():.2f} | {best_solution.objective().item():.2f} |
    | Improvement | - | **{_imp_one.item():.1f}%** vs initial | **{_imp_all.item():.1f}%** vs first repair |
    {_extra_rows}
    """)
    return


@app.cell(hide_code=True)
def _(best_solution, initial, plt, problem_dropdown, repaired_first):
    _is_cvrp = problem_dropdown.value == "cvrp"
    _fig, _axes = plt.subplots(1, 3, figsize=(19, 5))

    if _is_cvrp:
        initial.visualize(
            ax=_axes[0],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=False,
            legend_loc=None,
        )
        repaired_first.visualize(
            ax=_axes[1],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=False,
            legend_loc=None,
        )
        best_solution.visualize(
            ax=_axes[2],
            with_objective=False,
            with_route_labels=best_solution.num_routes.item() < 5,
            show_depot_edges=False,
            legend_loc="upper right",
        )
    else:
        initial.visualize(ax=_axes[0], with_objective=False)
        repaired_first.visualize(ax=_axes[1], with_objective=False)
        best_solution.visualize(ax=_axes[2], with_objective=False)

    _axes[0].set_title(f"Initial | obj: {initial.objective().item():.4f}", fontsize=11)
    _axes[1].set_title(
        f"After first repair | obj: {repaired_first.objective().item():.4f}", fontsize=11
    )
    _axes[2].set_title(f"Best found | obj: {best_solution.objective().item():.4f}", fontsize=11)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Last LNS step
    """)
    return


@app.cell(hide_code=True)
def _(
    before_last,
    deconstructed_last,
    last_solution,
    objectives,
    plt,
    problem_dropdown,
):
    _is_cvrp = problem_dropdown.value == "cvrp"
    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))

    if _is_cvrp:
        before_last.visualize(
            ax=_axes[0],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=False,
            legend_loc=None,
        )
        deconstructed_last.visualize(
            ax=_axes[1],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=False,
            legend_loc=None,
        )
        last_solution.visualize(
            ax=_axes[2],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=False,
            legend_loc="upper right",
        )
    else:
        before_last.visualize(ax=_axes[0], with_objective=False)
        deconstructed_last.visualize(ax=_axes[1], with_objective=False)
        last_solution.visualize(ax=_axes[2], with_objective=False)

    _axes[0].set_title(
        f"Before step {len(objectives) - 1} | obj: {before_last.objective().item():.4f}",
        fontsize=11,
    )
    _axes[1].set_title(
        f"After destroy | unassigned: {deconstructed_last.num_unassigned.item()}", fontsize=11
    )
    _axes[2].set_title(f"After repair | obj: {last_solution.objective().item():.4f}", fontsize=11)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Timing benchmark
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bench_n = mo.ui.dropdown(
        {"20": 20, "50": 50, "100": 100},
        value="100",
        label="N",
    )
    bench_batch_sizes = mo.ui.multiselect(
        [32, 64, 128, 256, 512, 1024],
        value=[128, 256, 512, 1024],
        label="Batch sizes",
    )
    bench_destruction_sizes = mo.ui.multiselect(
        [5, 10, 15, 20],
        value=[15],
        label="Destruction sizes",
    )
    bench_num_repeats = mo.ui.slider(1, 20, value=1, label="Repeats")
    bench_lns_steps = mo.ui.slider(10, 200, value=50, step=10, label="LNS steps")
    bench_run = mo.ui.run_button(label="Run")
    mo.hstack(
        [
            bench_n,
            bench_batch_sizes,
            bench_destruction_sizes,
            bench_num_repeats,
            bench_lns_steps,
            bench_run,
        ],
        gap=2,
    )
    return (
        bench_batch_sizes,
        bench_destruction_sizes,
        bench_lns_steps,
        bench_n,
        bench_num_repeats,
        bench_run,
    )


@app.cell(hide_code=True)
def _(
    CVRPConfig,
    CVRPGreedyReconnectRepair,
    CVRPSampler,
    RandomNodeDestroy,
    TSPConfig,
    TSPGreedyReconnectRepair,
    TSPSampler,
    bench_batch_sizes,
    bench_lns_steps,
    bench_n,
    bench_num_repeats,
    one_customer_per_route_graph,
    plt,
    problem_dropdown,
    random_permutation_graph,
    time,
    torch,
):
    def lns_timing_plot(*, destruction_size):
        n = bench_n.value
        batch_sizes = sorted(bench_batch_sizes.value)
        num_repeats = bench_num_repeats.value
        lns_steps = bench_lns_steps.value
        is_cvrp = problem_dropdown.value == "cvrp"
        label = "CVRP" if is_cvrp else "TSP"
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        colors = {"cpu": "steelblue", "cuda": "darkorange"}
        results = {}

        for device in devices:
            loop_ms = []
            for bs in batch_sizes:
                if is_cvrp:
                    inst = CVRPSampler(CVRPConfig.for_n(n), seed=0)(bs)
                    sol = one_customer_per_route_graph(inst)
                else:
                    inst = TSPSampler(TSPConfig(num_locations=n), seed=0)(bs)
                    sol = random_permutation_graph(inst, seed=0)
                if device == "cuda":
                    sol = sol.to(device="cuda")
                r = CVRPGreedyReconnectRepair() if is_cvrp else TSPGreedyReconnectRepair()
                d = RandomNodeDestroy(destruction_size=destruction_size, seed=0)

                for _ in range(2):  # warmup
                    cur = sol
                    for _ in range(lns_steps):
                        cur = r(d(cur))

                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(num_repeats):
                    cur = sol
                    for _ in range(lns_steps):
                        cur = r(d(cur))
                if device == "cuda":
                    torch.cuda.synchronize()
                loop_ms.append((time.perf_counter() - t0) / num_repeats * 1000)

            results[device] = {
                "ms": loop_ms,
                "ips": [bs * 1000 / t for bs, t in zip(batch_sizes, loop_ms)],
            }

        def style(ax, ylabel, title):
            ax.set_xlabel("Batch size")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xscale("log", base=2)
            ax.set_xticks(batch_sizes)
            ax.set_xticklabels([str(b) for b in batch_sizes])
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        kw = dict(marker="o", linewidth=1.5)

        for device, data in results.items():
            axes[0].plot(batch_sizes, data["ms"], label=device, color=colors[device], **kw)
            axes[1].plot(batch_sizes, data["ips"], label=device, color=colors[device], **kw)

        style(axes[0], "Time (ms)", f"LNS loop ({lns_steps} steps): time per loop")
        style(axes[1], "Instances / s", f"LNS loop ({lns_steps} steps): throughput")
        fig.suptitle(
            f"{label}(N={n}) greedy reconnect, destruction size {destruction_size}", fontsize=13
        )
        plt.tight_layout()
        return fig

    return (lns_timing_plot,)


@app.cell(hide_code=True)
def _(bench_destruction_sizes, bench_run, lns_timing_plot, mo):
    mo.stop(not bench_run.value, mo.md("Click **Run** to start the benchmark."))
    mo.vstack([lns_timing_plot(destruction_size=d) for d in sorted(bench_destruction_sizes.value)])
    return


if __name__ == "__main__":
    app.run()
