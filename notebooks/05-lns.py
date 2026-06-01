import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Large Neighbourhood Search

    This notebook enables to check how the LNS work with different destroy and repair operators.
    """)
    return


@app.cell(hide_code=True)
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from core.lns import GreedyAcceptance, search
    from core.lns.operators.destroy.random_node import RandomNodeDestroy
    from core.lns.operators.repair.greedy_insert import (
        CVRPGreedyInsertRepair,
        TSPGreedyInsertRepair,
    )
    from core.lns.operators.repair.greedy_reconnect import (
        CVRPGreedyReconnectRepair,
        TSPGreedyReconnectRepair,
    )
    from core.models.nds.models.ar import AutoregressiveNDSModel
    from core.problems.cvrp import CVRPConfig, CVRPSampler
    from core.problems.cvrp.solve import one_customer_per_route, one_customer_per_route_graph
    from core.problems.tsp import TSPConfig, TSPSampler
    from core.problems.tsp.solve import random_permutation, random_permutation_graph

    return (
        AutoregressiveNDSModel,
        CVRPConfig,
        CVRPGreedyInsertRepair,
        CVRPGreedyReconnectRepair,
        CVRPSampler,
        GreedyAcceptance,
        RandomNodeDestroy,
        TSPConfig,
        TSPGreedyInsertRepair,
        TSPGreedyReconnectRepair,
        TSPSampler,
        mo,
        one_customer_per_route,
        one_customer_per_route_graph,
        plt,
        random_permutation,
        random_permutation_graph,
        search,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo, torch):
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
    batch_size_slider = mo.ui.dropdown([8, 16, 32, 64, 128], value=8, label="Batch size")
    num_steps_widget = mo.ui.slider(10, 200, step=10, value=50, label="Improvement steps")
    destruction_widget = mo.ui.slider(5, 25, value=5, step=5, label="Destruction size")
    rollout_widget = mo.ui.slider(1, 128, value=1, step=1, label="Rollouts per step")
    seed_widget = mo.ui.number(0, 100, value=42, label="Seed")
    use_gpu_widget = mo.ui.checkbox(value=torch.cuda.is_available(), label="Use GPU")
    return (
        batch_size_slider,
        destruction_widget,
        n_dropdown,
        num_steps_widget,
        problem_dropdown,
        rollout_widget,
        seed_widget,
        use_gpu_widget,
    )


@app.cell(hide_code=True)
def _(
    batch_size_slider,
    destruction_widget,
    mo,
    n_dropdown,
    num_steps_widget,
    problem_dropdown,
    rollout_widget,
    seed_widget,
    use_gpu_widget,
):
    run = mo.ui.run_button(label="Run")
    mo.vstack(
        [
            mo.hstack(
                [
                    problem_dropdown,
                    n_dropdown,
                    use_gpu_widget,
                ],
                gap=2,
                justify="start",
            ),
            mo.hstack(
                [
                    batch_size_slider,
                    num_steps_widget,
                    destruction_widget,
                    rollout_widget,
                    seed_widget,
                ],
                gap=2,
            ),
            run,
        ],
        gap=2,
    )
    return (run,)


@app.cell
def _(
    CVRPConfig,
    CVRPSampler,
    TSPConfig,
    TSPSampler,
    batch_size_slider,
    mo,
    n_dropdown,
    problem_dropdown,
    run,
    seed_widget,
    use_gpu_widget,
):
    mo.stop(not run.value)

    if problem_dropdown.value == "cvrp":
        instance = CVRPSampler(CVRPConfig.for_n(n_dropdown.value), seed=seed_widget.value)(
            batch_size_slider.value
        )
    else:
        instance = TSPSampler(TSPConfig(num_locations=n_dropdown.value), seed=seed_widget.value)(
            batch_size_slider.value
        )
    if use_gpu_widget.value:
        instance = instance.to(device="cuda")
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
        repair_op = CVRPGreedyReconnectRepair(seed=seed_widget.value)
    else:
        initial = random_permutation_graph(instance, seed=seed_widget.value)
        repair_op = TSPGreedyReconnectRepair(seed=seed_widget.value)

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
    initial_one = initial[0]
    deconstructed_first = destroy_op(initial_one)
    repaired_first = repair_op(deconstructed_first)
    return deconstructed_first, initial_one, repaired_first


@app.cell(hide_code=True)
def _(deconstructed_first, initial_one, plt, problem_dropdown, repaired_first):
    _is_cvrp = problem_dropdown.value == "cvrp"
    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))

    if _is_cvrp:
        initial_one.visualize(
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
        initial_one.visualize(ax=_axes[0], with_objective=False)
        deconstructed_first.visualize(ax=_axes[1], with_objective=False)
        repaired_first.visualize(ax=_axes[2], with_objective=False)

    _axes[0].set_title(f"Initial | obj: {initial_one.objective().item():.4f}", fontsize=11)
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
    ## Search loop

    Runs the full search on the whole batch starting from the initial solution.
    """)
    return


@app.cell
def _(
    GreedyAcceptance,
    destroy_op,
    initial,
    num_steps_widget,
    repair_op,
    rollout_widget,
    search,
    torch,
):
    from core.lns.improvement_step import SequentialImprovementStep

    _step = SequentialImprovementStep(
        destroy=destroy_op,
        repair=repair_op,
        num_rollouts=rollout_widget.value,
    )
    _trajectory = [initial.objective().detach().cpu()]
    _result = search(
        initial,
        improvement_step=_step,
        criterion=GreedyAcceptance(),
        n_iterations=num_steps_widget.value,
        on_step=lambda _i, _sol: _trajectory.append(_sol.objective().detach().cpu()),
    )
    best_solution = _result.best_solution
    last_solution = _result.solution

    objectives_per_instance = torch.stack(_trajectory)
    return best_solution, last_solution, objectives_per_instance


@app.cell
def _(destroy_op, last_solution, repair_op):
    before_last = last_solution[0]
    deconstructed_last = destroy_op(before_last)
    last_repaired = repair_op(deconstructed_last)
    return before_last, deconstructed_last, last_repaired


@app.cell(hide_code=True)
def _(objectives_per_instance, plt):
    _steps = range(objectives_per_instance.shape[0])
    _mean = objectives_per_instance.mean(dim=1).numpy()
    _lo = objectives_per_instance.min(dim=1).values.numpy()
    _hi = objectives_per_instance.max(dim=1).values.numpy()
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.fill_between(_steps, _lo, _hi, alpha=0.2, color="steelblue", label="min-max")
    _ax.plot(_steps, _mean, color="steelblue", linewidth=1.8, label="mean")
    _ax.set_xlabel("Step")
    _ax.set_ylabel("Objective")
    _ax.set_title("Objective over improvement steps")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(best_solution, initial, mo, num_steps_widget, problem_dropdown):
    _is_cvrp = problem_dropdown.value == "cvrp"
    _init_mean = initial.objective().mean()
    _best_mean = best_solution.objective().mean()
    _imp = (_init_mean - _best_mean) / _init_mean * 100
    _extra_rows = (
        f"| Routes (median) | {initial.num_routes.median()} | {best_solution.num_routes.median()} |\n"
        f"| All feasible | {bool(initial.is_feasible.all().item())} | {bool(best_solution.is_feasible.all().item())} |"
        if _is_cvrp
        else ""
    )
    mo.md(f"""
    | | Initial | Best found |
    |---|---|---|
    | Improvement steps | 0 | {num_steps_widget.value} |
    | Objective (mean) | {_init_mean.item():.2f} | {_best_mean.item():.2f} |
    | Improvement (mean) | - | **{_imp.item():.1f}%** vs initial |
    {_extra_rows}
    """)
    return


@app.cell(hide_code=True)
def _(best_solution, initial, plt, problem_dropdown):
    _is_cvrp = problem_dropdown.value == "cvrp"
    _initial_one = initial[0]
    _best_one = best_solution[0]
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))

    if _is_cvrp:
        _initial_one.visualize(
            ax=_axes[0],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=False,
            legend_loc=None,
        )
        _best_one.visualize(
            ax=_axes[1],
            with_objective=False,
            with_route_labels=_best_one.num_routes.item() < 5,
            show_depot_edges=False,
            legend_loc="upper right",
        )
    else:
        _initial_one.visualize(ax=_axes[0], with_objective=False)
        _best_one.visualize(ax=_axes[1], with_objective=False)

    _axes[0].set_title(f"Initial | obj: {_initial_one.objective().item():.4f}", fontsize=11)
    _axes[1].set_title(f"Best found | obj: {_best_one.objective().item():.4f}", fontsize=11)
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
    last_repaired,
    num_steps_widget,
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
        last_repaired.visualize(
            ax=_axes[2],
            with_objective=False,
            with_route_labels=False,
            show_depot_edges=False,
            legend_loc="upper right",
        )
    else:
        before_last.visualize(ax=_axes[0], with_objective=False)
        deconstructed_last.visualize(ax=_axes[1], with_objective=False)
        last_repaired.visualize(ax=_axes[2], with_objective=False)

    _axes[0].set_title(
        f"Before step {num_steps_widget.value} | obj: {before_last.objective().item():.4f}",
        fontsize=11,
    )
    _axes[1].set_title(
        f"After destroy | unassigned: {deconstructed_last.num_unassigned.item()}", fontsize=11
    )
    _axes[2].set_title(f"After repair | obj: {last_repaired.objective().item():.4f}", fontsize=11)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Timing benchmark

    Times a single **destroy + repair** step, averaged over the chosen number of repeats.
    With rollout size `R`, each destroy produces `R` deconstructed solutions per instance, so the repair
    handles `B x R` of them.

    A comparison for both LNS operators:

    1. **Repair**: greedy reconnect vs. greedy insert, with random node destroy.
    2. **Destroy**: random node vs. randomly-initialised NDS model, with greedy reconnect repair.

    Each comparison is shown twice: a **batch-size sweep** (x-axis = batch size, at the
    fixed destruction size) and a **destruction-size sweep** (x-axis = destruction size,
    at the fixed batch size).

    Note: greedy insert is sequential and CPU-bound, so large `B x R` can take minutes.
    Increase the rollout size deliberately.
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
        label="Batch sizes (x-axis of batch plot)",
    )
    bench_batch_size = mo.ui.dropdown(
        {"32": 32, "64": 64, "128": 128, "256": 256, "512": 512, "1024": 1024},
        value="128",
        label="Batch size (fixed for destruction plot)",
    )
    bench_destruction_sizes = mo.ui.multiselect(
        [5, 10, 15, 20],
        value=[5, 10, 15, 20],
        label="Destruction sizes (x-axis of destruction plot)",
    )
    bench_destruction_size = mo.ui.dropdown(
        {"5": 5, "10": 10, "15": 15, "20": 20},
        value="15",
        label="Destruction size (fixed for batch plot)",
    )
    bench_rollout_size = mo.ui.dropdown(
        {"1": 1, "8": 8, "32": 32, "64": 64, "128": 128},
        value="1",
        label="Rollout size",
    )
    bench_num_repeats = mo.ui.slider(1, 20, value=10, label="Repeats")
    bench_run = mo.ui.run_button(label="Run")
    mo.vstack(
        [
            mo.md("**Shared**"),
            mo.hstack([bench_n, bench_rollout_size, bench_num_repeats, bench_run], gap=2),
            mo.md("**Batch-size sweep** (x-axis = batch size)"),
            mo.hstack([bench_batch_sizes, bench_destruction_size], gap=2),
            mo.md("**Destruction-size sweep** (x-axis = destruction size)"),
            mo.hstack([bench_destruction_sizes, bench_batch_size], gap=2),
        ],
        gap=1,
    )
    return (
        bench_batch_size,
        bench_batch_sizes,
        bench_destruction_size,
        bench_destruction_sizes,
        bench_n,
        bench_num_repeats,
        bench_rollout_size,
        bench_run,
    )


@app.cell(hide_code=True)
def _(
    AutoregressiveNDSModel,
    CVRPConfig,
    CVRPGreedyInsertRepair,
    CVRPGreedyReconnectRepair,
    CVRPSampler,
    RandomNodeDestroy,
    TSPConfig,
    TSPGreedyInsertRepair,
    TSPGreedyReconnectRepair,
    TSPSampler,
    bench_batch_size,
    bench_batch_sizes,
    bench_destruction_size,
    bench_destruction_sizes,
    bench_n,
    bench_num_repeats,
    bench_rollout_size,
    one_customer_per_route,
    one_customer_per_route_graph,
    plt,
    problem_dropdown,
    random_permutation,
    random_permutation_graph,
    time,
    torch,
):
    # color per (destroy, repair) combo; linestyle per device.
    _combo_colors = {
        ("random", "reconnect"): "steelblue",
        ("nds", "reconnect"): "darkorange",
        ("random", "insert"): "seagreen",
        ("nds", "insert"): "crimson",
    }
    _device_styles = {"cpu": "-", "cuda": "--"}

    def timing_plot(*, combos, title, x_axis):
        """Time a destroy+repair step over a swept axis.

        x_axis="batch": sweep batch sizes (bench_batch_sizes) at a fixed destruction
        size (bench_destruction_size). x_axis="destruction": sweep destruction sizes
        (bench_destruction_sizes) at a fixed batch size (bench_batch_size).
        """
        is_cvrp = problem_dropdown.value == "cvrp"
        problem_name = "cvrp" if is_cvrp else "tsp"
        label = "CVRP" if is_cvrp else "TSP"
        n = bench_n.value
        num_repeats = bench_num_repeats.value
        rollout_size = bench_rollout_size.value
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

        # Build the swept axis and the (batch_size, destruction_size) point list.
        if x_axis == "batch":
            x_values = sorted(bench_batch_sizes.value)
            fixed_k = bench_destruction_size.value
            points = [(bs, fixed_k) for bs in x_values]
            x_label = "Batch size"
            log_x = True
            fixed_text = f"destruction size {fixed_k}"
        else:
            x_values = sorted(bench_destruction_sizes.value)
            fixed_bs = bench_batch_size.value
            points = [(fixed_bs, k) for k in x_values]
            x_label = "Destruction size"
            log_x = False
            fixed_text = f"batch size {fixed_bs}"

        # Each repair operator dictates the solution representation it consumes:
        # reconnect works on the graph solution, insert on the sequence/tour solution.
        if is_cvrp:
            start_builders = {
                "reconnect": one_customer_per_route_graph,
                "insert": one_customer_per_route,
            }
            repair_classes = {
                "reconnect": CVRPGreedyReconnectRepair,
                "insert": CVRPGreedyInsertRepair,
            }
            sampler = CVRPSampler(CVRPConfig.for_n(n), seed=0)
        else:
            start_builders = {
                "reconnect": random_permutation_graph,
                "insert": random_permutation,
            }
            repair_classes = {
                "reconnect": TSPGreedyReconnectRepair,
                "insert": TSPGreedyInsertRepair,
            }
            sampler = TSPSampler(TSPConfig(num_locations=n), seed=0)

        def make_destroy(destroy_name, *, destruction_size, model):
            if destroy_name == "random":
                operator = RandomNodeDestroy(destruction_size=destruction_size, seed=0)

                def destroy(start):
                    # Sample one independent destroy per rollout: (B,) -> (B, R).
                    expanded = start.unsqueeze(1).expand(*start.batch_size, rollout_size)
                    return operator(expanded)

                return destroy

            @torch.no_grad()
            def destroy(start):
                rollout = model.rollout(
                    start,
                    rollouts_per_solution=rollout_size,
                    deconstruction_size=destruction_size,
                    greedy=False,
                )
                # actions: (B, R, k); broadcasts the (B,) solution to (B, R).
                return start.remove_nodes(rollout.actions)

            return destroy

        results = {}
        for device in devices:
            model = AutoregressiveNDSModel(problem_name=problem_name).to(device).eval()
            for destroy_name, repair_name in combos:
                step_ms = []
                candidates_per_s = []
                for bs, k in points:
                    start = start_builders[repair_name](sampler(bs))
                    if device == "cuda":
                        start = start.to(device="cuda")
                    repair = repair_classes[repair_name]()
                    destroy = make_destroy(destroy_name, destruction_size=k, model=model)

                    for _ in range(2):  # warmup
                        repair(destroy(start))

                    if device == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(num_repeats):
                        repair(destroy(start))
                    if device == "cuda":
                        torch.cuda.synchronize()
                    t = (time.perf_counter() - t0) / num_repeats * 1000
                    step_ms.append(t)
                    candidates_per_s.append(bs * rollout_size * 1000 / t)

                results[(destroy_name, repair_name, device)] = {
                    "ms": step_ms,
                    "candidates_per_s": candidates_per_s,
                }

        def style(ax, ylabel, subtitle, *, log_y=False):
            ax.set_xlabel(x_label)
            ax.set_ylabel(ylabel)
            ax.set_title(subtitle)
            if log_x:
                ax.set_xscale("log", base=2)
            if log_y:
                ax.set_yscale("log")
            ax.set_xticks(x_values)
            ax.set_xticklabels([str(v) for v in x_values])
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for (destroy_name, repair_name, device), data in results.items():
            line_kw = dict(
                marker="o",
                linewidth=1.5,
                color=_combo_colors[(destroy_name, repair_name)],
                linestyle=_device_styles[device],
                label=f"{destroy_name}+{repair_name} ({device})",
            )
            axes[0].plot(x_values, data["ms"], **line_kw)
            axes[1].plot(x_values, data["candidates_per_s"], **line_kw)

        style(axes[0], "Time per step (ms)", "Wall-clock per destroy+repair step", log_y=True)
        style(axes[1], "Candidates / s", "Throughput (B x R repaired / s)")
        fig.suptitle(
            f"{title} - {label}(N={n}), R={rollout_size}, {fixed_text}",
            fontsize=12,
        )
        plt.tight_layout()
        return fig

    return (timing_plot,)


@app.cell(hide_code=True)
def _(bench_run, mo, timing_plot):
    mo.stop(not bench_run.value, mo.md("Click **Run** to start the benchmark."))
    _combos = [("random", "reconnect"), ("random", "insert")]
    mo.vstack(
        [
            mo.md("### Repair: reconnect vs. insert (random node destroy)"),
            timing_plot(combos=_combos, title="Repair", x_axis="batch"),
            timing_plot(combos=_combos, title="Repair", x_axis="destruction"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(bench_run, mo, timing_plot):
    mo.stop(not bench_run.value, mo.md("Click **Run** to start the benchmark."))
    _combos = [("random", "reconnect"), ("nds", "reconnect")]
    mo.vstack(
        [
            mo.md("### Destroy: random-node vs. NDS model (greedy reconnect repair)"),
            timing_plot(combos=_combos, title="Destroy", x_axis="batch"),
            timing_plot(combos=_combos, title="Destroy", x_axis="destruction"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
