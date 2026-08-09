import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt

    from core.lns import GreedyAcceptance, search
    from core.lns.apply_and_repair import TorchApplyAndRepair
    from core.lns.improvement_step import ImprovementStep
    from core.lns.operators.destroy.random_node import RandomNodeDestroy
    from core.problems.cvrp import CVRPSampler
    from core.problems.cvrp.lns.operators.repair.insert import CVRPGreedyInsertRepair
    from core.problems.cvrp.solve import one_customer_per_route_graph
    from core.problems.tsp import TSPSampler
    from core.problems.tsp.lns.operators.repair.insert import TSPGreedyInsertRepair
    from core.problems.tsp.solve import random_permutation_graph
    from figures.utils import (
        figsize,
        hide_ticks,
        project_root,
        save_figure,
        set_thesis_style,
        set_white_background,
    )

    set_thesis_style()
    set_white_background()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # A single LNS step

    One destroy + repair step shown at a chosen point *inside* the search: the initial
    solution is improved for the selected number of LNS steps, then a single random-node
    destroy (stitched) and greedy-insert repair are applied directly with the operators, and the
    solution is shown before destroy, after destroy, and after repair.
    """)
    return


@app.cell
def _():
    problem_ui = mo.ui.dropdown({"CVRP": "cvrp", "TSP": "tsp"}, value="CVRP", label="Problem")
    seed_ui = mo.ui.number(start=0, stop=9999, step=1, value=18, label="Seed")
    steps_ui = mo.ui.slider(
        start=0, stop=25, step=1, value=1, label="LNS steps before", show_value=True
    )
    destruction_ui = mo.ui.slider(
        start=1, stop=15, step=1, value=4, label="Destruction size", show_value=True
    )
    run_ui = mo.ui.run_button(label="Run")
    mo.vstack(
        [mo.hstack([problem_ui, seed_ui], justify="start", gap=2), steps_ui, destruction_ui, run_ui]
    )
    return destruction_ui, problem_ui, run_ui, seed_ui, steps_ui


@app.cell
def _(destruction_ui, problem_ui, run_ui, seed_ui, steps_ui):
    mo.stop(not run_ui.value, mo.md("Set the controls and click **Run**."))

    N = 10
    SEED = seed_ui.value
    STEPS = steps_ui.value
    is_cvrp = problem_ui.value == "cvrp"

    if is_cvrp:
        instance = CVRPSampler.for_size(N, seed=SEED)(1)
        initial = one_customer_per_route_graph(instance)
        repair_op = CVRPGreedyInsertRepair(seed=SEED)
    else:
        instance = TSPSampler.for_size(N, seed=SEED)(1)
        initial = random_permutation_graph(instance, seed=SEED)
        repair_op = TSPGreedyInsertRepair(seed=SEED)

    destroy_op = RandomNodeDestroy(destruction_size=destruction_ui.value, seed=SEED, stitch=True)

    apply_and_repair = TorchApplyAndRepair(repair=repair_op, n_repairs=1)
    step = ImprovementStep(
        destroy=destroy_op, apply_and_repair=apply_and_repair, mode="sequential", num_rollouts=1
    )
    result = search(
        initial,
        improvement_step=step,
        criterion=GreedyAcceptance(),
        n_iterations=STEPS,
    )

    before = result.solution[0]
    deconstructed = destroy_op(before)
    repaired = repair_op(deconstructed)
    return before, deconstructed, is_cvrp, repaired


@app.cell
def _():
    scale_ui = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.5, label="Marker scale", show_value=True
    )
    scale_ui
    return (scale_ui,)


@app.cell
def _(before, deconstructed, is_cvrp, repaired, scale_ui):
    def show(solution, ax, title):
        if is_cvrp:
            solution.visualize(
                ax=ax,
                show_depot_edges=True,
                palette="tab10",
                with_problem_size=False,
                with_objective=False,
                scale=scale_ui.value,
                legend_loc="lower right",
            )
        else:
            solution.visualize(
                ax=ax, with_problem_size=False, with_objective=False, scale=scale_ui.value
            )
        hide_ticks(ax)
        ax.set_title(title)

    fig, axes = plt.subplots(1, 3, figsize=figsize(1.0, aspect=0.34))
    show(before, axes[0], f"Obj: {before.objective().item():.4f}")
    show(
        deconstructed,
        axes[1],
        f"After destroy | {int(deconstructed.num_unassigned.item())} removed",
    )
    show(repaired, axes[2], f"After repair | obj: {repaired.objective().item():.4f}")

    fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save figure")
    save_button
    return (save_button,)


@app.cell
def _(fig, save_button):
    mo.stop(not save_button.value, mo.md("Click **Save figure** to write the PDF."))

    figure_path = project_root() / "figures" / "out" / "lns_step.pdf"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, figure_path)
    mo.md(f"Saved to `{figure_path}`")
    return


if __name__ == "__main__":
    app.run()
