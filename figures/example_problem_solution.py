import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from core.lns.config import SearchParams, StartSolutionParams
    from core.methods import RandomDestroyMethodConfig, build_method
    from core.problems import make_problem_config
    from core.problems.cmstp import CMSTPSampler
    from core.problems.cvrp import CVRPSampler
    from core.problems.tsp import TSPSampler
    from core.solve import solve
    from figures.utils import project_root, save_figure, set_white_background

    set_white_background()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Example problem solutions
    """)
    return


@app.cell
def _():
    seed_ui = mo.ui.number(start=0, stop=9999, step=1, value=0, label="Seed")
    steps_ui = mo.ui.slider(
        start=10, stop=1000, step=10, value=300, label="Improvement steps", show_value=True
    )
    deconstruction_ui = mo.ui.slider(
        start=1, stop=20, step=1, value=15, label="Deconstruction size", show_value=True
    )
    include_cmstp_ui = mo.ui.checkbox(value=True, label="Include CMSTP")
    mo.vstack([seed_ui, steps_ui, deconstruction_ui, include_cmstp_ui])
    return deconstruction_ui, include_cmstp_ui, seed_ui, steps_ui


@app.cell
def _(deconstruction_ui, seed_ui, steps_ui):
    N = 20
    SEED = seed_ui.value
    DEVICE = torch.device("cpu")

    # Search configuration.
    NUM_IMPROVEMENT_STEPS = steps_ui.value
    NUM_ROLLOUTS = 1
    DECONSTRUCTION_SIZE = deconstruction_ui.value
    N_REPAIRS = 5
    WARMUP_STEPS = 50

    CVRP_FIGURE_PATH = project_root() / "figures" / "out" / "cvrp.pdf"
    CMSTP_FIGURE_PATH = project_root() / "figures" / "out" / "cmstp.pdf"
    COMBINED_FIGURE_PATH = project_root() / "figures" / "out" / "cvrp_vs_cmstp.pdf"
    TSP_FIGURE_PATH = project_root() / "figures" / "out" / "tsp.pdf"
    return (
        CMSTP_FIGURE_PATH,
        COMBINED_FIGURE_PATH,
        CVRP_FIGURE_PATH,
        DECONSTRUCTION_SIZE,
        DEVICE,
        N,
        NUM_IMPROVEMENT_STEPS,
        NUM_ROLLOUTS,
        N_REPAIRS,
        SEED,
        TSP_FIGURE_PATH,
        WARMUP_STEPS,
    )


@app.cell
def _(
    DECONSTRUCTION_SIZE,
    DEVICE,
    NUM_IMPROVEMENT_STEPS,
    NUM_ROLLOUTS,
    N_REPAIRS,
    SEED,
    WARMUP_STEPS,
):
    def build_random_method(problem_name, n):
        search = SearchParams(
            repair_type="greedy_insert",
            improvement_step_type="numba_sequential",
            num_improvement_steps=NUM_IMPROVEMENT_STEPS,
            num_rollouts=NUM_ROLLOUTS,
            deconstruction_size=DECONSTRUCTION_SIZE,
            n_repairs=N_REPAIRS,
            n_augmentations=1,
            start_solution_params=StartSolutionParams(num_improvement_steps=WARMUP_STEPS),
        )
        config = RandomDestroyMethodConfig(seed=SEED, search=search)
        return build_method(config, problem=make_problem_config(problem_name, n), device=DEVICE)

    return (build_random_method,)


@app.cell
def _(DEVICE, N, SEED, build_random_method):
    _instance = TSPSampler.for_size(N, seed=SEED)(1).to(DEVICE)
    with torch.no_grad():
        tsp_solution = solve(build_random_method("tsp", N), _instance)
    return (tsp_solution,)


@app.cell
def _(tsp_solution):
    tsp_fig, tsp_ax = plt.subplots(figsize=(7, 6))

    tsp_solution[0].visualize(ax=tsp_ax)
    tsp_ax.set_title("TSP")

    tsp_fig.tight_layout()
    tsp_fig
    return (tsp_fig,)


@app.cell
def _():
    tsp_save_button = mo.ui.run_button(label="Save TSP figure")
    tsp_save_button
    return (tsp_save_button,)


@app.cell
def _(TSP_FIGURE_PATH, tsp_fig, tsp_save_button):
    mo.stop(not tsp_save_button.value, mo.md("Click **Save TSP figure** to write the PDF."))

    TSP_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_figure(tsp_fig, TSP_FIGURE_PATH)
    mo.md(f"Saved to `{TSP_FIGURE_PATH}`")
    return


@app.cell
def _(DEVICE, N, SEED, build_random_method):
    _instance = CVRPSampler.for_size(N, seed=SEED)(1).to(DEVICE)
    with torch.no_grad():
        cvrp_solution = solve(build_random_method("cvrp", N), _instance)
    return (cvrp_solution,)


@app.cell
def _(DEVICE, N, SEED, build_random_method, include_cmstp_ui):
    cmstp_solution = None
    if include_cmstp_ui.value:
        _instance = CMSTPSampler.for_size(N, seed=SEED)(1).to(DEVICE)
        with torch.no_grad():
            cmstp_solution = solve(build_random_method("cmstp", N), _instance)
    return (cmstp_solution,)


@app.cell
def _(cvrp_solution):
    cvrp_fig, cvrp_ax = plt.subplots(figsize=(7, 6))

    cvrp_solution[0].visualize(ax=cvrp_ax, show_depot_edges=True, palette="tab10")
    cvrp_ax.set_title("CVRP")

    cvrp_fig.tight_layout()
    cvrp_fig
    return (cvrp_fig,)


@app.cell
def _():
    cvrp_save_button = mo.ui.run_button(label="Save CVRP figure")
    cvrp_save_button
    return (cvrp_save_button,)


@app.cell
def _(CVRP_FIGURE_PATH, cvrp_fig, cvrp_save_button):
    mo.stop(not cvrp_save_button.value, mo.md("Click **Save CVRP figure** to write the PDF."))

    CVRP_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_figure(cvrp_fig, CVRP_FIGURE_PATH)
    mo.md(f"Saved to `{CVRP_FIGURE_PATH}`")
    return


@app.cell
def _(cmstp_solution, include_cmstp_ui):
    mo.stop(
        not include_cmstp_ui.value,
        mo.md("Enable **Include CMSTP** to compute and plot the CMSTP solution."),
    )

    cmstp_fig, cmstp_ax = plt.subplots(figsize=(7, 6))

    cmstp_solution[0].visualize(ax=cmstp_ax, palette="tab10")
    cmstp_ax.set_title("CMSTP")

    cmstp_fig.tight_layout()
    cmstp_fig
    return (cmstp_fig,)


@app.cell
def _(include_cmstp_ui):
    cmstp_save_button = mo.ui.run_button(label="Save CMSTP figure")
    mo.stop(not include_cmstp_ui.value)
    cmstp_save_button
    return (cmstp_save_button,)


@app.cell
def _(CMSTP_FIGURE_PATH, cmstp_fig, cmstp_save_button):
    mo.stop(not cmstp_save_button.value, mo.md("Click **Save CMSTP figure** to write the PDF."))

    CMSTP_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_figure(cmstp_fig, CMSTP_FIGURE_PATH)
    mo.md(f"Saved to `{CMSTP_FIGURE_PATH}`")
    return


@app.cell
def _(cmstp_solution, cvrp_solution, include_cmstp_ui):
    mo.stop(
        not include_cmstp_ui.value,
        mo.md("Enable **Include CMSTP** to produce the combined CVRP vs CMSTP figure."),
    )

    combined_fig, combined_axes = plt.subplots(1, 2, figsize=(14, 6))

    cvrp_solution[0].visualize(ax=combined_axes[0], show_depot_edges=True, palette="tab10")
    cmstp_solution[0].visualize(ax=combined_axes[1], palette="tab10")

    combined_axes[0].set_title("CVRP")
    combined_axes[1].set_title("CMSTP")

    combined_fig.tight_layout()
    combined_fig
    return (combined_fig,)


@app.cell
def _(include_cmstp_ui):
    combined_save_button = mo.ui.run_button(label="Save combined figure")
    mo.stop(not include_cmstp_ui.value)
    combined_save_button
    return (combined_save_button,)


@app.cell
def _(COMBINED_FIGURE_PATH, combined_fig, combined_save_button):
    mo.stop(
        not combined_save_button.value, mo.md("Click **Save combined figure** to write the PDF.")
    )

    COMBINED_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_figure(combined_fig, COMBINED_FIGURE_PATH)
    mo.md(f"Saved to `{COMBINED_FIGURE_PATH}`")
    return


if __name__ == "__main__":
    app.run()
