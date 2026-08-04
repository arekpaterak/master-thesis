import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    from collections.abc import Iterable

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    from core.problems.cmstp import CMSTPSampler
    from core.problems.cvrp import CVRPSampler
    from core.problems.solve import generate_initial_solution
    from core.problems.tsp import TSPSampler
    from figures.utils import project_root, save_figure, set_white_background

    set_white_background()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Start solutions (before warm-up)

    The initial solutions produced directly by `generate_initial_solution`, before any
    random-improvement warm-up or search: a random tour (TSP), one route per customer
    (CVRP), and a star to the root (CMSTP).
    """)
    return


@app.cell
def _():
    seed_ui = mo.ui.number(start=0, stop=9999, step=1, value=0, label="Seed")
    seed_ui
    return (seed_ui,)


@app.cell
def _(seed_ui):
    N = 20
    SEED = seed_ui.value
    DEVICE = torch.device("cpu")

    FIGURE_PATH = project_root() / "figures" / "out" / "cvrp_vs_cmstp_start.pdf"
    TSP_FIGURE_PATH = project_root() / "figures" / "out" / "tsp_start.pdf"
    return DEVICE, FIGURE_PATH, N, SEED, TSP_FIGURE_PATH


@app.cell
def _(DEVICE, N, SEED):
    _instance = TSPSampler.for_size(N, seed=SEED)(1).to(DEVICE)
    tsp_start = generate_initial_solution(_instance)
    return (tsp_start,)


@app.cell
def _(tsp_start):
    tsp_fig, tsp_ax = plt.subplots(figsize=(7, 6))

    tsp_start[0].visualize(ax=tsp_ax, with_direction=False)
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
def _(DEVICE, N, SEED):
    _instance = CVRPSampler.for_size(N, seed=SEED)(1).to(DEVICE)
    cvrp_start = generate_initial_solution(_instance)
    return (cvrp_start,)


@app.cell
def _(DEVICE, N, SEED):
    _instance = CMSTPSampler.for_size(N, seed=SEED)(1).to(DEVICE)
    # cmstp_start = generate_initial_solution(_instance)
    return


@app.cell
def _(cvrp_start):
    fig, axes = plt.subplots(1, 1, figsize=(7, 6))
    axes = [axes] if not isinstance(axes, Iterable) else axes

    cvrp_start[0].visualize(ax=axes[0], show_depot_edges=True, with_direction=False)
    # cmstp_start[0].visualize(ax=axes[1])

    axes[0].set_title("CVRP / CMSTP")
    # axes[1].set_title("CMSTP")

    fig.tight_layout()
    fig
    return (fig,)


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save figure")
    save_button
    return (save_button,)


@app.cell
def _(FIGURE_PATH, fig, save_button):
    mo.stop(not save_button.value, mo.md("Click **Save figure** to write the PDF."))

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIGURE_PATH)
    mo.md(f"Saved to `{FIGURE_PATH}`")
    return


if __name__ == "__main__":
    app.run()
