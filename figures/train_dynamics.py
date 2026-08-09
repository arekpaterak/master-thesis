import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    import wandb
    from figures.runs import METHOD_COLORS, TRAIN_RUNS, WANDB_PROJECT
    from figures.utils import (
        figsize,
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
    # Training dynamics

    Two views of how each model variant develops during training, fetched from Weights &
    Biases because these metrics were not saved locally: the **validation objective** against
    epoch (`val/objective_final`) and the **normalized policy entropy** against training step
    (`train/entropy_norm`). Each line is one trained variant.

    The runs are defined once in `figures/runs.py` (`TRAIN_RUNS`), where each variant points
    at a training run directory under `experiments/runs`. Each directory's `wandb_run_id` file
    is read to resolve the wandb run, whose history is then queried.

    Pick the problem and size, then click **Fetch** to query wandb (needs `wandb login` and
    network). A variant with no run filled in `TRAIN_RUNS` is skipped.
    """)
    return


@app.cell
def _():
    RUNS_DIR = project_root() / "experiments" / "runs"
    OUT_DIR = project_root() / "figures" / "out"

    # Metric stem -> (wandb key, x-axis key, x-axis label, y-axis label). Validation is logged
    # per epoch; entropy is logged per training batch, indexed by the wandb step counter.
    VAL_METRIC = ("val/objective_final", "epoch", "Epoch", "Validation objective")
    ENTROPY_METRIC = ("train/entropy_norm", "_step", "Training step", "Normalised entropy")
    return ENTROPY_METRIC, OUT_DIR, RUNS_DIR, VAL_METRIC


@app.cell
def _(RUNS_DIR):
    def read_wandb_run_id(run_dir: str) -> str:
        """Read the wandb run id saved next to a training run's checkpoints."""
        path = RUNS_DIR / run_dir / "wandb_run_id"
        if not path.exists():
            raise FileNotFoundError(
                f"No wandb_run_id at {path}. Check the run dir in figures/runs.py."
            )
        return path.read_text().strip()

    return (read_wandb_run_id,)


@app.cell
def _():
    problem_names = sorted({spec["problem"] for spec in TRAIN_RUNS.values()})
    default_problem = "cvrp" if "cvrp" in problem_names else problem_names[0]
    problem_ui = mo.ui.dropdown(options=problem_names, value=default_problem, label="Problem")
    return (problem_ui,)


@app.cell
def _(problem_ui):
    # Sizes available for the chosen problem, parsed off each title (e.g. CVRP100 -> "100").
    problem_sizes = sorted(
        (
            title.removeprefix(spec["problem"].upper())
            for title, spec in TRAIN_RUNS.items()
            if spec["problem"] == problem_ui.value
        ),
        key=int,
    )
    default_size = "100" if "100" in problem_sizes else problem_sizes[-1]
    size_ui = mo.ui.dropdown(options=problem_sizes, value=default_size, label="Size")
    return (size_ui,)


@app.cell
def _(problem_ui, size_ui):
    fetch_button = mo.ui.run_button(label="Fetch")
    mo.hstack([problem_ui, size_ui, fetch_button], justify="start", gap=1)
    return (fetch_button,)


@app.cell
def _(
    ENTROPY_METRIC,
    VAL_METRIC,
    fetch_button,
    problem_ui,
    read_wandb_run_id,
    size_ui,
):
    mo.stop(
        not fetch_button.value,
        mo.md("Pick a problem and size, then click **Fetch** to query wandb."),
    )

    api = wandb.Api()
    entity = api.default_entity or os.environ.get("WANDB_ENTITY")
    if not entity:
        raise RuntimeError("No wandb entity found. Run `wandb login` or set WANDB_ENTITY.")

    def fetch_history(run_id: str, keys: list[str]) -> pl.DataFrame:
        """Full (unsampled) metric history for the given keys as a polars frame."""
        run = api.run(f"{entity}/{WANDB_PROJECT}/{run_id}")
        rows = list(run.scan_history(keys=keys))
        return pl.DataFrame(rows)

    title = f"{problem_ui.value.upper()}{size_ui.value}"
    spec = TRAIN_RUNS[title]
    tasks = [(label, run_dir) for label, run_dir in spec["runs"] if run_dir]

    _val_key, _val_x, _, _ = VAL_METRIC
    _entropy_key, _entropy_x, _, _ = ENTROPY_METRIC

    # Method label -> {"val": frame, "entropy": frame} for the selected problem and size.
    curves: dict[str, dict[str, pl.DataFrame]] = {}
    for label, run_dir in mo.status.progress_bar(
        tasks, title=f"Fetching {title}", subtitle="wandb history"
    ):
        run_id = read_wandb_run_id(run_dir)
        curves[label] = {
            "val": fetch_history(run_id, [_val_x, _val_key]),
            "entropy": fetch_history(run_id, [_entropy_x, _entropy_key]),
        }
    return curves, spec, title


@app.cell
def _():
    smooth_ui = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=10,
        label="Entropy smoothing window (steps)",
        show_value=True,
    )
    smooth_ui
    return (smooth_ui,)


@app.function
def plot_metric(
    curves: dict[str, dict[str, pl.DataFrame]],
    which: str,
    metric_key: str,
    x_key: str,
    title: str,
    x_label: str,
    y_label: str,
    smooth_window: int = 1,
):
    """Line plot of one metric per variant, colored by the shared METHOD_COLORS palette."""
    fig, ax = plt.subplots(figsize=figsize(1.0))
    for label, frames in curves.items():
        frame = frames[which].sort(x_key)
        values = frame[metric_key]
        if smooth_window > 1:
            values = values.rolling_mean(window_size=smooth_window, min_periods=1)
        ax.plot(
            frame[x_key],
            values,
            label=label,
            color=METHOD_COLORS.get(label),
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0)
    ax.legend()
    fig.tight_layout()
    return fig


@app.cell
def _(
    ENTROPY_METRIC,
    VAL_METRIC,
    curves: dict[str, dict[str, pl.DataFrame]],
    smooth_ui,
    title,
):
    val_key, val_x, val_xlabel, val_ylabel = VAL_METRIC
    entropy_key, entropy_x, entropy_xlabel, entropy_ylabel = ENTROPY_METRIC

    # Keyed by output file stem, matching the requested figure names.
    figures = (
        {
            "train_validation_obj": plot_metric(
                curves, "val", val_key, val_x, title, val_xlabel, val_ylabel
            ),
            "train_entropy_norm": plot_metric(
                curves,
                "entropy",
                entropy_key,
                entropy_x,
                title,
                entropy_xlabel,
                entropy_ylabel,
                smooth_window=smooth_ui.value,
            ),
        }
        if curves
        else {}
    )
    return (figures,)


@app.cell
def _(curves: dict[str, dict[str, pl.DataFrame]], figures, title):
    if curves:
        view = mo.vstack(
            [mo.md(f"### {title}"), figures["train_validation_obj"], figures["train_entropy_norm"]]
        )
    else:
        view = mo.md(
            f"No filled-in runs for **{title}** yet. Fill a variant's run dir in "
            "`figures/runs.py` to plot it."
        )
    view
    return


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save figures")
    save_button
    return (save_button,)


@app.cell
def _(OUT_DIR, figures, save_button, spec):
    mo.stop(not save_button.value, mo.md("Click **Save figures** to write the PDFs."))
    mo.stop(not figures, mo.md("Nothing to save for this selection."))

    problem_dir = OUT_DIR / spec["problem"]
    problem_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for _stem, _fig in figures.items():
        path = problem_dir / f"{_stem}_{spec['task']}.pdf"
        save_figure(_fig, path)
        saved_paths.append(path)

    mo.md("Saved:\n" + "\n".join(f"- `{p}`" for p in saved_paths))
    return


if __name__ == "__main__":
    app.run()
