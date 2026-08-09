import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl

    from core.results.plots import plot_objective
    from figures.runs import EVAL_RUNS, METHOD_COLORS, METHOD_MARKERS
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
    # Objective plots

    A pair of plots for one problem/size, read from the saved `steps.csv` files: objective
    against wall-clock time, and objective against improvement step.

    Pick the problem and size from the dropdowns. Runs are defined in `figures/runs.py`
    (`EVAL_RUNS`); a run is filled when its `steps.csv` exists, and runs that are not
    evaluated yet are left commented so the intended method name is documented and easy to
    enable later.

    Each run resolves to `results/eval/<problem>/<method>/<task>/steps.csv`.
    """)
    return


@app.cell
def _():
    # Per-instance time keeps runs comparable regardless of test-set size.
    X_COLUMN = "time_total_s"

    OUT_DIR = project_root() / "figures" / "out"
    return OUT_DIR, X_COLUMN


@app.cell
def _(X_COLUMN):
    RESULTS_EVAL = project_root() / "experiments" / "results" / "eval"

    def load_steps(problem: str, method: str, task: str) -> pl.DataFrame:
        path = RESULTS_EVAL / problem / method / task / "steps.csv"
        if not path.exists():
            raise FileNotFoundError(f"No steps file at {path}")
        frame = pl.read_csv(path)
        if X_COLUMN not in frame.columns:
            raise ValueError(
                f"{path} has no '{X_COLUMN}' column (columns: {frame.columns}). "
                "Re-run the eval with the current code to log per-step time."
            )
        return frame

    return (load_steps,)


@app.cell
def _():
    problem_names = sorted({spec["problem"] for spec in EVAL_RUNS.values()})
    default_problem = "cvrp" if "cvrp" in problem_names else problem_names[0]
    problem_ui = mo.ui.dropdown(options=problem_names, value=default_problem, label="Problem")
    return (problem_ui,)


@app.cell
def _(problem_ui):
    # Sizes available for the chosen problem, parsed off each title (e.g. CVRP200 -> "200").
    problem_sizes = sorted(
        (
            title.removeprefix(spec["problem"].upper())
            for title, spec in EVAL_RUNS.items()
            if spec["problem"] == problem_ui.value
        ),
        key=int,
    )
    default_size = "100" if "100" in problem_sizes else problem_sizes[-1]
    size_ui = mo.ui.dropdown(options=problem_sizes, value=default_size, label="Size")
    return (size_ui,)


@app.cell
def _(problem_ui, size_ui):
    # Runs available for the current problem/size (those with a filled method name); the
    # multiselect defaults to all and lets a subset be picked to plot.
    run_labels = [
        label
        for label, method in EVAL_RUNS[f"{problem_ui.value.upper()}{size_ui.value}"]["runs"]
        if method
    ]
    runs_ui = mo.ui.multiselect(options=run_labels, value=run_labels, label="Runs to plot")
    return (runs_ui,)


@app.cell
def _():
    # Divisor from seconds; the label is the unit shown on the time axis.
    TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0}
    unit_ui = mo.ui.dropdown(options=list(TIME_UNITS), value="min", label="Time unit")
    return TIME_UNITS, unit_ui


@app.cell
def _(problem_ui, runs_ui, size_ui):
    mo.hstack([problem_ui, size_ui, runs_ui], justify="start")
    return


@app.cell
def _(load_steps, problem_ui, runs_ui, size_ui):
    title = f"{problem_ui.value.upper()}{size_ui.value}"
    spec = EVAL_RUNS[title]

    # Skip runs whose method name is an unfilled placeholder; a named-but-missing file
    # still raises in load_steps so typos are caught. Only plot the runs selected in the
    # multiselect.
    selected = set(runs_ui.value)
    curves = {
        label: load_steps(spec["problem"], method, spec["task"])
        for label, method in spec["runs"]
        if method and label in selected
    }
    return curves, spec, title


@app.cell
def _(unit_ui):
    marker_every_ui = mo.ui.slider(
        start=1, stop=50, step=1, value=10, label="Marker every N steps", show_value=True
    )
    skip_first_ui = mo.ui.slider(
        start=0,
        stop=100,
        step=1,
        value=10,
        label="Clip y-range past first N steps",
        show_value=True,
    )
    marker_size_ui = mo.ui.slider(
        start=0, stop=12, step=1, value=5, label="Marker size", show_value=True
    )
    mo.vstack([marker_every_ui, skip_first_ui, marker_size_ui, unit_ui])
    return marker_every_ui, marker_size_ui, skip_first_ui


@app.cell
def _(
    TIME_UNITS,
    X_COLUMN,
    curves,
    marker_every_ui,
    marker_size_ui,
    skip_first_ui,
    title,
    unit_ui,
):
    X_LABELS = {
        "time_per_instance_s": "Time per instance",
        "time_total_s": "Total time",
    }
    unit_divisor = TIME_UNITS[unit_ui.value]
    time_label = f"{X_LABELS.get(X_COLUMN, X_COLUMN)} ({unit_ui.value})"
    # For the selected problem/size: an obj(time) and an obj(step) figure, keyed by output
    # file stem. Empty when the selection has no filled-in runs yet.
    figures = (
        {
            "obj_time": plot_objective(
                curves,
                title=title,
                x_column=X_COLUMN,
                x_label=time_label,
                colors=METHOD_COLORS,
                markers=METHOD_MARKERS,
                markevery=marker_every_ui.value,
                skip_first=skip_first_ui.value,
                x_divisor=unit_divisor,
                markersize=marker_size_ui.value,
                figsize=figsize(1.0),
            ),
            "obj_step": plot_objective(
                curves,
                title=title,
                x_column="step",
                x_label="Improvement step",
                colors=METHOD_COLORS,
                markers=METHOD_MARKERS,
                markevery=marker_every_ui.value,
                skip_first=skip_first_ui.value,
                markersize=marker_size_ui.value,
                figsize=figsize(1.0),
            ),
        }
        if curves
        else {}
    )
    return (figures,)


@app.cell
def _(curves, figures, title):
    if curves:
        view = mo.vstack([mo.md(f"### {title}"), figures["obj_time"], figures["obj_step"]])
    else:
        view = mo.md(
            f"No filled-in runs for **{title}** yet. Uncomment a run in `EVAL_RUNS` "
            "(or evaluate it) to plot it."
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

    # Group by problem (matching the other figure scripts) and name each file by the task
    # so it is self-describing, e.g. out/cvrp/obj_time_cvrp100.pdf.
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
