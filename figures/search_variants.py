import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl

    from core.results.plots import plot_objective
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
    # Search variants

    Objective against wall-clock time and against improvement step for the same model
    (split encoder, NAR decoder) run under different searches: unconditional acceptance,
    greedy acceptance, simulated annealing (SA), and augmented SA (ASA). Each run is a line
    read from its saved `steps.csv`.

    Runs are defined in `SEARCH_RUNS` below (label -> eval config name); a run is filled once
    its `steps.csv` exists and the others are left commented. Pick the problem and size from
    the dropdowns - only CVRP100 has an evaluated run so far.

    Each run resolves to `results/eval/<problem>/<method>/<task>/steps.csv`.
    """)
    return


@app.cell
def _():
    def search_runs(*, unconditional="", greedy="", sa="", asa="") -> list[tuple[str, str]]:
        # Ordered to match the search-comparison table (Unconditional, Greedy, SA, ASA).
        return [
            ("Unconditional", unconditional),
            ("Greedy", greedy),
            ("SA", sa),
            ("ASA (x8 aug.)", asa),
        ]

    # Title -> {problem, task, runs}. The runs share the split+NAR model and differ only in
    # the search; each name is its eval config (results dir). Only SA is evaluated so far.
    SEARCH_RUNS: dict[str, dict] = {
        "TSP100": {
            "problem": "tsp",
            "task": "tsp100",
            "runs": search_runs(
                # unconditional="nds_tsp_100_split+simple_unconditional",  # not evaluated yet
                # greedy="nds_tsp_100_split+simple_greedy",                # not evaluated yet
                # sa="nds_tsp_100_split+simple",                           # not evaluated yet
                # asa="nds_tsp_100_split+simple_asa",                      # not evaluated yet
            ),
        },
        "CVRP100": {
            "problem": "cvrp",
            "task": "cvrp100",
            "runs": search_runs(
                # unconditional="nds_cvrp_100_split+simple_unconditional",
                greedy="nds_cvrp_100_split+simple_greedy",
                sa="nds_cvrp_100_split+simple",
                asa="nds_cvrp_100_split+simple_asa",
            ),
        },
        "CMSTP100": {
            "problem": "cmstp",
            "task": "cmstp100",
            "runs": search_runs(
                # unconditional="nds_cmstp_100_split+simple_unconditional",  # not evaluated yet
                # greedy="nds_cmstp_100_split+simple_greedy",                # not evaluated yet
                # sa="nds_cmstp_100_split+simple",                           # not evaluated yet
                # asa="nds_cmstp_100_split+simple_asa",                      # not evaluated yet
            ),
        },
    }

    # Distinct color and marker per search variant, consistent across both plots.
    SEARCH_COLORS = {
        "Unconditional": "tab:gray",
        "Greedy": "tab:blue",
        "SA": "tab:red",  # matches the Split + NAR method color in the other figures
        "ASA (x8 aug.)": "tab:purple",
    }
    SEARCH_MARKERS = {
        "Unconditional": "o",
        "Greedy": "s",
        "SA": "^",
        "ASA (x8 aug.)": "D",
    }

    # Per-instance time keeps runs comparable regardless of test-set size.
    X_COLUMN = "time_total_s"
    OUT_DIR = project_root() / "figures" / "out"
    return OUT_DIR, SEARCH_COLORS, SEARCH_MARKERS, SEARCH_RUNS, X_COLUMN


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
def _(SEARCH_RUNS: dict[str, dict]):
    problem_names = sorted({spec["problem"] for spec in SEARCH_RUNS.values()})
    default_problem = "cvrp" if "cvrp" in problem_names else problem_names[0]
    problem_ui = mo.ui.dropdown(options=problem_names, value=default_problem, label="Problem")
    return (problem_ui,)


@app.cell
def _(SEARCH_RUNS: dict[str, dict], problem_ui):
    # Sizes available for the chosen problem, parsed off each title (e.g. CVRP100 -> "100").
    problem_sizes = sorted(
        (
            title.removeprefix(spec["problem"].upper())
            for title, spec in SEARCH_RUNS.items()
            if spec["problem"] == problem_ui.value
        ),
        key=int,
    )
    default_size = "100" if "100" in problem_sizes else problem_sizes[-1]
    size_ui = mo.ui.dropdown(options=problem_sizes, value=default_size, label="Size")
    return (size_ui,)


@app.cell
def _():
    # Divisor from seconds; the label is the unit shown on the time axis.
    TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0}
    unit_ui = mo.ui.dropdown(options=list(TIME_UNITS), value="min", label="Time unit")
    return TIME_UNITS, unit_ui


@app.cell
def _(problem_ui, size_ui):
    mo.hstack([problem_ui, size_ui], justify="start")
    return


@app.cell
def _(SEARCH_RUNS: dict[str, dict], load_steps, problem_ui, size_ui):
    title = f"{problem_ui.value.upper()}{size_ui.value}"
    spec = SEARCH_RUNS[title]

    # Skip runs whose method name is an unfilled placeholder; a named-but-missing file
    # still raises in load_steps so typos are caught.
    curves = {
        label: load_steps(spec["problem"], method, spec["task"])
        for label, method in spec["runs"]
        if method
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
    SEARCH_COLORS,
    SEARCH_MARKERS,
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
            "search_obj_time": plot_objective(
                curves,
                title=title,
                x_column=X_COLUMN,
                x_label=time_label,
                colors=SEARCH_COLORS,
                markers=SEARCH_MARKERS,
                markevery=marker_every_ui.value,
                skip_first=skip_first_ui.value,
                x_divisor=unit_divisor,
                markersize=marker_size_ui.value,
                figsize=figsize(1.0),
            ),
            "search_obj_step": plot_objective(
                curves,
                title=title,
                x_column="step",
                x_label="Improvement step",
                colors=SEARCH_COLORS,
                markers=SEARCH_MARKERS,
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
        view = mo.vstack(
            [mo.md(f"### {title}"), figures["search_obj_time"], figures["search_obj_step"]]
        )
    else:
        view = mo.md(
            f"No filled-in runs for **{title}** yet. Uncomment a run in `SEARCH_RUNS` "
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

    # Group by problem and name each file by the task, e.g.
    # out/cvrp/search_obj_time_cvrp100.pdf.
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
