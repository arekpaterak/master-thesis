import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

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
    # Objective curves

    One pair of plots per problem, read from the saved `curve.csv` files: objective
    against wall-clock time, and objective against LNS step. Each run is a line whose
    points are the logged steps.

    Add a problem by giving `PROBLEMS` a new titled entry; add a run by naming its eval
    config in that problem's `runs`. Each run resolves to
    `results/eval/<problem>/<method>/<task>/curve.csv`.
    """)
    return


@app.cell
def _():
    # title -> {problem, task, runs: [(legend label, eval config name)]}.
    # The title (e.g. "CVRP100") labels the plots and names the output subfolder.
    #
    # One entry per problem x size, each with the 2x2 variants + Random. Fill each variant
    # with the name of its eval config (e.g. nds_cvrp_100+ar, nds_cvrp_100_split+simple,
    # random_destroy); empty strings are skipped so the notebook renders with whatever is
    # available. Each run resolves to experiments/results/eval/<problem>/<method>/<task>/curve.csv,
    # where <method> is the config name and <task> is the problem label (cvrp100, cmstp100, ...).
    def variant_runs(*, joint_ar="", joint_nar="", split_ar="", split_nar="", random="") -> list:
        return [
            ("Joint + AR", joint_ar),
            ("Joint + NAR", joint_nar),
            ("Split + AR", split_ar),
            ("Split + NAR", split_nar),
            ("Random", random),
        ]

    PROBLEMS: dict[str, dict] = {
        "TSP50": {"problem": "tsp", "task": "tsp50", "runs": variant_runs()},
        "TSP100": {"problem": "tsp", "task": "tsp100", "runs": variant_runs()},
        "CVRP50": {"problem": "cvrp", "task": "cvrp50", "runs": variant_runs()},
        "CVRP100": {
            "problem": "cvrp",
            "task": "cvrp100",
            "runs": variant_runs(random="random_destroy"),
        },
        "CMSTP50": {"problem": "cmstp", "task": "cmstp50", "runs": variant_runs()},
        "CMSTP100": {"problem": "cmstp", "task": "cmstp100", "runs": variant_runs()},
    }

    # Per-instance time keeps runs comparable regardless of test-set size.
    X_COLUMN = "time_total_s"

    OUT_DIR = project_root() / "figures" / "out"
    return OUT_DIR, PROBLEMS, X_COLUMN


@app.cell
def _(PROBLEMS: dict[str, dict], X_COLUMN):
    RESULTS_EVAL = project_root() / "experiments" / "results" / "eval"

    def load_curve(problem: str, method: str, task: str) -> pl.DataFrame:
        path = RESULTS_EVAL / problem / method / task / "curve.csv"
        if not path.exists():
            raise FileNotFoundError(f"No curve at {path}")
        frame = pl.read_csv(path)
        if X_COLUMN not in frame.columns:
            raise ValueError(
                f"{path} has no '{X_COLUMN}' column (columns: {frame.columns}). "
                "Re-run the eval with the current code to log per-step time."
            )
        return frame

    # Skip runs whose method name is an unfilled placeholder; a named-but-missing curve
    # still raises in load_curve so typos are caught.
    curves_by_problem = {
        title: {
            label: load_curve(spec["problem"], method, spec["task"])
            for label, method in spec["runs"]
            if method
        }
        for title, spec in PROBLEMS.items()
    }
    return (curves_by_problem,)


@app.cell
def _():
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
    mo.vstack([marker_every_ui, skip_first_ui])
    return marker_every_ui, skip_first_ui


@app.function
def plot_obj(
    curves: dict[str, pl.DataFrame],
    title: str,
    x_column: str,
    x_label: str,
    markevery: int,
    skip_first: int,
):
    fig, ax = plt.subplots(figsize=figsize(1.0))

    # Let matplotlib autoscale the y-axis from the post-skip tail only, so the steep
    # early descent runs off the top instead of flattening the interesting tail.
    for frame in curves.values():
        tail = frame[skip_first:]
        ax.plot(tail[x_column], tail["obj_mean"])
    ylim = ax.get_ylim()
    ax.cla()

    # Redraw the full curves and pin the tail-based y-range back.
    for label, frame in curves.items():
        n = len(frame)
        # Mark every N steps, but always mark the final point too.
        marks = list(range(0, n, markevery))
        if n and (n - 1) not in marks:
            marks.append(n - 1)
        ax.plot(
            frame[x_column],
            frame["obj_mean"],
            marker="o",
            markersize=3,
            markevery=marks,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Objective")
    ax.set_xlim(left=0)
    ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    return fig


@app.cell
def _(X_COLUMN, curves_by_problem, marker_every_ui, skip_first_ui):
    X_LABELS = {
        "time_per_instance_s": "Time per instance (s)",
        "time_total_s": "Total time (s)",
    }
    # Per problem: an obj(time) and an obj(step) figure, keyed by output file stem.
    figures = {
        title: {
            "obj_time": plot_obj(
                curves,
                title,
                X_COLUMN,
                X_LABELS.get(X_COLUMN, X_COLUMN),
                marker_every_ui.value,
                skip_first_ui.value,
            ),
            "obj_step": plot_obj(
                curves,
                title,
                "lns_step",
                "Improvement step",
                marker_every_ui.value,
                skip_first_ui.value,
            ),
        }
        # Skip problems with no filled-in runs yet.
        for title, curves in curves_by_problem.items()
        if curves
    }

    mo.vstack(
        [
            mo.vstack([mo.md(f"### {title}"), figs["obj_time"], figs["obj_step"]])
            for title, figs in figures.items()
        ]
    )
    return (figures,)


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save all figures")
    save_button
    return (save_button,)


@app.cell
def _(OUT_DIR, figures, save_button):
    mo.stop(not save_button.value, mo.md("Click **Save all figures** to write the PDFs."))

    saved_paths = []
    for _title, _figs in figures.items():
        problem_dir = OUT_DIR / _title.lower()
        problem_dir.mkdir(parents=True, exist_ok=True)
        for _stem, _fig in _figs.items():
            path = problem_dir / f"{_stem}.pdf"
            save_figure(_fig, path)
            saved_paths.append(path)

    mo.md("Saved:\n" + "\n".join(f"- `{p}`" for p in saved_paths))
    return


if __name__ == "__main__":
    app.run()
