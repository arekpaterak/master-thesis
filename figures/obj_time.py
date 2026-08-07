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
    # Objective plots

    A pair of plots for one problem/size, read from the saved `steps.csv` files: objective
    against wall-clock time, and objective against improvement step. Each run is a line
    whose points are the logged steps.

    Pick the problem and size from the dropdowns. A run is filled in `PROBLEMS` when its
    `steps.csv` exists; runs that are not evaluated yet are left commented so the intended
    method name is documented and easy to enable later.

    Each run resolves to `results/eval/<problem>/<method>/<task>/steps.csv`.
    """)
    return


@app.cell
def _():
    def variant_runs(*, joint_ar="", joint_nar="", split_ar="", split_nar="", random="") -> list:
        # Random first so the baseline leads the legend (and colour cycle).
        return [
            ("Random", random),
            ("Joint + AR", joint_ar),
            ("Joint + NAR", joint_nar),
            ("Split + AR", split_ar),
            ("Split + NAR", split_nar),
        ]

    # Fill a run with its method (eval config) name once its steps.csv exists; keep the
    # not-yet-evaluated variants commented with the name they will take.
    PROBLEMS: dict[str, dict] = {
        "TSP20": {
            "problem": "tsp",
            "task": "tsp20",
            # NDS is only trained at sizes 50/100.
            "runs": variant_runs(random="random_destroy"),
        },
        "TSP50": {
            "problem": "tsp",
            "task": "tsp50",
            "runs": variant_runs(
                # joint_ar="nds_tsp_50+ar",             # config exists, not evaluated yet
                # joint_nar="nds_tsp_50+simple",        # config exists, not evaluated yet
                # split_nar="nds_tsp_50_split+simple",  # config exists, not evaluated yet
                random="random_destroy",
            ),
        },
        "TSP100": {
            "problem": "tsp",
            "task": "tsp100",
            "runs": variant_runs(
                joint_ar="nds_tsp_100+ar",
                # joint_nar="nds_tsp_100+simple",     # config exists, not evaluated yet
                # split_ar="nds_tsp_100_split+ar",    # no config yet
                split_nar="nds_tsp_100_split+simple",
                random="random_destroy",
            ),
        },
        "CVRP20": {
            "problem": "cvrp",
            "task": "cvrp20",
            # NDS is only trained at sizes 50/100.
            "runs": variant_runs(random="random_destroy"),
        },
        "CVRP50": {
            "problem": "cvrp",
            "task": "cvrp50",
            "runs": variant_runs(
                # joint_ar="nds_cvrp_50+ar",             # config exists, not evaluated yet
                # joint_nar="nds_cvrp_50+simple",        # config exists, not evaluated yet
                # split_nar="nds_cvrp_50_split+simple",  # config exists, not evaluated yet
                random="random_destroy",
            ),
        },
        "CVRP100": {
            "problem": "cvrp",
            "task": "cvrp100",
            # Original-NDS baselines exist too (nds_original_cvrp_100[+nowarmup]); they are
            # a separate comparison and left out of the variant sweep here.
            "runs": variant_runs(
                joint_ar="nds_cvrp_100+ar",
                joint_nar="nds_cvrp_100+simple",
                # split_ar="nds_cvrp_100_split+ar",  # config exists, not evaluated yet
                split_nar="nds_cvrp_100_split+simple",
                random="random_destroy",
            ),
        },
        "CVRP200": {
            "problem": "cvrp",
            "task": "cvrp200",
            # NDS is trained at size 100 but evaluated here for generalisation to size 200,
            # so the runs reuse the size-100 methods with the cvrp200 task.
            "runs": variant_runs(
                # joint_ar="nds_cvrp_100+ar",             # not evaluated on cvrp200 yet
                # joint_nar="nds_cvrp_100+simple",        # not evaluated on cvrp200 yet
                # split_ar="nds_cvrp_100_split+ar",       # config exists, not evaluated
                # split_nar="nds_cvrp_100_split+simple",  # not evaluated on cvrp200 yet
                random="random_destroy",
            ),
        },
        "CMSTP20": {
            "problem": "cmstp",
            "task": "cmstp20",
            # NDS is only trained at sizes 50/100.
            "runs": variant_runs(random="random_destroy"),
        },
        "CMSTP50": {
            "problem": "cmstp",
            "task": "cmstp50",
            "runs": variant_runs(
                # joint_ar="nds_cmstp_50+ar",              # config exists, not evaluated yet
                # joint_nar="nds_cmstp_50+simple",         # config exists, not evaluated yet
                # split_ar="nds_cmstp_50_split+ar",        # config exists, not evaluated yet
                # split_nar="nds_cmstp_50_split+simple",   # config exists, not evaluated yet
                random="random_destroy",
            ),
        },
        "CMSTP100": {
            "problem": "cmstp",
            "task": "cmstp100",
            "runs": variant_runs(
                # joint_ar="nds_cmstp_100+ar",             # config exists, not evaluated yet
                # joint_nar="nds_cmstp_100+simple",        # config exists, not evaluated yet
                # split_ar="nds_cmstp_100_split+ar",       # config exists, not evaluated yet
                # split_nar="nds_cmstp_100_split+simple",  # config exists, not evaluated yet
                random="random_destroy",
            ),
        },
    }

    # Per-instance time keeps runs comparable regardless of test-set size.
    X_COLUMN = "time_total_s"

    OUT_DIR = project_root() / "figures" / "out"
    return OUT_DIR, PROBLEMS, X_COLUMN


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
def _(PROBLEMS: dict[str, dict]):
    problem_names = sorted({spec["problem"] for spec in PROBLEMS.values()})
    default_problem = "cvrp" if "cvrp" in problem_names else problem_names[0]
    problem_ui = mo.ui.dropdown(options=problem_names, value=default_problem, label="Problem")
    return (problem_ui,)


@app.cell
def _(PROBLEMS: dict[str, dict], problem_ui):
    # Sizes available for the chosen problem, parsed off each title (e.g. CVRP200 -> "200").
    problem_sizes = sorted(
        (
            title.removeprefix(spec["problem"].upper())
            for title, spec in PROBLEMS.items()
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
    unit_ui = mo.ui.dropdown(options=list(TIME_UNITS), value="s", label="Time unit")
    return TIME_UNITS, unit_ui


@app.cell
def _(problem_ui, size_ui):
    mo.hstack([problem_ui, size_ui], justify="start")
    return


@app.cell
def _(PROBLEMS: dict[str, dict], load_steps, problem_ui, size_ui):
    title = f"{problem_ui.value.upper()}{size_ui.value}"
    spec = PROBLEMS[title]

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
    mo.vstack([marker_every_ui, skip_first_ui, unit_ui])
    return marker_every_ui, skip_first_ui


@app.function
def plot_obj(
    curves: dict[str, pl.DataFrame],
    title: str,
    x_column: str,
    x_label: str,
    markevery: int,
    skip_first: int,
    x_divisor: float = 1.0,
):
    fig, ax = plt.subplots(figsize=figsize(1.0))

    # Let matplotlib autoscale the y-axis from the post-skip tail only, so the steep
    # early descent runs off the top instead of flattening the interesting tail.
    for frame in curves.values():
        tail = frame[skip_first:]
        ax.plot(tail[x_column] / x_divisor, tail["obj_mean"])
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
            frame[x_column] / x_divisor,
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
def _(
    TIME_UNITS,
    X_COLUMN,
    curves,
    marker_every_ui,
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
            "obj_time": plot_obj(
                curves,
                title,
                X_COLUMN,
                time_label,
                marker_every_ui.value,
                skip_first_ui.value,
                x_divisor=unit_divisor,
            ),
            "obj_step": plot_obj(
                curves,
                title,
                "step",
                "Improvement step",
                marker_every_ui.value,
                skip_first_ui.value,
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
            f"No filled-in runs for **{title}** yet. Uncomment a run in `PROBLEMS` "
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
