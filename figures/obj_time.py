import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl

    from figures.utils import figsize, project_root, set_thesis_style

    set_thesis_style()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Objective curves

    Two curves per chosen eval run, read from the saved `<label>_curve.csv` files:
    objective against wall-clock time, and objective against LNS step. Each run is a
    line whose points are the logged steps.

    Add a run by appending a `(label, eval_dir)` pair to `RUNS`.

    Paths are relative to the project root; each `eval_dir` is one timestamped result folder
    under `results/eval/<problem>/<config>/<eval_id>/`.
    """)
    return


@app.cell
def _():
    # (legend label, eval result dir relative to project root)
    RUNS: list[tuple[str, str]] = [
        ("Random", "experiments/results/eval/cvrp/random_destroy/20260727-151500"),
        ("NDS (original)", "experiments/results/eval/cvrp/nds_pretrained_cvrp_100/20260727-160436"),
    ]

    # Per-instance time keeps runs comparable regardless of test-set size.
    CURVE_FILE = "cvrp100_curve.csv"
    X_COLUMN = "time_total_s"

    TIME_FIGURE_PATH = project_root() / "figures" / "out" / "obj_time.pdf"
    STEP_FIGURE_PATH = project_root() / "figures" / "out" / "obj_step.pdf"
    return CURVE_FILE, RUNS, STEP_FIGURE_PATH, TIME_FIGURE_PATH, X_COLUMN


@app.cell
def _(CURVE_FILE, RUNS: list[tuple[str, str]], X_COLUMN):
    def load_curve(eval_dir: str) -> pl.DataFrame:
        path = project_root() / eval_dir / CURVE_FILE
        if not path.exists():
            raise FileNotFoundError(f"No curve at {path}")
        frame = pl.read_csv(path)
        if X_COLUMN not in frame.columns:
            raise ValueError(
                f"{path} has no '{X_COLUMN}' column (columns: {frame.columns}). "
                "Re-run the eval with the current code to log per-step time."
            )
        return frame

    curves = {label: load_curve(eval_dir) for label, eval_dir in RUNS}
    return (curves,)


@app.cell
def _():
    marker_every_ui = mo.ui.slider(
        start=1, stop=50, step=1, value=10, label="Marker every N steps", show_value=True
    )
    skip_first_ui = mo.ui.slider(
        start=0, stop=100, step=1, value=10, label="Skip first N steps", show_value=True
    )
    mo.vstack([marker_every_ui, skip_first_ui])
    return marker_every_ui, skip_first_ui


@app.cell
def _(curves):
    def plot_obj(x_column: str, x_label: str, markevery: int, skip_first: int):
        fig, ax = plt.subplots(figsize=figsize(1.0))
        for label, frame in curves.items():
            frame = frame[skip_first:]
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
        ax.set_xlabel(x_label)
        ax.set_ylabel("Objective")
        ax.set_xlim(left=0)
        ax.legend()
        fig.tight_layout()
        return fig

    return (plot_obj,)


@app.cell
def _(X_COLUMN, marker_every_ui, plot_obj, skip_first_ui):
    X_LABELS = {
        "time_per_instance_s": "Time per instance (s)",
        "time_total_s": "Total time (s)",
    }
    time_fig = plot_obj(
        X_COLUMN, X_LABELS.get(X_COLUMN, X_COLUMN), marker_every_ui.value, skip_first_ui.value
    )
    time_fig
    return (time_fig,)


@app.cell
def _():
    time_save_button = mo.ui.run_button(label="Save obj(time) figure")
    time_save_button
    return (time_save_button,)


@app.cell
def _(TIME_FIGURE_PATH, time_fig, time_save_button):
    mo.stop(not time_save_button.value, mo.md("Click **Save obj(time) figure** to write the PDF."))

    TIME_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    time_fig.savefig(TIME_FIGURE_PATH, bbox_inches="tight")
    mo.md(f"Saved to `{TIME_FIGURE_PATH}`")
    return


@app.cell
def _(marker_every_ui, plot_obj, skip_first_ui):
    step_fig = plot_obj("lns_step", "Improvement step", marker_every_ui.value, skip_first_ui.value)
    step_fig
    return (step_fig,)


@app.cell
def _():
    step_save_button = mo.ui.run_button(label="Save obj(step) figure")
    step_save_button
    return (step_save_button,)


@app.cell
def _(STEP_FIGURE_PATH, step_fig, step_save_button):
    mo.stop(not step_save_button.value, mo.md("Click **Save obj(step) figure** to write the PDF."))

    STEP_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    step_fig.savefig(STEP_FIGURE_PATH, bbox_inches="tight")
    mo.md(f"Saved to `{STEP_FIGURE_PATH}`")
    return


if __name__ == "__main__":
    app.run()
