import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import math
    import os

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    import yaml

    import wandb
    from core.models.nds.config import TrainConfig
    from core.models.nds.factory import build_model
    from core.models.nds.module import NDSModule
    from core.models.optimizer import build_optimizer
    from core.problems.sampler import Sampler
    from figures.runs import TRAIN_RUNS, WANDB_PROJECT
    from figures.utils import (
        figsize,
        project_root,
        robust_mean,
        save_figure,
        set_white_background,
    )

    set_white_background()

    # Data sources: the live on-machine benchmark and the mean over each run's wandb history.
    # SOURCE_WANDB pulls each method's training run from figures/runs.py (TRAIN_RUNS). The wandb
    # values are the real recorded training throughput, so they keep the plain output names; the
    # benchmark is a quick on-machine timing, tagged "_estimate".
    SOURCE_BENCHMARK = "Own benchmark"
    SOURCE_WANDB = "wandb"
    SOURCE_TAGS = {SOURCE_BENCHMARK: "_estimate", SOURCE_WANDB: ""}

    # Encoder x decoder axes of the 2x2 grid. Assumes "<encoder> + <decoder>" method labels.
    GRID_ENCODERS = ["Joint", "Split"]
    GRID_DECODERS = ["AR", "NAR"]
    GRID_CMAP = "Blues"
    # Per-cell value size, larger than the 9 pt ticks so the values read clearly. The grid
    # width (as a fraction of the text width) is chosen with a slider; author it at the width
    # it is included at in the thesis so the \includegraphics scale factor stays at 1 and fonts
    # keep their true point size.
    GRID_ANNOT_FONTSIZE = 12


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Training time per model

    For the selected problem, two views of two metrics - mean training batch time (seconds)
    and mean training throughput (instances/second) - each drawn as a horizontal bar chart
    (methods on the y-axis) and an encoder x decoder grid heatmap.

    Two data sources can be selected (either or both), each drawn as its own section:

    - **Own benchmark**: measured live on this machine from the training configs in `CONFIGS`.
      The config is built into a real `NDSModule` + optimizer; the first `WARMUP` batches
      (torch.compile / CUDA init) are discarded, then `N_BATCHES` are timed and reduced with
      an outlier-robust mean. Absolute numbers reflect this GPU, but every method is timed
      under identical conditions.
    - **wandb**: the robust mean of `train/batch_time_sec` and `train/instances_per_sec` over
      each method's training run, taken from `figures/runs.py` (`TRAIN_RUNS`). Needs
      `wandb login` and network.

    Pick the problem, size and sources, then click **Measure**. Add a benchmark method by
    appending a `(label, {size: config})` entry to a problem in `CONFIGS`; leave a config path
    empty to skip it, and its grid cell stays blank.
    """)
    return


@app.cell
def _():
    # Problem title -> method label -> {problem size: train config path (under the repo root)}.
    # The title (e.g. "CVRP") labels the plots and names the output subfolder. An empty
    # path is skipped and leaves its grid cell blank.
    TRAIN_DIR = "experiments/configs/train"
    CONFIGS: dict[str, dict[str, dict[int, str]]] = {
        "TSP": {
            "Joint + AR": {
                50: f"{TRAIN_DIR}/tsp/nds_tsp_50+ar.yml",
                100: f"{TRAIN_DIR}/tsp/nds_tsp_100+ar.yml",
            },
            "Joint + NAR": {
                50: f"{TRAIN_DIR}/tsp/nds_tsp_50+simple.yml",
                100: f"{TRAIN_DIR}/tsp/nds_tsp_100+simple.yml",
            },
            "Split + AR": {
                50: f"{TRAIN_DIR}/tsp/nds_tsp_50_split+ar.yml",
                100: f"{TRAIN_DIR}/tsp/nds_tsp_100_split+ar.yml",
            },
            "Split + NAR": {
                50: f"{TRAIN_DIR}/tsp/nds_tsp_50_split+simple.yml",
                100: f"{TRAIN_DIR}/tsp/nds_tsp_100_split+simple.yml",
            },
        },
        "CVRP": {
            "Joint + AR": {
                50: f"{TRAIN_DIR}/cvrp/nds_cvrp_50+ar.yml",
                100: f"{TRAIN_DIR}/cvrp/nds_cvrp_100+ar.yml",
            },
            "Joint + NAR": {
                50: f"{TRAIN_DIR}/cvrp/nds_cvrp_50+simple.yml",
                100: f"{TRAIN_DIR}/cvrp/nds_cvrp_100+simple.yml",
            },
            "Split + AR": {
                50: f"{TRAIN_DIR}/cvrp/nds_cvrp_50_split+ar.yml",
                100: f"{TRAIN_DIR}/cvrp/nds_cvrp_100_split+ar.yml",
            },
            "Split + NAR": {
                50: f"{TRAIN_DIR}/cvrp/nds_cvrp_50_split+simple.yml",
                100: f"{TRAIN_DIR}/cvrp/nds_cvrp_100_split+simple.yml",
            },
        },
        "CMSTP": {
            "Joint + AR": {
                50: f"{TRAIN_DIR}/cmstp/nds_cmstp_50+ar.yml",
                100: f"{TRAIN_DIR}/cmstp/nds_cmstp_100+ar.yml",
            },
            "Joint + NAR": {
                50: f"{TRAIN_DIR}/cmstp/nds_cmstp_50+simple.yml",
                100: f"{TRAIN_DIR}/cmstp/nds_cmstp_100+simple.yml",
            },
            "Split + AR": {
                50: f"{TRAIN_DIR}/cmstp/nds_cmstp_50_split+ar.yml",
                100: f"{TRAIN_DIR}/cmstp/nds_cmstp_100_split+ar.yml",
            },
            "Split + NAR": {
                50: f"{TRAIN_DIR}/cmstp/nds_cmstp_50_split+simple.yml",
                100: f"{TRAIN_DIR}/cmstp/nds_cmstp_100_split+simple.yml",
            },
        },
    }

    # Timed batches per config (after WARMUP discarded batches) and the device to run on.
    WARMUP = 2
    N_BATCHES = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Metric file stem -> (BatchTrainSummary attribute, axis label).
    METRICS = {
        "batch_time": ("batch_time_sec", "Batch time (s)"),
        "instances_per_sec": ("instances_per_sec", "Instances per second"),
    }

    OUT_DIR = project_root() / "figures" / "out"
    return CONFIGS, DEVICE, METRICS, N_BATCHES, OUT_DIR, WARMUP


@app.cell
def _(CONFIGS: dict[str, dict[str, dict[int, str]]]):
    # Sizes offered by the selector are those present in the CONFIGS dict.
    sizes = sorted(
        {
            size
            for methods in CONFIGS.values()
            for config_by_size in methods.values()
            for size in config_by_size
        }
    )
    problem_ui = mo.ui.dropdown(options=list(CONFIGS), value="CVRP", label="Problem")
    size_ui = mo.ui.dropdown(
        options={str(size): size for size in sizes},
        value="100" if 100 in sizes else str(sizes[-1]),
        label="Problem size",
    )
    source_ui = mo.ui.multiselect(
        options=[SOURCE_BENCHMARK, SOURCE_WANDB],
        value=[SOURCE_WANDB],
        label="Sources",
    )
    measure_button = mo.ui.run_button(label="Measure")
    mo.hstack([problem_ui, size_ui, source_ui, measure_button], justify="start", gap=1)
    return measure_button, problem_ui, size_ui, source_ui


@app.cell
def _():
    # Benchmark-only parameters. Defined here (independent of the source selection) so their
    # values are preserved when the source is toggled; shown by the next cell only when needed.
    batch_size_ui = mo.ui.number(start=1, step=1, value=64, label="Batch size")
    rollouts_ui = mo.ui.number(start=1, step=1, value=128, label="Rollouts")
    return batch_size_ui, rollouts_ui


@app.cell
def _(batch_size_ui, rollouts_ui, source_ui):
    # The batch size and rollout count only affect the on-machine benchmark, so surface them
    # only when the benchmark source is selected.
    (
        mo.hstack([batch_size_ui, rollouts_ui], justify="start", gap=1)
        if SOURCE_BENCHMARK in source_ui.value
        else mo.md("")
    )
    return


@app.function
def measure_config(
    config_path: str,
    label: str,
    *,
    device: torch.device,
    batch_size: int,
    rollouts: int,
    warmup: int,
    n_batches: int,
    metrics: dict[str, tuple[str, str]],
) -> dict[str, float]:
    """Build the config into a real training step and time ``n_batches`` batches.

    The instance ``batch_size`` and ``search.num_rollouts`` override the config so all methods
    are timed under the same conditions. The first ``warmup`` batches (torch.compile / CUDA
    init) are discarded; the rest are reduced with an outlier-robust mean. A per-method batch
    progress bar shows live progress and is removed once the method finishes.
    """
    config = TrainConfig.model_validate(yaml.safe_load((project_root() / config_path).read_text()))
    config.search.num_rollouts = rollouts
    model = build_model(config.model).to(device)
    module = NDSModule(config=config, model=model)
    optimizer = build_optimizer(config.optimizer, module.parameters())
    sampler = Sampler.from_config(config.problem)

    samples: dict[str, list[float]] = {stem: [] for stem in metrics}
    for batch_index in mo.status.progress_bar(
        range(warmup + n_batches),
        title=label,
        subtitle="warm-up + timed batches",
        remove_on_exit=True,
    ):
        instance = sampler(batch_size).to(device)
        summary = module.training_step(optimizer=optimizer, instance=instance)
        if batch_index >= warmup:
            for stem, (attribute, _) in metrics.items():
                samples[stem].append(getattr(summary, attribute))

    del model, module, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {stem: robust_mean(values) for stem, values in samples.items()}


@app.function
def load_wandb_run(run_dir: str, *, api, entity: str, project: str):
    """Resolve a training run directory to its wandb run via the saved ``wandb_run_id``."""
    id_path = project_root() / "experiments" / "runs" / run_dir / "wandb_run_id"
    if not id_path.exists():
        raise FileNotFoundError(f"No wandb_run_id at {id_path} (see figures/runs.py).")
    return api.run(f"{entity}/{project}/{id_path.read_text().strip()}")


@app.function
def measure_wandb(run, *, metrics: dict[str, tuple[str, str]]) -> dict[str, float]:
    """Robust mean of each metric over the whole training run's wandb history."""
    keys = {stem: f"train/{attribute}" for stem, (attribute, _) in metrics.items()}
    rows = list(run.scan_history(keys=list(keys.values())))
    return {
        stem: robust_mean([row[key] for row in rows if row.get(key) is not None])
        for stem, key in keys.items()
    }


@app.function
def measure_source(
    source: str,
    *,
    problem: str,
    size: int,
    configs: dict[str, dict[str, dict[int, str]]],
    device: torch.device,
    metrics: dict[str, tuple[str, str]],
    warmup: int,
    n_batches: int,
    batch_size: int,
    rollouts: int,
) -> dict[str, dict[str, float]]:
    """Method label -> {metric stem: mean} for one source at the selected problem/size."""
    if source == SOURCE_BENCHMARK:
        tasks = [
            (label, config_by_size[size])
            for label, config_by_size in configs[problem].items()
            if config_by_size.get(size)
        ]
        api = entity = None
    else:
        # wandb: methods come from figures/runs.py, keyed by "<PROBLEM><size>".
        runs = TRAIN_RUNS.get(f"{problem}{size}", {}).get("runs", [])
        tasks = [(label, run_dir) for label, run_dir in runs if run_dir]
        api = wandb.Api()
        entity = api.default_entity or os.environ.get("WANDB_ENTITY")
        if not entity:
            raise RuntimeError("No wandb entity found. Run `wandb login` or set WANDB_ENTITY.")

    means: dict[str, dict[str, float]] = {}
    for label, ref in mo.status.progress_bar(
        tasks, title=f"{source}: {problem}{size}", subtitle=f"{len(tasks)} methods"
    ):
        if source == SOURCE_BENCHMARK:
            means[label] = measure_config(
                ref,
                label,
                device=device,
                batch_size=batch_size,
                rollouts=rollouts,
                warmup=warmup,
                n_batches=n_batches,
                metrics=metrics,
            )
        else:
            run = load_wandb_run(ref, api=api, entity=entity, project=WANDB_PROJECT)
            means[label] = measure_wandb(run, metrics=metrics)
    return means


@app.cell
def _(
    CONFIGS: dict[str, dict[str, dict[int, str]]],
    DEVICE,
    METRICS,
    N_BATCHES,
    WARMUP,
    batch_size_ui,
    measure_button,
    problem_ui,
    rollouts_ui,
    size_ui,
    source_ui,
):
    mo.stop(
        not measure_button.value,
        mo.md("Pick a problem and size, then click **Measure** to time the configs."),
    )
    mo.stop(
        not source_ui.value,
        mo.md("Select at least one data source (own benchmark and/or wandb)."),
    )

    problem = problem_ui.value
    size = size_ui.value

    # Source -> {method label -> {metric stem: mean}}; only sources with data are kept.
    means_by_source = {
        source: means
        for source in source_ui.value
        if (
            means := measure_source(
                source,
                problem=problem,
                size=size,
                configs=CONFIGS,
                device=DEVICE,
                metrics=METRICS,
                warmup=WARMUP,
                n_batches=N_BATCHES,
                batch_size=batch_size_ui.value,
                rollouts=rollouts_ui.value,
            )
        )
    }
    return means_by_source, problem


@app.function
def plot_bars(means: dict[str, dict[str, float]], stem: str, title: str, x_label: str):
    """Horizontal bar chart of one metric with method labels on the y-axis."""
    labels = list(means.keys())
    values = [means[label][stem] for label in labels]

    fig, ax = plt.subplots(figsize=figsize(1, aspect=0.75))
    positions = range(len(labels))
    ax.barh(list(positions), values, color="tab:blue")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first method at the top

    # Annotate each bar with its value so exact numbers stay readable.
    for position, value in zip(positions, values):
        ax.text(value, position, f" {value:.3g}", va="center", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    vmax = max(values, default=1.0)
    ax.set_xlim(0, vmax * 1.15)  # headroom for the value labels
    return fig


@app.function
def plot_grid(
    means: dict[str, dict[str, float]],
    stem: str,
    title: str,
    cbar_label: str,
    width: float = 0.5,
):
    """Heatmap of one metric over the encoder x decoder grid; missing cells stay blank.

    ``width`` is the figure width as a fraction of the text width; set it to the width the
    grid is included at in the thesis so its fonts print at their true point size.
    """
    matrix = np.array(
        [
            [
                means.get(f"{encoder} + {decoder}", {}).get(stem, math.nan)
                for decoder in GRID_DECODERS
            ]
            for encoder in GRID_ENCODERS
        ]
    )

    # seaborn annotates each cell, auto-contrasts the text color, and leaves masked (missing)
    # cells blank; masking NaN keeps them unannotated.
    fig, ax = plt.subplots(figsize=figsize(width, aspect=0.9))
    sns.heatmap(
        matrix,
        mask=np.isnan(matrix),
        cmap=GRID_CMAP,
        annot=True,
        fmt=".3g",
        annot_kws={"size": GRID_ANNOT_FONTSIZE},
        xticklabels=GRID_DECODERS,
        yticklabels=GRID_ENCODERS,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    # No internal cell separators; seaborn hides the axes spines, so show them to keep only a
    # black outer border, matching the earlier matplotlib heatmap. Outline the colorbar to match.
    for spine in ax.spines.values():
        spine.set_visible(True)
    colorbar = ax.collections[0].colorbar
    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor("black")
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Encoder")
    ax.set_title(title)
    return fig


@app.cell
def _():
    grid_width_ui = mo.ui.slider(
        start=0.2,
        stop=1.0,
        step=0.05,
        value=0.5,
        label="Grid width (fraction of text width)",
        show_value=True,
    )
    grid_width_ui
    return (grid_width_ui,)


@app.cell
def _(METRICS, grid_width_ui, means_by_source, problem):
    # Source -> {output file stem -> figure}: a bar chart and an encoder x decoder grid per
    # metric. Each source is drawn as its own section. The figure titles stay the plain
    # problem name so the saved PDFs are unchanged; the source is shown only in the header.
    figures_by_source = {
        source: {
            **{
                stem: plot_bars(means, stem, problem, label) for stem, (_, label) in METRICS.items()
            },
            **{
                f"{stem}_grid": plot_grid(means, stem, problem, label, width=grid_width_ui.value)
                for stem, (_, label) in METRICS.items()
            },
        }
        for source, means in means_by_source.items()
    }

    sections = []
    for source, figs in figures_by_source.items():
        sections.append(mo.md(f"### {problem} - {source}"))
        sections.append(mo.hstack([f for s, f in figs.items() if not s.endswith("_grid")]))
        sections.append(mo.hstack([f for s, f in figs.items() if s.endswith("_grid")]))
    mo.vstack(sections)
    return (figures_by_source,)


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save figures")
    save_button
    return (save_button,)


@app.cell
def _(OUT_DIR, figures_by_source, problem, save_button, size_ui):
    mo.stop(not save_button.value, mo.md("Click **Save figures** to write the PDFs."))

    problem_dir = OUT_DIR / problem.lower()
    problem_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    # Each source appends its tag (SOURCE_TAGS): wandb keeps the plain name, the benchmark is
    # tagged "_estimate", so the two sources' files sit side by side.
    for _source, _figs in figures_by_source.items():
        _tag = SOURCE_TAGS[_source]
        for _stem, _fig in _figs.items():
            path = problem_dir / f"train_{_stem}{_tag}_{problem.lower()}{size_ui.value}.pdf"
            save_figure(_fig, path)
            saved_paths.append(path)

    mo.md("Saved:\n" + "\n".join(f"- `{p}`" for p in saved_paths))
    return


if __name__ == "__main__":
    app.run()
