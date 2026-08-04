import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import yaml

    from core.models.nds.config import TrainConfig
    from core.models.nds.factory import build_model
    from core.models.nds.module import NDSModule
    from core.models.optimizer import build_optimizer
    from core.problems.sampler import Sampler
    from figures.utils import (
        figsize,
        project_root,
        robust_mean,
        save_figure,
        set_white_background,
    )

    set_white_background()

    # Encoder x decoder axes of the 2x2 grid. Assumes "<encoder> + <decoder>" method labels.
    GRID_ENCODERS = ["Joint", "Split"]
    GRID_DECODERS = ["AR", "NAR"]
    GRID_CMAP = "Blues"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Training time per model

    For the selected problem, two views of two metrics measured live from its training
    configs in `CONFIGS`: mean training batch time (seconds) and mean training throughput
    (instances/second). Each metric is drawn both as a horizontal bar chart (methods on
    the y-axis) and as an encoder x decoder grid heatmap.

    Each value is measured on this machine: the config is built into a real
    `NDSModule` + optimizer and a few training batches are run. The first `WARMUP`
    batches (torch.compile / CUDA init) are discarded, then `N_BATCHES` are timed and
    reduced with an outlier-robust mean of `BatchTrainSummary.batch_time_sec`. Absolute
    numbers therefore reflect this GPU, but every method is timed under identical
    conditions, which is what the comparison needs.

    Pick the problem and size, then click **Measure**.
    Add a method by appending a `(label, {size: config})` entry to a problem;
    leave a config path empty to skip it, and its grid cell stays blank.
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
    batch_size_ui = mo.ui.number(start=1, step=1, value=64, label="Batch size")
    measure_button = mo.ui.run_button(label="Measure")
    mo.hstack([problem_ui, size_ui, batch_size_ui, measure_button], justify="start", gap=1)
    return batch_size_ui, measure_button, problem_ui, size_ui


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
    size_ui,
):
    mo.stop(
        not measure_button.value,
        mo.md("Pick a problem and size, then click **Measure** to time the configs."),
    )

    def measure_config(config_path: str, label: str) -> dict[str, float]:
        """Build the config into a real training step and time N_BATCHES batches.

        The config's own ``trainer.batch_size`` is overridden by ``batch_size_ui`` so all
        methods are timed at the same batch size. A per-method batch progress bar shows
        live progress; it is removed once the method finishes.
        """
        config = TrainConfig.model_validate(
            yaml.safe_load((project_root() / config_path).read_text())
        )
        model = build_model(config.model).to(DEVICE)
        module = NDSModule(config=config, model=model)
        optimizer = build_optimizer(config.optimizer, module.parameters())
        sampler = Sampler.from_config(config.problem)

        samples: dict[str, list[float]] = {stem: [] for stem in METRICS}
        for batch_index in mo.status.progress_bar(
            range(WARMUP + N_BATCHES),
            title=label,
            subtitle="warm-up + timed batches",
            remove_on_exit=True,
        ):
            instance = sampler(batch_size_ui.value).to(DEVICE)
            summary = module.training_step(optimizer=optimizer, instance=instance)
            if batch_index >= WARMUP:
                for stem, (attribute, _) in METRICS.items():
                    samples[stem].append(getattr(summary, attribute))

        del model, module, optimizer
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return {stem: robust_mean(values) for stem, values in samples.items()}

    problem = problem_ui.value
    size = size_ui.value
    tasks = [
        (label, config_by_size[size])
        for label, config_by_size in CONFIGS[problem].items()
        if config_by_size.get(size)
    ]

    # Method label -> {metric stem: robust mean}, for the selected problem and size.
    # Outer bar tracks methods (x/N); the inner per-method bar tracks its batches.
    means = {}
    with mo.status.progress_bar(
        total=len(tasks),
        title=f"Measuring {problem}{size}",
        subtitle=f"0/{len(tasks)} methods",
    ) as methods_bar:
        for done, (label, config_path) in enumerate(tasks):
            means[label] = measure_config(config_path, label)
            methods_bar.update(subtitle=f"{done + 1}/{len(tasks)} methods")
    return means, problem


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
def plot_grid(means: dict[str, dict[str, float]], stem: str, title: str, cbar_label: str):
    """Heatmap of one metric over the encoder x decoder grid; missing cells stay blank."""
    matrix = [
        [means.get(f"{encoder} + {decoder}", {}).get(stem, math.nan) for decoder in GRID_DECODERS]
        for encoder in GRID_ENCODERS
    ]

    fig, ax = plt.subplots(figsize=figsize(1, aspect=0.9))
    image = ax.imshow(matrix, cmap=GRID_CMAP, aspect="auto")

    ax.set_xticks(range(len(GRID_DECODERS)), labels=GRID_DECODERS)
    ax.set_yticks(range(len(GRID_ENCODERS)), labels=GRID_ENCODERS)
    ax.set_xlabel("Decoder")
    ax.set_ylabel("Encoder")
    ax.set_title(title)

    # Annotate each cell, picking black or white text from the mapped cell luminance so
    # the label stays legible whichever colormap is used.
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if math.isnan(value):
                continue
            r, g, b, _ = image.cmap(image.norm(value))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.5 else "black"
            ax.text(j, i, f"{value:.3g}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(image, ax=ax, label=cbar_label)
    # fig.tight_layout()
    return fig


@app.cell
def _(METRICS, means, problem):
    # A bar chart and an encoder x decoder grid per metric, keyed by output file stem.
    figures = {
        **{stem: plot_bars(means, stem, problem, label) for stem, (_, label) in METRICS.items()},
        **{
            f"{stem}_grid": plot_grid(means, stem, problem, label)
            for stem, (_, label) in METRICS.items()
        },
    }

    mo.vstack(
        [
            mo.md(f"### {problem}"),
            mo.hstack([fig for stem, fig in figures.items() if not stem.endswith("_grid")]),
            mo.hstack([fig for stem, fig in figures.items() if stem.endswith("_grid")]),
        ]
    )
    return (figures,)


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save figures")
    save_button
    return (save_button,)


@app.cell
def _(OUT_DIR, figures, problem, save_button, size_ui):
    mo.stop(not save_button.value, mo.md("Click **Save figures** to write the PDFs."))

    problem_dir = OUT_DIR / problem.lower()
    problem_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for _stem, _fig in figures.items():
        path = problem_dir / f"{_stem}_{size_ui.value}.pdf"
        save_figure(_fig, path)
        saved_paths.append(path)

    mo.md("Saved:\n" + "\n".join(f"- `{p}`" for p in saved_paths))
    return


if __name__ == "__main__":
    app.run()
