import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import yaml

    from core.models.nds.config import TrainConfig
    from core.models.nds.factory import build_model
    from figures.utils import figsize, project_root, set_thesis_style

    set_thesis_style()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Model parameters

    Trainable parameters per method, split into encoder and decoder, for the selected
    problem and size. Each method is built from its real training config, so the counts
    match the trained models. Random Destroy has no model (0 parameters).

    `model.encoder` and `model.decoder` partition the parameters exactly (feature
    extractors and positional encoding hold none). For the Split encoder the encoder bar
    includes both the one-time instance encoder and the per-step fusion block.
    """)
    return


@app.cell
def _():
    # Problem title -> method label -> {problem size: train config path (under the repo root)}.
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

    OUT_DIR = project_root() / "figures" / "out"
    return CONFIGS, OUT_DIR


@app.cell
def _(CONFIGS: dict[str, dict[str, dict[int, str]]]):
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
    mo.hstack([problem_ui, size_ui], justify="start", gap=1)
    return problem_ui, size_ui


@app.function
def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters())


@app.cell
def _(CONFIGS: dict[str, dict[str, dict[int, str]]], problem_ui, size_ui):
    def encoder_decoder_params(config_path: str) -> tuple[int, int]:
        config = TrainConfig.model_validate(
            yaml.safe_load((project_root() / config_path).read_text())
        )
        model = build_model(config.model)
        return count_parameters(model.encoder), count_parameters(model.decoder)

    size = size_ui.value
    # Method label -> (encoder params, decoder params), for the selected problem and size.
    params = {
        label: encoder_decoder_params(config_by_size[size])
        for label, config_by_size in CONFIGS[problem_ui.value].items()
        if config_by_size.get(size)
    }
    return (params,)


@app.function
def plot_params(params: dict[str, tuple[int, int]], title: str):
    """Horizontal stacked bars of encoder vs decoder parameter counts, methods on the y-axis."""
    labels = list(params)
    encoder = [params[label][0] for label in labels]
    decoder = [params[label][1] for label in labels]

    fig, ax = plt.subplots(figsize=figsize(1, aspect=0.55))
    positions = range(len(labels))
    encoder_bars = ax.barh(list(positions), encoder, color="tab:blue", label="Encoder")
    ax.barh(list(positions), decoder, left=encoder, color="tab:orange", label="Decoder")
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first method at the top

    # Annotate each segment with its count and the total at the bar end.
    for position, enc, dec in zip(positions, encoder, decoder):
        if enc > 0:
            ax.text(enc / 2, position, f"{enc:,}", ha="center", va="center", fontsize=7)
        if dec > 0:
            ax.text(enc + dec / 2, position, f"{dec:,}", ha="center", va="center", fontsize=7)
        ax.text(enc + dec, position, f"  {enc + dec:,}", ha="left", va="center", fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("Trainable parameters")
    vmax = max(encoder + decoder, default=1.0)
    ax.set_xlim(0, vmax * 2)  # headroom for the value labels
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, encoder_bars


@app.cell
def _(params, problem_ui, size_ui):
    fig, _ = plot_params(params, f"{problem_ui.value}{size_ui.value}")
    fig
    return (fig,)


@app.cell
def _():
    save_button = mo.ui.run_button(label="Save figure")
    save_button
    return (save_button,)


@app.cell
def _(OUT_DIR, fig, problem_ui, save_button, size_ui):
    mo.stop(not save_button.value, mo.md("Click **Save figure** to write the PDF."))

    problem_dir = OUT_DIR / problem_ui.value.lower()
    problem_dir.mkdir(parents=True, exist_ok=True)
    path = problem_dir / f"params_{size_ui.value}.pdf"
    fig.savefig(path, bbox_inches="tight")
    mo.md(f"Saved to `{path}`")
    return


if __name__ == "__main__":
    app.run()
