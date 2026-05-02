import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")

with app.setup:
    import torch
    import torch.nn


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notebook Description

    ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Environment Setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv

    load_dotenv(".env")
    return (mo,)


@app.cell
def _(env_config, is_script_mode, wandb):
    env_config if not is_script_mode and not wandb.login() else None
    return


@app.cell
def _(ModelParams, mo, wandb):
    import sys

    from wigglystuff import EnvConfig

    is_script_mode = mo.app_meta().mode == "script"

    env_config = mo.ui.anywidget(
        EnvConfig(
            {
                "WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True),
            }
        )
    )

    # If run in script mode and no CLI arguments passed, show help
    if is_script_mode and not mo.cli_args():
        from rich.console import Console
        from rich.table import Table

        table = Table(title="CLI Options")
        table.add_column("Flag", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Default", style="yellow")
        table.add_column("Description")

        for name, field in ModelParams.model_fields.items():
            flag = f"--{name.replace('_', '-')}"
            type_name = (
                field.annotation.__name__
                if hasattr(field.annotation, "__name__")
                else str(field.annotation)
            )
            table.add_row(flag, type_name, str(field.default), field.description or "")

        Console().print(table)
        sys.exit(0)
    return env_config, is_script_mode


@app.cell
def _():
    import wandb

    return (wandb,)


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Parameters
    """)
    return


@app.cell
def _(params_form):
    params_form
    return


@app.cell
def _():
    import hashlib
    import json

    from pydantic import BaseModel, Field, computed_field

    class ModelParams(BaseModel):
        problem_name: str = Field(default="tsp", description="Problem to solve (tsp or cvrp).")
        problem_size: int = Field(default=20, description="Number of nodes in the problem.")
        epochs: int = Field(default=1, description="Number of training epochs.")
        batches_per_epoch: int = Field(default=2500, description="Batches per epoch.")
        batch_size: int = Field(default=32, description="Training batch size.")
        learning_rate: float = Field(default=1e-4, description="Learning rate for AdamW.")
        d_h: int = Field(default=128, description="Hidden dimension.")
        num_heads: int = Field(default=8, description="Number of attention heads.")
        num_layers: int = Field(default=3, description="Number of attention layers.")
        wandb_project: str = Field(
            default="attention-model", description="W&B project name (empty to skip)."
        )

        @computed_field
        @property
        def run_name(self) -> str:
            parts = [
                f"{self.problem_name}{self.problem_size}",
                f"e{self.epochs}",
                f"bs{self.batch_size}",
                f"lr{self.learning_rate:.0e}",
            ]
            params_dict = {
                "problem_name": self.problem_name,
                "problem_size": self.problem_size,
                "epochs": self.epochs,
                "batches_per_epoch": self.batches_per_epoch,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "d_h": self.d_h,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
            }
            h = hashlib.md5(json.dumps(params_dict, sort_keys=True).encode()).hexdigest()[:6]
            return "-".join(parts) + f"-{h}"

    return (ModelParams,)


@app.cell
def _(mo):
    params_form = (
        mo.md("""
    ## Model parameters
    {problem_name}
    {problem_size}
    {epochs}
    {batches_per_epoch}
    {batch_size}
    {learning_rate}
    {d_h}
    {num_heads}
    {num_layers}
    """)
        .batch(
            problem_name=mo.ui.dropdown(["tsp", "cvrp"], value="tsp", label="Problem Name"),
            problem_size=mo.ui.number(10, 100, value=20, step=1, label="Problem Size"),
            epochs=mo.ui.slider(1, 100, value=1, step=1, label="epochs"),
            batches_per_epoch=mo.ui.number(
                100, 10000, value=2500, step=100, label="batches_per_epoch"
            ),
            batch_size=mo.ui.slider(8, 512, value=32, step=8, label="batch size"),
            learning_rate=mo.ui.slider(1e-5, 5e-4, value=1e-4, step=1e-5, label="learning rate"),
            d_h=mo.ui.dropdown([64, 128, 256, 512], value=128, label="d_h"),
            num_heads=mo.ui.slider(1, 16, value=8, step=1, label="num_heads"),
            num_layers=mo.ui.slider(1, 6, value=3, step=1, label="num_layers"),
        )
        .form()
    )
    return (params_form,)


@app.cell
def _(ModelParams, is_script_mode, mo, params_form):
    mo.stop(
        not is_script_mode and params_form.value is None,
        mo.md("*Submit the form to start training.*"),
    )

    if is_script_mode:
        model_params = ModelParams(**{k.replace("-", "_"): v for k, v in mo.cli_args().items()})
    else:
        model_params = ModelParams(**params_form.value)
    return (model_params,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Setup
    """)
    return


@app.cell
def _(model_params):
    from core.problems.cvrp import CVRPConfig, CVRPSampler
    from core.problems.tsp import TSPConfig, TSPSampler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_params.problem_name == "tsp":
        sampler = TSPSampler(TSPConfig(num_locations=model_params.problem_size))
    elif model_params.problem_name == "cvrp":
        sampler = CVRPSampler(CVRPConfig.for_n(model_params.problem_size))
    else:
        raise ValueError(f"Unknown problem name: {model_params.problem_name}")
    return device, sampler


@app.cell
def _(device):
    device
    return


@app.cell
def _(device, model_params):
    from core.models.attention_model.model import AttentionModel

    model = AttentionModel(
        problem_name=model_params.problem_name,
        d_h=model_params.d_h,
        num_heads=model_params.num_heads,
        num_layers=model_params.num_layers,
    ).to(device)

    model
    return (model,)


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training
    """)
    return


@app.cell
def _(mo, model, model_params, sampler, wandb):
    from core.models.attention_model.training import (
        EpochSummary,
        configure_train_epoch,
        prepare_baseline,
    )
    from core.models.train import Trainer

    if model_params.wandb_project:
        wandb.init(
            project=model_params.wandb_project,
            name=model_params.run_name,
            config=model_params.model_dump(),
        )

    with (
        mo.status.progress_bar(total=model_params.epochs, title="Epochs") as epoch_bar,
        mo.status.progress_bar(
            total=model_params.epochs * model_params.batches_per_epoch, title="Batches"
        ) as step_bar,
    ):

        def on_batch_end(step_idx: int, loss: float):
            step_bar.update(subtitle=f"Loss: {loss:.4f}")

        def on_epoch_end(summary: EpochSummary):
            if model_params.wandb_project:
                wandb.log(summary.asdict())
            epoch_bar.update(subtitle=f"Avg Loss: {summary.loss:.4f}")

        baseline = prepare_baseline(model)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_params.learning_rate,
            weight_decay=0.0,
        )
        train_epoch = configure_train_epoch(
            instance_sampler=sampler,
            batches_per_epoch=model_params.batches_per_epoch,
            batch_size=model_params.batch_size,
            on_batch_end=on_batch_end,
        )

        _ = Trainer(
            max_epochs=model_params.epochs,
            train_epoch=train_epoch,
            on_epoch_end=on_epoch_end,
        ).train(model=model, optimizer=optimizer, baseline=baseline)

    if model_params.wandb_project:
        wandb.finish()

    mo.md("**Training complete.**")
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
