import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")

with app.setup:
    import torch
    import torch.nn

    import wandb


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv

    load_dotenv(".env")
    return (mo,)


@app.cell
def _(Config, mo):
    import sys

    from wigglystuff import EnvConfig

    is_script_mode = mo.app_meta().mode == "script"

    env_config_widget = mo.ui.anywidget(
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

        # Flatten nested config for CLI
        config = Config()
        cli_fields = {
            "problem_name": (
                str,
                config.problem.name,
                "Problem to solve (currently only cvrp is supported).",
            ),
            "problem_size": (int, config.problem.size, "Number of customers."),
            "epochs": (int, config.trainer.epochs, "Number of training epochs."),
            "batches_per_epoch": (int, config.trainer.batches_per_epoch, "Batches per epoch."),
            "batch_size": (int, config.trainer.batch_size, "Training batch size."),
            "learning_rate": (float, config.optimizer.learning_rate, "Learning rate."),
            "hidden_dim": (int, config.model.hidden_dim, "Hidden dimension."),
            "num_heads": (int, config.model.num_heads, "Number of attention heads."),
            "ff_dim": (int, config.model.ff_dim, "Feed-forward dimension."),
            "poly_dim": (int, config.model.poly_dim, "Polynomial dimension."),
            "seed_dim": (int, config.model.seed_dim, "Seed (latent) dimension."),
            "num_initial_improvement_steps": (
                int,
                config.trainer.num_initial_improvement_steps,
                "Initial improvement steps before training.",
            ),
            "num_main_improvement_steps": (
                int,
                config.trainer.num_main_improvement_steps,
                "Number of improvement steps per batch.",
            ),
            "rollouts_per_solution": (
                int,
                config.trainer.rollouts_per_instance,
                "Rollouts per solution.",
            ),
            "deconstruction_size": (
                int,
                config.trainer.deconstruction_size,
                "Number of customers removed per rollout.",
            ),
            "wandb_project": (
                str,
                config.logging.wandb_project,
                "W&B project name (empty to skip).",
            ),
        }

        for name, (type_name, default, description) in cli_fields.items():
            flag = f"--{name.replace('_', '-')}"
            table.add_row(flag, type_name.__name__, str(default), description)

        Console().print(table)
        sys.exit(0)
    return env_config_widget, is_script_mode


@app.cell
def _(env_config_widget, is_script_mode):
    env_config_widget if not is_script_mode and not wandb.login() else None
    return


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
    from core.models.nds.config import TrainConfig

    return (TrainConfig,)


@app.cell
def _(Config, mo):
    defaults = Config()
    params_form = (
        mo.md(
            """
    {problem_name}
    {problem_size}
    {epochs}
    {batches_per_epoch}
    {batch_size}
    {learning_rate}
    {hidden_dim}
    {num_heads}
    {ff_dim}
    {poly_dim}
    {seed_dim}
    {num_main_improvement_steps}
    {rollouts_per_solution}
    {num_initial_improvement_steps}
    {deconstruction_size}
    """
        )
        .batch(
            problem_name=mo.ui.dropdown(
                ["cvrp"], value=defaults.problem.name, label="Problem name"
            ),
            problem_size=mo.ui.number(
                10, 200, value=defaults.problem.size, step=1, label="Problem size"
            ),
            epochs=mo.ui.slider(1, 100, value=defaults.trainer.epochs, step=1, label="Epochs"),
            batches_per_epoch=mo.ui.number(
                10,
                10000,
                value=defaults.trainer.batches_per_epoch,
                step=10,
                label="Batches per epoch",
            ),
            batch_size=mo.ui.slider(
                8, 512, value=defaults.trainer.batch_size, step=8, label="Batch size"
            ),
            learning_rate=mo.ui.slider(
                1e-5, 5e-4, value=defaults.optimizer.learning_rate, step=1e-5, label="Learning rate"
            ),
            hidden_dim=mo.ui.dropdown(
                [64, 128, 256, 512], value=defaults.model.hidden_dim, label="Hidden dim"
            ),
            num_heads=mo.ui.slider(
                1, 16, value=defaults.model.num_heads, step=1, label="Num heads"
            ),
            ff_dim=mo.ui.dropdown(
                [128, 256, 512, 1024], value=defaults.model.ff_dim, label="FF dim"
            ),
            poly_dim=mo.ui.dropdown(
                [64, 128, 256, 512], value=defaults.model.poly_dim, label="Poly dim"
            ),
            seed_dim=mo.ui.dropdown(
                [4, 8, 10, 16, 32], value=defaults.model.seed_dim, label="Seed dim"
            ),
            num_main_improvement_steps=mo.ui.slider(
                1, 10, value=defaults.trainer.num_main_improvement_steps, step=1, label="Main steps"
            ),
            rollouts_per_solution=mo.ui.slider(
                1,
                16,
                value=defaults.trainer.rollouts_per_instance,
                step=1,
                label="Rollouts/solution",
            ),
            num_initial_improvement_steps=mo.ui.slider(
                0,
                10,
                value=defaults.trainer.num_initial_improvement_steps,
                step=1,
                label="Initial steps",
            ),
            deconstruction_size=mo.ui.slider(
                1,
                20,
                value=defaults.trainer.deconstruction_size,
                step=1,
                label="Deconstruction size",
            ),
        )
        .form()
    )
    return (params_form,)


@app.cell
def _(Config, is_script_mode, mo, params_form):
    from core.models.nds.config import (
        LoggingConfig,
        ModelParams,
        OptimizerParams,
        ProblemParams,
        TrainerParams,
    )

    mo.stop(
        not is_script_mode and params_form.value is None,
        mo.md("*Submit the form to start training.*"),
    )

    if is_script_mode:
        form_data = {k.replace("-", "_"): v for k, v in mo.cli_args().items()}
    else:
        form_data = params_form.value

    # Reconstruct nested config from flat form data
    config = Config(
        problem=ProblemParams(
            name=form_data.get("problem_name", "cvrp"),
            size=form_data.get("problem_size", 20),
        ),
        model=ModelParams(
            hidden_dim=form_data.get("hidden_dim", 128),
            num_heads=form_data.get("num_heads", 8),
            ff_dim=form_data.get("ff_dim", 512),
            poly_dim=form_data.get("poly_dim", 256),
            seed_dim=form_data.get("seed_dim", 10),
        ),
        optimizer=OptimizerParams(
            learning_rate=form_data.get("learning_rate", 1e-4),
        ),
        trainer=TrainerParams(
            epochs=form_data.get("epochs", 1),
            batches_per_epoch=form_data.get("batches_per_epoch", 500),
            batch_size=form_data.get("batch_size", 256),
            num_initial_improvement_steps=form_data.get("num_initial_improvement_steps", 0),
            num_main_improvement_steps=form_data.get("num_main_improvement_steps", 2),
            rollouts_per_instance=form_data.get("rollouts_per_solution", 4),
            deconstruction_size=form_data.get("deconstruction_size", 5),
        ),
        logging=LoggingConfig(
            wandb_project=form_data.get("wandb_project", "nds"),
        ),
    )
    return (config,)


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Setup
    """)
    return


@app.cell
def _(config):
    from core.problems.cvrp import CVRPConfig, CVRPSampler

    if config.problem.name != "cvrp":
        raise ValueError(f"Unsupported problem name: {config.problem.name}")

    sampler = CVRPSampler(CVRPConfig.for_n(config.problem.size))
    return (sampler,)


@app.cell
def _(device, config):
    from core.models.nds.model import NDSModel

    model = NDSModel(
        problem_name=config.problem.name,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        ff_dim=config.model.ff_dim,
        poly_dim=config.model.poly_dim,
        seed_dim=config.model.seed_dim,
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
def _(device, mo, model, config, sampler):
    import torch.optim as optim

    from core.models.nds.functions import (
        apply_rollout,
        greedy_cvrp_repair,
        noop_improvement_step,
    )
    from core.models.nds.training import BatchSummary, EpochSummary, configure_training_step
    from core.models.trainer import Trainer
    from core.problems.cvrp.solve import one_customer_per_route

    if config.logging.wandb_project:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.run_name,
            config=config.model_dump(),
        )

    def generate_start_solution(instance):
        return one_customer_per_route(instance)

    with (
        mo.status.progress_bar(total=config.trainer.epochs, title="Epochs") as epoch_bar,
        mo.status.progress_bar(
            total=config.trainer.epochs * config.trainer.batches_per_epoch, title="Batches"
        ) as step_bar,
    ):

        def on_step_end(summary: BatchSummary):
            step_bar.update(subtitle=f"Loss: {summary.loss:.4f}")
            if config.logging.wandb_project:
                wandb.log({f"train/{k}": v for k, v in summary.model_dump().items()})

        def on_epoch_end(summary: EpochSummary):
            epoch_bar.update(subtitle=f"Avg. Loss: {summary.loss:.4f}")

        optimizer = optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)
        train_step = configure_training_step(
            num_main_improvement_steps=config.trainer.num_main_improvement_steps,
            rollouts_per_solution=config.trainer.rollouts_per_instance,
            num_initial_improvement_steps=config.trainer.num_initial_improvement_steps,
            deconstruction_size=config.trainer.deconstruction_size,
            generate_start_solution=generate_start_solution,
            improvement_step=noop_improvement_step,
            apply_rollout=apply_rollout,
            repair=greedy_cvrp_repair,
        )

        _ = Trainer(
            max_epochs=config.trainer.epochs,
            batches_per_epoch=config.trainer.batches_per_epoch,
            batch_size=config.trainer.batch_size,
            training_step=train_step,
            aggregate_epoch=EpochSummary.from_step_summaries,
            on_step_end=on_step_end,
            on_epoch_end=[on_epoch_end],
        ).fit(
            model=model,
            optimizer=optimizer,
            instance_sampler=lambda batch_size: sampler(batch_size).to(device),
        )

    if config.logging.wandb_project:
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
