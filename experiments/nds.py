import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")

with app.setup:
    import matplotlib.pyplot as plt
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
def _():
    from core.models.nds.config import TrainConfig

    return (TrainConfig,)


@app.cell(hide_code=True)
def _(TrainConfig, mo):
    import sys

    import yaml
    from wigglystuff import EnvConfig

    is_script_mode = mo.app_meta().mode == "script"

    env_config_widget = mo.ui.anywidget(
        EnvConfig(
            {
                "WANDB_API_KEY": lambda k: wandb.login(key=k, verify=True),
            }
        )
    )

    yaml_data = {}
    if is_script_mode:
        _cli_args = mo.cli_args()

        if not _cli_args:
            from rich.console import Console
            from rich.table import Table

            _usage = Table(title="NDS Training — CLI Usage")
            _usage.add_column("Flag", style="cyan")
            _usage.add_column("Description")
            _usage.add_row("--config <path>", "Path to YAML config file")
            _usage.add_row(
                "--<section>.<field> <value>",
                "Override a nested config field, e.g. --trainer.epochs 5",
            )
            Console().print(_usage)

            _defaults = TrainConfig()
            _fields = Table(title="Config fields and defaults")
            _fields.add_column("Key", style="cyan")
            _fields.add_column("Default", style="yellow")
            for _section, _params in _defaults.model_dump(exclude={"run_name", "logging"}).items():
                for _field, _value in _params.items():
                    _fields.add_row(f"{_section}.{_field}", str(_value))
            Console().print(_fields)

            sys.exit(0)

        _config_path = _cli_args.get("config")
        if _config_path:
            with open(_config_path) as _f:
                yaml_data = yaml.safe_load(_f) or {}

        for _key, _value in _cli_args.items():
            if _key == "config":
                continue
            _parts = _key.split(".")
            _node = yaml_data
            for _part in _parts[:-1]:
                _node = _node.setdefault(_part, {})
            _node[_parts[-1]] = _value
    return env_config_widget, is_script_mode, yaml_data


@app.cell(hide_code=True)
def _(env_config_widget, is_script_mode):
    env_config_widget if not is_script_mode and not wandb.login() else None
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Parameters
    """)
    return


@app.cell(hide_code=True)
def _(is_script_mode, mo, params_accordion, run_button):
    mo.vstack([params_accordion, run_button]) if not is_script_mode else None
    return


@app.cell(hide_code=True)
def _(
    TrainConfig,
    is_script_mode,
    logging_widgets,
    mo,
    model_widgets,
    optimizer_widgets,
    problem_widgets,
    run_button,
    trainer_widgets,
    validation_widgets,
    yaml_data,
):
    from core.models.nds.config import (
        LoggingConfig,
        ModelParams,
        OptimizerParams,
        ProblemParams,
        TrainerParams,
        ValidationConfig,
    )

    if is_script_mode:
        config = TrainConfig.model_validate(yaml_data)
    else:
        mo.stop(
            run_button.value == 0,
            mo.md("*Click 'Start Training' to begin.*"),
        )
        config = TrainConfig(
            problem=ProblemParams(**problem_widgets.value),
            model=ModelParams(**model_widgets.value),
            optimizer=OptimizerParams(**optimizer_widgets.value),
            trainer=TrainerParams(**trainer_widgets.value),
            logging=LoggingConfig(**logging_widgets.value),
            validation=ValidationConfig(**validation_widgets.value),
        )
    return (config,)


@app.cell(hide_code=True)
def _(TrainConfig, mo):
    defaults = TrainConfig()

    problem_widgets = mo.ui.dictionary(
        {
            "name": mo.ui.dropdown(
                ["cvrp", "tsp"], value=defaults.problem.name, label="Problem name"
            ),
            "size": mo.ui.number(
                10, 200, value=defaults.problem.size, step=1, label="Problem size"
            ),
        }
    )

    model_widgets = mo.ui.dictionary(
        {
            "hidden_dim": mo.ui.dropdown(
                [64, 128, 256, 512], value=defaults.model.hidden_dim, label="Hidden dim"
            ),
            "num_heads": mo.ui.slider(
                1, 16, value=defaults.model.num_heads, step=1, label="Num heads"
            ),
            "ff_dim": mo.ui.dropdown(
                [128, 256, 512, 1024], value=defaults.model.ff_dim, label="FF dim"
            ),
            "poly_dim": mo.ui.dropdown(
                [64, 128, 256, 512], value=defaults.model.poly_dim, label="Poly dim"
            ),
            "seed_dim": mo.ui.dropdown(
                [4, 8, 10, 16, 32], value=defaults.model.seed_dim, label="Seed dim"
            ),
        }
    )

    optimizer_widgets = mo.ui.dictionary(
        {
            "learning_rate": mo.ui.slider(
                1e-5,
                5e-4,
                value=defaults.optimizer.learning_rate,
                step=1e-5,
                label="Learning rate",
            ),
        }
    )

    trainer_widgets = mo.ui.dictionary(
        {
            "epochs": mo.ui.slider(1, 100, value=defaults.trainer.epochs, step=1, label="Epochs"),
            "instances_per_epoch": mo.ui.number(
                100,
                1500,
                value=defaults.trainer.instances_per_epoch,
                step=100,
                label="Instances per epoch",
            ),
            "batch_size": mo.ui.slider(
                8, 512, value=defaults.trainer.batch_size, step=8, label="Batch size"
            ),
            "rollouts_per_instance": mo.ui.slider(
                1,
                256,
                value=defaults.trainer.rollouts_per_instance,
                step=1,
                label="Rollouts/solution",
            ),
            "num_initial_improvement_steps": mo.ui.slider(
                0,
                10,
                value=defaults.trainer.num_initial_improvement_steps,
                step=1,
                label="Initial steps",
            ),
            "num_main_improvement_steps": mo.ui.slider(
                1,
                50,
                value=defaults.trainer.num_main_improvement_steps,
                step=1,
                label="Main steps",
            ),
            "deconstruction_size": mo.ui.slider(
                1,
                20,
                value=defaults.trainer.deconstruction_size,
                step=1,
                label="Deconstruction size",
            ),
        }
    )

    logging_widgets = mo.ui.dictionary(
        {
            "wandb_project": mo.ui.text(
                value=defaults.logging.wandb_project, label="W&B project (empty to skip)"
            ),
        }
    )

    validation_widgets = mo.ui.dictionary(
        {
            "num_instances": mo.ui.number(
                32, 5000, value=defaults.validation.num_instances, step=32, label="Num instances"
            ),
            "batch_size": mo.ui.slider(
                8, 512, value=defaults.validation.batch_size, step=8, label="Batch size"
            ),
            "num_improvement_steps": mo.ui.slider(
                1,
                200,
                value=defaults.validation.num_improvement_steps,
                step=1,
                label="Improvement steps",
            ),
            "every_n_epochs": mo.ui.number(
                1, 100, value=defaults.validation.every_n_epochs, step=1, label="Every N epochs"
            ),
            "seed": mo.ui.number(0, 99999, value=defaults.validation.seed, step=1, label="Seed"),
            "plot_enabled": mo.ui.checkbox(
                value=defaults.validation.plot_enabled, label="Plot enabled"
            ),
            "plot_every_n_lns_steps": mo.ui.slider(
                1,
                20,
                value=defaults.validation.plot_every_n_improvement_steps,
                step=1,
                label="Plot every N steps",
            ),
            "plot_dir": mo.ui.text(value=defaults.validation.plot_dir, label="Plot directory"),
            "plot_save_to_disk": mo.ui.checkbox(
                value=defaults.validation.plot_save_to_disk, label="Save to disk"
            ),
            "plot_show_in_marimo": mo.ui.checkbox(
                value=defaults.validation.plot_show_in_marimo, label="Show in marimo"
            ),
        }
    )

    params_accordion = mo.accordion(
        {
            "Problem": problem_widgets,
            "Model": model_widgets,
            "Optimizer": optimizer_widgets,
            "Trainer": trainer_widgets,
            "Logging": logging_widgets,
            "Validation": validation_widgets,
        }
    )

    run_button = mo.ui.button(label="Start Training", value=0, on_click=lambda count: count + 1)
    return (
        logging_widgets,
        model_widgets,
        optimizer_widgets,
        params_accordion,
        problem_widgets,
        run_button,
        trainer_widgets,
        validation_widgets,
    )


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem Setup
    """)
    return


@app.cell
def _(config):
    from core.problems.cvrp import CVRPConfig, CVRPSampler
    from core.problems.tsp import TSPConfig, TSPSampler

    match config.problem.name:
        case "cvrp":
            sampler = CVRPSampler(CVRPConfig.for_n(config.problem.size))
            val_sampler = CVRPSampler(
                CVRPConfig.for_n(config.problem.size), seed=config.validation.seed
            )
        case "tsp":
            sampler = TSPSampler(TSPConfig(num_locations=config.problem.size))
            val_sampler = TSPSampler(
                TSPConfig(num_locations=config.problem.size), seed=config.validation.seed
            )
        case _:
            raise ValueError(f"Unsupported problem name: {config.problem.name}")
    return sampler, val_sampler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Setup
    """)
    return


@app.cell
def _(config, device):
    from core.models.nds.model import NDSModel

    model = NDSModel(
        problem_name=config.problem.name,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        ff_dim=config.model.ff_dim,
        poly_dim=config.model.poly_dim,
        seed_dim=config.model.seed_dim,
        with_neighbours_encoding=config.model.with_neighbours_encoding,
        with_tour_encoding=config.model.with_tour_encoding,
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
def _(config, device, mo, model, sampler, val_sampler):
    from pathlib import Path

    import torch.optim as optim

    from core.models.nds.evaluation import EpochValidationSummary, plot_lns_progress
    from core.models.nds.functions import (
        apply_rollout,
        greedy_cvrp_repair,
        greedy_tsp_repair,
    )
    from core.models.nds.module import NDSModule
    from core.models.nds.training import BatchTrainSummary, EpochTrainSummary
    from core.models.trainer import Trainer
    from core.problems.cvrp.solve import one_customer_per_route
    from core.problems.tsp.solve import random_permutation_graph

    if config.logging.wandb_project:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.run_name,
            config=config.model_dump(),
        )

    match config.problem.name:
        case "cvrp":
            generate_start_solution = one_customer_per_route
            _repair = greedy_cvrp_repair
        case "tsp":
            generate_start_solution = random_permutation_graph
            _repair = greedy_tsp_repair

    nds_module = NDSModule(
        config=config,
        model=model,
        generate_start_solution=generate_start_solution,
        apply_rollout=apply_rollout,
        repair=_repair,
    )

    with (
        mo.status.progress_bar(total=config.trainer.epochs, title="Epochs") as epoch_bar,
        mo.status.progress_bar(
            total=config.trainer.epochs * config.trainer.batches_per_epoch, title="Batches (steps)"
        ) as step_bar,
    ):

        def on_step_end(summary: BatchTrainSummary):
            step_bar.update(subtitle=f"Loss: {summary.loss:.4f}")
            if config.logging.wandb_project:
                wandb.log({f"train/{k}": v for k, v in summary.model_dump().items()})

        def on_epoch_end(summary: EpochTrainSummary):
            epoch_bar.update(subtitle=f"Avg. Loss: {summary.loss:.4f}")

        def on_validation_end(val_summary: EpochValidationSummary):
            if config.logging.wandb_project:
                wandb.log(
                    {
                        "val/objective_initial": val_summary.objective_initial,
                        "val/objective_final": val_summary.objective_final,
                        "val/improvement": val_summary.improvement,
                        "val/relative_improvement": val_summary.relative_improvement,
                    }
                )
            if config.validation.plot_enabled:
                fig = plot_lns_progress(
                    val_summary.snapshots, val_summary.step_labels, epoch=val_summary.epoch
                )
                if config.validation.plot_save_to_disk:
                    plot_path = Path(config.validation.plot_dir) / config.run_name
                    plot_path.mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        plot_path / f"epoch_{val_summary.epoch:04d}.png",
                        dpi=100,
                        bbox_inches="tight",
                    )
                if config.validation.plot_show_in_marimo:
                    pass
                plt.close(fig)

        optimizer = optim.AdamW(
            nds_module.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

        _ = Trainer(
            max_epochs=config.trainer.epochs,
            training_instances=config.trainer.instances_per_epoch,
            batch_size=config.trainer.batch_size,
            on_batch_train_end=on_step_end,
            on_epoch_train_end=[on_epoch_end],
            validation_instances=config.validation.num_instances,
            validation_batch_size=config.validation.batch_size,
            validate_every_n_epochs=config.validation.every_n_epochs,
            on_epoch_validation_end=[on_validation_end],
        ).fit(
            model=nds_module,
            optimizer=optimizer,
            instance_sampler=lambda n: sampler(n).to(device),
            validation_instance_sampler=lambda n: val_sampler(n).to(device),
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
