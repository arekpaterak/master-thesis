import marimo

__generated_with = "0.22.5"
app = marimo.App(width="columns")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import torch
    import torch.nn
    import yaml
    from dotenv import load_dotenv

    import wandb
    from core.models.utils import save_checkpoint

    load_dotenv(".env")


@app.cell
def _():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device
    return (device,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(TrainConfig):
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

            _usage = Table(title="AM Training - CLI Usage")
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

        _resume_path = _cli_args.get("resume")
        resume_run_dir = Path(_resume_path) if _resume_path else None

        _config_path = _cli_args.get("config") or (
            str(resume_run_dir / "config.yml") if resume_run_dir else None
        )
        if _config_path:
            with open(_config_path) as _f:
                yaml_data = yaml.safe_load(_f) or {}

        for _key, _value in _cli_args.items():
            if _key in ("config", "resume"):
                continue
            _parts = _key.split(".")
            _node = yaml_data
            for _part in _parts[:-1]:
                _node = _node.setdefault(_part, {})
            _node[_parts[-1]] = _value
    else:
        resume_run_dir = None
    return env_config_widget, is_script_mode, resume_run_dir, yaml_data


@app.cell
def _(env_config_widget, is_script_mode):
    env_config_widget if not is_script_mode and not wandb.login() else None
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    ## Training Parameters
    """)
    return


@app.cell(hide_code=True)
def _(is_script_mode, params_accordion, run_button):
    mo.vstack([params_accordion, run_button]) if not is_script_mode else None
    return


@app.cell
def _():
    from core.models.attention_model.config import TrainConfig

    return (TrainConfig,)


@app.cell
def _(TrainConfig):
    defaults = TrainConfig()

    problem_widgets = mo.ui.dictionary(
        {
            "name": mo.ui.dropdown(
                ["tsp", "cvrp"], value=defaults.problem.name, label="Problem name"
            ),
            "size": mo.ui.number(
                10, 100, value=defaults.problem.size, step=1, label="Problem size"
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
            "num_layers": mo.ui.slider(
                1, 6, value=defaults.model.num_layers, step=1, label="Num layers"
            ),
        }
    )

    optimizer_widgets = mo.ui.dictionary(
        {
            "learning_rate": mo.ui.slider(
                1e-5, 5e-4, value=defaults.optimizer.learning_rate, step=1e-5, label="Learning rate"
            ),
        }
    )

    trainer_widgets = mo.ui.dictionary(
        {
            "epochs": mo.ui.slider(1, 100, value=defaults.trainer.epochs, step=1, label="Epochs"),
            "batches_per_epoch": mo.ui.number(
                100,
                10000,
                value=defaults.trainer.batches_per_epoch,
                step=100,
                label="Batches per epoch",
            ),
            "batch_size": mo.ui.slider(
                8, 512, value=defaults.trainer.batch_size, step=8, label="Batch size"
            ),
            "checkpoint_every_n_epochs": mo.ui.number(
                1,
                100,
                value=defaults.trainer.checkpoint_every_n_epochs,
                step=1,
                label="Checkpoint every N epochs",
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

    params_accordion = mo.accordion(
        {
            "Problem": problem_widgets,
            "Model": model_widgets,
            "Optimizer": optimizer_widgets,
            "Trainer": trainer_widgets,
            "Logging": logging_widgets,
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
    )


@app.cell(hide_code=True)
def _(
    TrainConfig,
    is_script_mode,
    logging_widgets,
    model_widgets,
    optimizer_widgets,
    problem_widgets,
    run_button,
    trainer_widgets,
    yaml_data,
):
    from core.models.attention_model.config import (
        LoggingConfig,
        ModelParams,
        TrainerParams,
    )
    from core.models.optimizer import OptimizerConfig
    from core.problems import make_problem_config

    if is_script_mode:
        config = TrainConfig.model_validate(yaml_data)
    else:
        mo.stop(
            run_button.value == 0,
            mo.md("*Click 'Start Training' to begin.*"),
        )
        config = TrainConfig(
            problem=make_problem_config(**problem_widgets.value),
            model=ModelParams(**model_widgets.value),
            optimizer=OptimizerConfig(**optimizer_widgets.value),
            trainer=TrainerParams(**trainer_widgets.value),
            logging=LoggingConfig(**logging_widgets.value),
        )
    return (config,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ## Model Setup
    """)
    return


@app.cell
def _(config):
    from core.problems.sampler import Sampler

    sampler = Sampler.from_config(config.problem)
    return (sampler,)


@app.cell
def _(config, device):
    from core.models.attention_model.model import AttentionModel

    model = AttentionModel(
        problem_name=config.problem.name,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
    ).to(device)

    model
    return (model,)


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    ## Training
    """)
    return


@app.cell
def _(config, model, resume_run_dir, sampler):
    from core.models.attention_model.module import AMModule
    from core.models.attention_model.statistical_test import OneSidedPairedTTestResult
    from core.models.attention_model.training import BatchSummary, EpochSummary
    from core.models.optimizer import build_optimizer
    from core.models.trainer import Trainer

    def _find_latest_checkpoint(run_dir: Path) -> tuple[Path, int] | None:
        checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        epoch = int(latest.stem.split("_")[1])
        return latest, epoch

    run_dir = resume_run_dir or (Path(__file__).parent / "runs" / f"am-{config.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if resume_run_dir is None:
        with open(run_dir / "config.yml", "w") as _f:
            yaml.safe_dump(config.model_dump(), _f, sort_keys=False)

    _wandb_id_file = run_dir / "wandb_run_id"
    if config.logging.wandb_project:
        if resume_run_dir is not None and _wandb_id_file.exists():
            wandb.init(
                project=config.logging.wandb_project,
                resume="must",
                id=_wandb_id_file.read_text().strip(),
            )
        else:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.run_name,
                config=config.model_dump(),
            )
            if wandb.run is not None:
                _wandb_id_file.write_text(wandb.run.id)

    def on_test_end(test: OneSidedPairedTTestResult):
        if config.logging.wandb_project:
            wandb.log(
                {
                    "eval/obj_model": test.obj_model_mean,
                    "eval/obj_baseline": test.obj_baseline_mean,
                    "eval/baseline_updated": test.improved,
                }
            )

    attention_module = AMModule(model, sampler, on_baseline_sync=on_test_end)
    optimizer = build_optimizer(config.optimizer, attention_module.parameters())

    initial_epoch = 0
    _checkpoint_result = _find_latest_checkpoint(run_dir)
    if _checkpoint_result is not None:
        _checkpoint_path, _latest_epoch = _checkpoint_result
        _ckpt = torch.load(_checkpoint_path, map_location="cpu", weights_only=True)
        attention_module.model.load_state_dict(_ckpt["model"])
        attention_module.baseline.load_state_dict(_ckpt["baseline"])
        optimizer.load_state_dict(_ckpt["optimizer"])
        initial_epoch = _latest_epoch + 1
    elif config.trainer.init_weights:
        _ckpt = torch.load(config.trainer.init_weights, map_location="cpu", weights_only=True)
        attention_module.model.load_state_dict(_ckpt.get("model", _ckpt))
        attention_module.baseline.load_state_dict(
            _ckpt["baseline"] if "baseline" in _ckpt else attention_module.model.state_dict()
        )

    with (
        mo.status.progress_bar(total=config.trainer.epochs, title="Epochs") as epoch_bar,
        mo.status.progress_bar(
            total=config.trainer.epochs * config.trainer.batches_per_epoch, title="Batches"
        ) as step_bar,
    ):

        def on_batch_train_end(summary: BatchSummary):
            step_bar.update(subtitle=f"Loss: {summary.loss:.4f}")
            if config.logging.wandb_project:
                wandb.log({f"train/{k}": v for k, v in summary.model_dump().items()})

        def on_epoch_train_end(summary: EpochSummary):
            epoch_bar.update(subtitle=f"Avg. Loss: {summary.loss:.4f}")
            is_last = summary.epoch == config.trainer.epochs - 1
            is_checkpoint = (summary.epoch + 1) % config.trainer.checkpoint_every_n_epochs == 0
            if is_last or is_checkpoint:
                checkpoints_path = run_dir / "checkpoints"
                checkpoints_path.mkdir(exist_ok=True)
                save_checkpoint(
                    path=checkpoints_path / f"epoch_{summary.epoch:04d}.pt",
                    model=attention_module.model,
                    optimizer=optimizer,
                    epoch=summary.epoch,
                    config=config,
                    extra={"baseline": attention_module.baseline.state_dict()},
                )

        Trainer(
            max_epochs=config.trainer.epochs,
            training_batches_per_epoch=config.trainer.batches_per_epoch,
            batch_size=config.trainer.batch_size,
            on_batch_train_end=on_batch_train_end,
            on_epoch_train_end=[on_epoch_train_end],
            initial_epoch=initial_epoch,
        ).fit(module=attention_module, optimizer=optimizer, instance_sampler=sampler)

    if config.logging.wandb_project:
        wandb.finish()

    mo.md("**Training complete.**")
    return


if __name__ == "__main__":
    app.run()
