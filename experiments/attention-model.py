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
def _(ModelParams, mo):
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

            _usage = Table(title="AM Training - CLI Usage")
            _usage.add_column("Flag", style="cyan")
            _usage.add_column("Description")
            _usage.add_row("--config <path>", "Path to YAML config file")
            _usage.add_row("--<field> <value>", "Override a config field, e.g. --epochs 10")
            Console().print(_usage)

            _fields = Table(title="Config fields and defaults")
            _fields.add_column("Key", style="cyan")
            _fields.add_column("Default", style="yellow")
            _fields.add_column("Description")
            for _name, _field in ModelParams.model_fields.items():
                _fields.add_row(_name, str(_field.default), _field.description or "")
            Console().print(_fields)

            sys.exit(0)

        from pathlib import Path

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
            yaml_data[_key.replace("-", "_")] = _value
    else:
        resume_run_dir = None
    return env_config_widget, is_script_mode, resume_run_dir, yaml_data


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
    import hashlib
    import json

    from pydantic import BaseModel, Field, computed_field

    class ModelParams(BaseModel):
        problem_name: str = Field(default="tsp", description="Problem to solve (tsp or cvrp).")
        problem_size: int = Field(default=20, description="Number of nodes in the problem.")
        epochs: int = Field(default=1, description="Number of training epochs.")
        batches_per_epoch: int = Field(default=500, description="Batches per epoch.")
        batch_size: int = Field(default=256, description="Training batch size.")
        learning_rate: float = Field(default=1e-4, description="Learning rate.")
        hidden_dim: int = Field(default=128, description="Hidden dimension.")
        num_heads: int = Field(default=8, description="Number of attention heads.")
        num_layers: int = Field(default=3, description="Number of attention layers.")
        wandb_project: str = Field(
            default="attention-model", description="W&B project name (empty to skip)."
        )
        checkpoint_every_n_epochs: int = Field(
            default=10,
            description="Save a checkpoint every N epochs (always saves the last epoch).",
        )
        init_weights: str | None = Field(
            default=None, description="Path to a checkpoint to load model weights from (new run)."
        )

        @computed_field
        @property
        def run_name(self) -> str:
            parts = [
                f"{self.problem_name}{self.problem_size}",
            ]
            params_dict = {
                "problem_name": self.problem_name,
                "problem_size": self.problem_size,
                "epochs": self.epochs,
                "batches_per_epoch": self.batches_per_epoch,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
            }
            h = hashlib.md5(json.dumps(params_dict, sort_keys=True).encode()).hexdigest()[:6]
            return "-".join(parts) + f"-{h}"

    return (ModelParams,)


@app.cell
def _(ModelParams, mo):
    defaults = ModelParams()
    params_form = (
        mo.md("""
    {problem_name}
    {problem_size}
    {epochs}
    {batches_per_epoch}
    {batch_size}
    {learning_rate}
    {hidden_dim}
    {num_heads}
    {num_layers}
    {checkpoint_every_n_epochs}
    """)
        .batch(
            problem_name=mo.ui.dropdown(
                ["tsp", "cvrp"], value=defaults.problem_name, label="Problem name"
            ),
            problem_size=mo.ui.number(
                10, 100, value=defaults.problem_size, step=1, label="Problem size"
            ),
            epochs=mo.ui.slider(1, 100, value=defaults.epochs, step=1, label="Epochs"),
            batches_per_epoch=mo.ui.number(
                100, 10000, value=defaults.batches_per_epoch, step=100, label="Batches per epoch"
            ),
            batch_size=mo.ui.slider(8, 512, value=defaults.batch_size, step=8, label="Batch size"),
            learning_rate=mo.ui.slider(
                1e-5, 5e-4, value=defaults.learning_rate, step=1e-5, label="Learning rate"
            ),
            hidden_dim=mo.ui.dropdown(
                [64, 128, 256, 512], value=defaults.hidden_dim, label="Hidden dim"
            ),
            num_heads=mo.ui.slider(1, 16, value=defaults.num_heads, step=1, label="Num heads"),
            num_layers=mo.ui.slider(1, 6, value=defaults.num_layers, step=1, label="Num layers"),
            checkpoint_every_n_epochs=mo.ui.number(
                1,
                100,
                value=defaults.checkpoint_every_n_epochs,
                step=1,
                label="Checkpoint every N epochs",
            ),
        )
        .form()
    )
    return (params_form,)


@app.cell
def _(ModelParams, is_script_mode, mo, params_form, yaml_data):
    mo.stop(
        not is_script_mode and params_form.value is None,
        mo.md("*Submit the form to start training.*"),
    )

    if is_script_mode:
        model_params = ModelParams.model_validate(yaml_data)
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

    if model_params.problem_name == "tsp":
        sampler = TSPSampler(TSPConfig(num_locations=model_params.problem_size))
    elif model_params.problem_name == "cvrp":
        sampler = CVRPSampler(CVRPConfig.for_n(model_params.problem_size))
    else:
        raise ValueError(f"Unknown problem name: {model_params.problem_name}")
    return (sampler,)


@app.cell
def _(device, model_params):
    from core.models.attention_model.model import AttentionModel

    model = AttentionModel(
        problem_name=model_params.problem_name,
        hidden_dim=model_params.hidden_dim,
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
def _(mo, model, model_params, resume_run_dir, sampler):
    from pathlib import Path

    import yaml

    from core.models.attention_model.module import AMModule
    from core.models.attention_model.statistical_test import OneSidedPairedTTestResult
    from core.models.attention_model.training import BatchSummary, EpochSummary
    from core.models.trainer import Trainer

    def _find_latest_checkpoint(run_dir: Path) -> tuple[Path, int] | None:
        checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        epoch = int(latest.stem.split("_")[1])
        return latest, epoch

    run_dir = resume_run_dir or (
        Path(__file__).parent / "results" / "runs" / f"am-{model_params.run_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    if resume_run_dir is None:
        with open(run_dir / "config.yml", "w") as _f:
            yaml.safe_dump(model_params.model_dump(), _f, sort_keys=False)

    _wandb_id_file = run_dir / "wandb_run_id"
    if model_params.wandb_project:
        if resume_run_dir is not None and _wandb_id_file.exists():
            wandb.init(
                project=model_params.wandb_project,
                resume="must",
                id=_wandb_id_file.read_text().strip(),
            )
        else:
            wandb.init(
                project=model_params.wandb_project,
                name=model_params.run_name,
                config=model_params.model_dump(),
            )
            if wandb.run is not None:
                _wandb_id_file.write_text(wandb.run.id)

    def on_test_end(test: OneSidedPairedTTestResult):
        if model_params.wandb_project:
            wandb.log(
                {
                    "eval/obj_model": test.obj_model_mean,
                    "eval/obj_baseline": test.obj_baseline_mean,
                    "eval/baseline_updated": test.improved,
                }
            )

    attention_module = AMModule(model, sampler, on_baseline_sync=on_test_end)
    optimizer = torch.optim.AdamW(
        attention_module.parameters(),
        lr=model_params.learning_rate,
        weight_decay=0.0,
    )

    initial_epoch = 0
    _checkpoint_result = _find_latest_checkpoint(run_dir)
    if _checkpoint_result is not None:
        _checkpoint_path, _latest_epoch = _checkpoint_result
        _ckpt = torch.load(_checkpoint_path, map_location="cpu", weights_only=True)
        attention_module.model.load_state_dict(_ckpt["model"])
        attention_module.baseline.load_state_dict(_ckpt["baseline"])
        optimizer.load_state_dict(_ckpt["optimizer"])
        initial_epoch = _latest_epoch + 1
    elif model_params.init_weights:
        _ckpt = torch.load(model_params.init_weights, map_location="cpu", weights_only=True)
        attention_module.model.load_state_dict(_ckpt.get("model", _ckpt))
        attention_module.baseline.load_state_dict(
            _ckpt["baseline"] if "baseline" in _ckpt else attention_module.model.state_dict()
        )

    with (
        mo.status.progress_bar(total=model_params.epochs, title="Epochs") as epoch_bar,
        mo.status.progress_bar(
            total=model_params.epochs * model_params.batches_per_epoch, title="Batches"
        ) as step_bar,
    ):

        def on_batch_train_end(summary: BatchSummary):
            step_bar.update(subtitle=f"Loss: {summary.loss:.4f}")
            if model_params.wandb_project:
                wandb.log({f"train/{k}": v for k, v in summary.model_dump().items()})

        def on_epoch_train_end(summary: EpochSummary):
            epoch_bar.update(subtitle=f"Avg. Loss: {summary.loss:.4f}")
            is_last = summary.epoch == model_params.epochs - 1
            is_checkpoint = (summary.epoch + 1) % model_params.checkpoint_every_n_epochs == 0
            if is_last or is_checkpoint:
                checkpoints_path = run_dir / "checkpoints"
                checkpoints_path.mkdir(exist_ok=True)
                torch.save(
                    {
                        "model": attention_module.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "baseline": attention_module.baseline.state_dict(),
                    },
                    checkpoints_path / f"epoch_{summary.epoch:04d}.pt",
                )

        Trainer(
            max_epochs=model_params.epochs,
            training_batches_per_epoch=model_params.batches_per_epoch,
            batch_size=model_params.batch_size,
            on_batch_train_end=on_batch_train_end,
            on_epoch_train_end=[on_epoch_train_end],
            initial_epoch=initial_epoch,
        ).fit(module=attention_module, optimizer=optimizer, instance_sampler=sampler)

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
