"""Standalone Typer entrypoint for training the NDS destroy model on CMST.

Example:
    uv run python train_cmst.py --size 100 --epochs 1 --batch-size 16
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

import wandb
from core.lns.acceptance_criterion import GreedyAcceptanceConfig
from core.lns.config import SearchParams, StartSolutionParams
from core.models.nds.config import (
    JointModelConfig,
    SplitModelConfig,
    TrainConfig,
    TrainerParams,
    ValidationConfig,
)
from core.models.nds.encoders.config import JointEncoderConfig, SplitEncoderConfig
from core.models.nds.factory import build_model
from core.models.nds.module import NDSModule
from core.models.optimizer import build_optimizer
from core.models.trainer import Trainer
from core.models.utils import save_checkpoint
from core.problems.cmst.config import CMSTConfig
from core.problems.sampler import Sampler

RUNS_DIR = Path(__file__).parent / "experiments" / "runs"

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def build_config(
    *,
    size: int,
    epochs: int,
    batch_size: int,
    instances_per_epoch: int,
    num_main_improvement_steps: int,
    deconstruction_size: int,
    num_rollouts: int,
    n_repairs: int,
    split: bool,
    validation_instances: int,
    validation_steps: int,
) -> TrainConfig:
    """Assemble a CMST training config using the numba backend end to end (no warmup)."""
    search = SearchParams(
        repair_type="greedy_insert",
        apply_and_repair_backend="numba",
        improvement_step_type="numba_sequential",
        num_improvement_steps=0,  # main loop uses apply-and-repair; no initial search
        num_rollouts=num_rollouts,
        deconstruction_size=deconstruction_size,
        n_repairs=n_repairs,
        acceptance=GreedyAcceptanceConfig(),
        start_solution_params=StartSolutionParams(num_improvement_steps=0),  # skip warmup
    )
    validation_search = SearchParams(
        repair_type="greedy_insert",
        apply_and_repair_backend="numba",
        improvement_step_type="numba_sequential",
        num_improvement_steps=validation_steps,
        num_rollouts=num_rollouts,
        deconstruction_size=deconstruction_size,
        n_repairs=n_repairs,
        acceptance=GreedyAcceptanceConfig(),
        start_solution_params=StartSolutionParams(num_improvement_steps=0),
    )

    encoder = SplitEncoderConfig() if split else JointEncoderConfig()
    model_cls = SplitModelConfig if split else JointModelConfig
    model = model_cls(problem_name="cmst", positional_encoding=None, encoder=encoder)

    return TrainConfig(
        problem=CMSTConfig(size=size),
        model=model,
        trainer=TrainerParams(
            epochs=epochs,
            instances_per_epoch=instances_per_epoch,
            batch_size=batch_size,
            num_main_improvement_steps=num_main_improvement_steps,
        ),
        search=search,
        validation=ValidationConfig(
            num_instances=validation_instances,
            batch_size=min(batch_size, validation_instances),
            plot_enabled=False,
            search=validation_search,
            validate_before_training=True,
        ),
    )


@app.command()
def main(
    size: int = typer.Option(100, help="Number of terminals (problem size)."),
    epochs: int = typer.Option(1, help="Training epochs."),
    batch_size: int = typer.Option(64, help="Training batch size."),
    instances_per_epoch: int = typer.Option(1500, help="Instances per epoch."),
    num_main_improvement_steps: int = typer.Option(100, help="Main improvement steps per batch."),
    deconstruction_size: int = typer.Option(15, help="Terminals removed per rollout."),
    num_rollouts: int = typer.Option(100, help="Rollouts per improvement step."),
    n_repairs: int = typer.Option(5, help="Repair attempts per rollout; best is kept."),
    split: bool = typer.Option(False, "--split", help="Use the split encoder instead of joint."),
    validation_instances: int = typer.Option(1000, help="Validation instances."),
    validation_steps: int = typer.Option(50, help="Validation improvement steps."),
    wandb_enabled: bool = typer.Option(True, "--wandb/--no-wandb", help="Log to Weights & Biases."),
    wandb_project: str = typer.Option("nds", help="W&B project name."),
) -> None:
    load_dotenv(".env")

    config = build_config(
        size=size,
        epochs=epochs,
        batch_size=batch_size,
        instances_per_epoch=instances_per_epoch,
        num_main_improvement_steps=num_main_improvement_steps,
        deconstruction_size=deconstruction_size,
        num_rollouts=num_rollouts,
        n_repairs=n_repairs,
        split=split,
        validation_instances=validation_instances,
        validation_steps=validation_steps,
    )
    config.logging.wandb_project = wandb_project if wandb_enabled else ""
    use_wandb = bool(config.logging.wandb_project)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Training CMST-{size} ({'split' if split else 'joint'} encoder) on {device}")

    run_dir = RUNS_DIR / f"nds-{config.run_name}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yml", "w") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)

    sampler = Sampler.from_config(config.problem)
    validation_sampler = Sampler.from_config(config.problem, seed=config.validation.seed)

    model = build_model(config.model).to(device)
    module = NDSModule(
        config=config,
        model=model,
        improvement_step_progress=lambda it: tqdm(it, desc="Improvement steps", leave=False),
        validation_improvement_step_progress=lambda it: tqdm(it, desc="Val steps", leave=False),
    )
    optimizer = build_optimizer(config.optimizer, module.parameters())

    if use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.run_name,
            config=config.model_dump(),
        )
        wandb.config.update(
            {"model/num_params": sum(p.numel() for p in model.parameters())},
            allow_val_change=True,
        )
        wandb.define_metric("val/*", step_metric="epoch")

    def on_batch_train_end(summary) -> None:
        if use_wandb:
            wandb.log({f"train/{k}": v for k, v in summary.model_dump().items()})

    def on_epoch_train_end(summary) -> None:
        save_checkpoint(
            path=run_dir / "checkpoints" / f"epoch_{summary.epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=summary.epoch,
            config=config,
        )

    def on_epoch_validation_end(summary) -> None:
        if use_wandb:
            metrics = summary.model_dump(exclude={"snapshots", "objective_curve", "step_labels"})
            wandb.log(
                {f"val/{k}": v for k, v in metrics.items() if k != "epoch"}
                | {"epoch": summary.epoch}
            )

    error: Exception | None = None
    try:
        Trainer(
            max_epochs=config.trainer.epochs,
            training_instances=config.trainer.instances_per_epoch,
            batch_size=config.trainer.batch_size,
            on_batch_train_end=on_batch_train_end,
            on_epoch_train_end=[on_epoch_train_end],
            validation_instances=config.validation.num_instances,
            validation_batch_size=config.validation.batch_size,
            validate_every_n_epochs=config.validation.every_n_epochs,
            on_epoch_validation_end=[on_epoch_validation_end],
            epoch_progress=lambda it: tqdm(it, desc="Epochs", total=config.trainer.epochs),
            batch_progress=lambda it: tqdm(it, desc="Batches", leave=False),
            validation_batch_progress=lambda it: tqdm(it, desc="Val batches", leave=False),
        ).fit(
            module=module,
            optimizer=optimizer,
            instance_sampler=sampler,
            validation_instance_sampler=validation_sampler,
        )
    except Exception as exc:
        error = exc
    finally:
        if use_wandb:
            wandb.finish(exit_code=1 if error is not None else 0)

    if error is not None:
        raise error

    typer.echo("Training completed.")


if __name__ == "__main__":
    app()
