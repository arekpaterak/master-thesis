from __future__ import annotations

import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: plots are saved/logged, never shown

import matplotlib.pyplot as plt
import torch
import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import wandb
from core.models.nds.config import JointModelConfig, TrainConfig
from core.models.nds.factory import build_model
from core.models.nds.module import NDSModule
from core.models.nds.pretrained.load import (
    load_original_pretrained_decoder_weights,
    load_original_pretrained_encoder_weights,
)
from core.models.nds.training import BatchTrainSummary, EpochTrainSummary
from core.models.nds.validation import EpochValidationSummary
from core.models.optimizer import build_optimizer
from core.models.trainer import Trainer
from core.models.utils import (
    checkpoint_epoch,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from core.problems.dataset import DatasetSampler
from core.problems.sampler import Sampler
from core.results.plots import plot_solutions_from_search, save_figure
from core.results.wandb_artifacts import upload_final_checkpoint

EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
TRAIN_CONFIGS_DIR = EXPERIMENTS_DIR / "configs" / "train"
RUNS_DIR = EXPERIMENTS_DIR / "runs"

console = Console()
app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@app.command()
def main(
    ctx: typer.Context,
    config_arg: Path | None = typer.Argument(
        None, metavar="[CONFIG]", help="Path to YAML config file (or name under configs/train)."
    ),
    config: Path | None = typer.Option(None, help="Path to YAML config file."),
    resume: bool = typer.Option(
        False, "--resume", help="Continue an existing run from its latest checkpoint."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite the run directory if it already exists."
    ),
    tmp: bool = typer.Option(
        False,
        "--tmp",
        help="Write to a temporary subdir, always overwriting it but leaving the main dir untouched.",
    ),
    device: str | None = typer.Option(None, help="Torch device (default: cuda if available)."),
    help_config: bool = typer.Option(
        False, "--help-config", help="Show config fields with their defaults and exit."
    ),
) -> None:
    """Train the NDS model from a YAML config, with optional nested --section.field overrides."""
    if help_config:
        _print_usage()
        raise typer.Exit()

    config_path = config or config_arg
    if config_path is None:
        _print_usage()
        raise typer.Exit()

    selected = [
        name
        for name, enabled in (("--resume", resume), ("--force", force), ("--tmp", tmp))
        if enabled
    ]
    if len(selected) > 1:
        raise typer.BadParameter(f"{' and '.join(selected)} cannot be combined.")

    # Resolve the config path against the invocation cwd, then run from the project root
    # so config-relative paths (dataset_path, .env) resolve regardless of where invoked.
    config_path = config_path.resolve()
    os.chdir(PROJECT_ROOT)
    load_dotenv(".env")

    overrides = _parse_overrides(ctx.args)
    train_config: TrainConfig = _load_config(config_path, overrides)
    run_dir = _resolve_run_dir(train_config, config_path, resume=resume, force=force, tmp=tmp)
    if resume:
        # Prefer the config saved with the run so its original settings persist, then
        # re-apply any CLI overrides on top of it.
        train_config = _load_config(run_dir / "config.yml", overrides)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    console.print(
        f"[bold]{train_config.run_name}[/bold]  |  problem: "
        f"{train_config.problem.name}{train_config.problem.size}  |  "
        f"encoder: {train_config.model.encoder_mode}  |  device: {torch_device}"
    )
    print(yaml.safe_dump(train_config.model_dump(exclude={"run_name"}), sort_keys=False))

    sampler, val_sampler = _build_samplers(train_config)
    model = _build_model(train_config)
    initial_epoch = _prepare_run_dir(train_config, model, run_dir, resume=resume)
    _run_training(
        config=train_config,
        model=model,
        sampler=sampler,
        val_sampler=val_sampler,
        run_dir=run_dir,
        initial_epoch=initial_epoch,
        resume=resume,
        device=torch_device,
    )
    console.print("[bold green]Training completed.[/bold green]")


def _print_usage() -> None:
    """Print CLI usage and the config fields with their defaults."""
    usage = Table(title="NDS Training -- CLI Usage")
    usage.add_column("Flag", style="cyan")
    usage.add_column("Description")
    usage.add_row("<path>", "Path to YAML config file (positional)")
    usage.add_row("--config <path>", "Path to YAML config file")
    usage.add_row("--resume", "Continue an existing run from its latest checkpoint.")
    usage.add_row("--force", "Overwrite the run directory if it already exists.")
    usage.add_row(
        "--tmp",
        "Write to a temporary subdir, always overwriting it but leaving the main dir untouched.",
    )
    usage.add_row(
        "--<section>.<field> <value>",
        "Override a nested config field, e.g. --trainer.epochs 5",
    )
    console.print(usage)

    fields = Table(title="Config fields and defaults")
    fields.add_column("Key", style="cyan")
    fields.add_column("Default", style="yellow")
    for section, params in TrainConfig().model_dump(exclude={"run_name", "logging"}).items():
        for field, value in params.items():
            fields.add_row(f"{section}.{field}", str(value))
    console.print(fields)


def _parse_overrides(args: list[str]) -> dict[str, object]:
    """Turn ``["--trainer.epochs", "5", "--amp.enabled=true"]`` into nested override paths.

    Values are parsed through ``yaml.safe_load`` so scalars pick up their natural type
    (int, float, bool, null) while plain identifiers stay strings, matching how the
    same value would have been read from the YAML config.
    """
    overrides: dict[str, object] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            raise typer.BadParameter(f"Unexpected argument: {token}")
        key = token[2:]
        if "=" in key:
            key, raw_value = key.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(args):
                raise typer.BadParameter(f"Missing value for override --{key}")
            raw_value = args[index + 1]
            index += 2
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def _load_config(config_path: Path, overrides: dict[str, object]) -> TrainConfig:
    """Build the training config from a YAML file plus CLI overrides."""
    resolved = config_path
    if not resolved.exists():
        resolved = TRAIN_CONFIGS_DIR / resolved.name
    with open(resolved) as f:
        yaml_data: dict = yaml.safe_load(f) or {}

    _merge_overrides(yaml_data, overrides)
    _backfill_run_name(yaml_data, config_path)
    return TrainConfig.model_validate(yaml_data)


def _backfill_run_name(yaml_data: dict, config_path: Path) -> None:
    """Default ``run_name`` to the config filename stem plus a timestamp.

    Used only as the wandb display label; the run directory is keyed on the config
    stem alone (see ``_resolve_run_dir``). Skipped when the config already sets
    ``run_name`` -- notably the config saved with a run, so a resumed run keeps its
    original name.
    """
    if yaml_data.get("run_name"):
        return
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    yaml_data["run_name"] = f"{config_path.stem}-{timestamp}"


def _merge_overrides(yaml_data: dict, overrides: dict[str, object]) -> None:
    """Merge dotted override paths into ``yaml_data`` in place."""
    for key, value in overrides.items():
        parts = [part.replace("-", "_") for part in key.split(".")]
        node = yaml_data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _build_samplers(config: TrainConfig) -> tuple[Sampler, Sampler]:
    """Create the training sampler and the validation sampler."""
    sampler = Sampler.from_config(config.problem)
    if config.validation.dataset_path is None:
        return sampler, Sampler.from_config(config.problem, seed=config.validation.seed)

    val_sampler = DatasetSampler.from_file(config.problem.name, config.validation.dataset_path)
    if config.validation.num_instances > len(val_sampler):
        raise ValueError(
            f"validation.num_instances={config.validation.num_instances} exceeds "
            f"the {len(val_sampler)} instances in {config.validation.dataset_path}."
        )
    return sampler, val_sampler


def _build_model(config: TrainConfig) -> torch.nn.Module:
    model = build_model(config.model)

    # original_pretrained_weights / freeze_encoder are joint-only;
    # the split model cannot initialise from a pretrained NDS encoder.
    is_joint = isinstance(config.model, JointModelConfig)
    if is_joint and config.model.original_pretrained_weights:
        pretrained = config.model.original_pretrained_weights
        load_original_pretrained_encoder_weights(model.encoder, pretrained.path)
        console.print(f"Loaded pretrained encoder weights from {pretrained.path}")
        if pretrained.load_decoder:
            load_original_pretrained_decoder_weights(model.decoder, pretrained.path)
            console.print(f"Loaded pretrained decoder weights from {pretrained.path}")
    elif config.model.init_weights:
        load_checkpoint(config.model.init_weights, model)
        console.print(f"Initiated model with weights from {config.model.init_weights}")

    if is_joint and config.model.freeze_encoder:
        model.encoder.requires_grad_(False)
        console.print("Frozen encoder weights")

    return model


def _resolve_run_dir(
    config: TrainConfig,
    config_path: Path,
    *,
    resume: bool,
    force: bool,
    tmp: bool,
) -> Path:
    """Resolve the run directory from the config file name and apply the exist policy.

    The directory identity is stable -- ``runs/<problem>/<config-stem>`` -- mirroring how
    eval keys its results on the config name. The flags decide what happens when that
    directory already exists: ``--resume`` requires it, ``--force`` and ``--tmp`` overwrite
    it, and a plain run refuses to touch an existing one.
    """
    run_dir = RUNS_DIR / config.problem.name / config_path.stem
    if tmp:
        run_dir = run_dir / "tmp"

    if resume:
        if not run_dir.exists():
            raise typer.BadParameter(
                f"Cannot resume: no run at {run_dir}. Start it without --resume first."
            )
        return run_dir

    if run_dir.exists():
        if force or tmp:
            shutil.rmtree(run_dir)
        else:
            raise typer.BadParameter(
                f"Run already exists at {run_dir}. Use --resume to continue it, "
                "--force to overwrite it, or --tmp to write to a scratch subdir."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _prepare_run_dir(
    config: TrainConfig,
    model: torch.nn.Module,
    run_dir: Path,
    *,
    resume: bool,
) -> int:
    """Restore model weights on resume, persist the config on a fresh run, init wandb."""
    initial_epoch = 0
    if resume:
        checkpoint_path, last_epoch = _find_latest_checkpoint(run_dir)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"])
            initial_epoch = last_epoch + 1
    else:
        with open(run_dir / "config.yml", "w") as f:
            yaml.safe_dump(config.model_dump(), f, sort_keys=False)

    if config.logging.wandb_project:
        wandb_id_path = run_dir / "wandb_run_id"
        if resume and wandb_id_path.exists():
            wandb.init(
                project=config.logging.wandb_project,
                resume="must",
                id=wandb_id_path.read_text().strip(),
            )
        else:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.run_name,
                config=config.model_dump(),
                tags=_run_tags(config),
            )
            wandb_id_path.write_text(wandb.run.id)
        wandb.config.update(
            {
                "model/num_params": sum(p.numel() for p in model.parameters()),
                "model/num_params_trainable": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "model/num_params_decoder": sum(p.numel() for p in model.decoder.parameters()),
            },
            allow_val_change=True,
        )
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("train/epoch_time_sec", step_metric="epoch")

    return initial_epoch


def _run_tags(config: TrainConfig) -> list[str]:
    """Tags describing the model variant, for filtering runs in wandb."""
    decoder_tag = "ar" if config.model.decoder.type == "ar" else "nar"
    return [
        config.model.encoder_mode,
        decoder_tag,
        f"M={config.search.deconstruction_size}",
    ]


def _find_latest_checkpoint(run_dir: Path) -> tuple[Path | None, int]:
    """Return the highest-numbered checkpoint in ``run_dir`` and its epoch."""
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    return latest, checkpoint_epoch(latest)


def _run_training(
    config: TrainConfig,
    model: torch.nn.Module,
    sampler: Sampler,
    val_sampler: Sampler,
    run_dir: Path,
    initial_epoch: int,
    resume: bool,
    device: torch.device,
) -> None:
    """Run the training loop with checkpointing, wandb logging and plotting."""
    model.to(device)
    nds_module = NDSModule(
        config=config,
        model=model,
        improvement_step_progress=lambda it: tqdm(it, desc="Improvement steps", leave=False),
        validation_improvement_step_progress=lambda it: tqdm(
            it, desc="Validation improvement steps", leave=False
        ),
    )
    optimizer = build_optimizer(config.optimizer, nds_module.parameters())

    if resume:
        checkpoint_path, _ = _find_latest_checkpoint(run_dir)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(ckpt["optimizer"])

    def on_batch_train_end(summary: BatchTrainSummary) -> None:
        if config.logging.wandb_project:
            wandb.log({f"train/{k}": v for k, v in summary.model_dump().items()})

    def on_epoch_train_end(summary: EpochTrainSummary) -> None:
        if config.logging.wandb_project:
            wandb.log({"train/epoch_time_sec": summary.epoch_time_sec, "epoch": summary.epoch})
        is_last = summary.epoch == config.trainer.epochs - 1
        is_checkpoint = (summary.epoch + 1) % config.trainer.checkpoint_every_n_epochs == 0
        if is_last or is_checkpoint:
            checkpoints_path = run_dir / "checkpoints"
            checkpoints_path.mkdir(exist_ok=True)
            save_checkpoint(
                path=checkpoints_path / f"epoch_{summary.epoch:04d}.pt",
                model=nds_module.model,
                optimizer=optimizer,
                epoch=summary.epoch,
                config=config,
            )

    def on_epoch_validation_end(val_summary: EpochValidationSummary) -> None:
        fig = None
        if config.validation.plot_enabled:
            fig = plot_solutions_from_search(
                val_summary.snapshots, val_summary.step_labels, epoch=val_summary.epoch
            )
            if config.validation.plot_save_to_disk:
                stem = run_dir / "plots" / f"epoch_{val_summary.epoch:04d}"
                save_figure(fig, f"{stem}.pdf")
                save_figure(fig, f"{stem}.png", dpi=150)
        if config.logging.wandb_project:
            log = {
                "val/objective_initial": val_summary.objective_initial,
                "val/objective_final": val_summary.objective_final,
                "val/improvement": val_summary.improvement,
                "val/relative_improvement": val_summary.relative_improvement,
                "val/batch_time_sec": val_summary.batch_time_sec,
                "val/epoch_time_sec": val_summary.epoch_time_sec,
                "val/instances_per_sec": val_summary.instances_per_sec,
                "val/steps_to_best": val_summary.steps_to_best,
                "val/diversity_overlap": val_summary.diversity_overlap,
                "val/unique_rollouts": val_summary.unique_rollouts,
                "val/objective_curve": wandb.plot.line_series(
                    xs=val_summary.step_labels,
                    ys=[val_summary.objective_curve],
                    keys=["objective"],
                    title=f"Validation objective (epoch {val_summary.epoch})",
                    xname="Improvement step",
                ),
                "epoch": val_summary.epoch,
            }
            if val_summary.objective_final_augmented is not None:
                log["val/objective_final_augmented"] = val_summary.objective_final_augmented
            if fig is not None:
                log["val/search_progress"] = wandb.Image(fig)
            wandb.log(log)
        if fig is not None:
            plt.close(fig)

    error: Exception | None = None
    try:
        Trainer(
            max_epochs=config.trainer.epochs,
            training_instances=config.trainer.instances_per_epoch,
            batch_size=config.trainer.batch_size,
            on_batch_train_end=on_batch_train_end,
            on_epoch_train_end=[on_epoch_train_end],
            initial_epoch=initial_epoch,
            validation_instances=config.validation.num_instances,
            validation_batch_size=config.validation.batch_size,
            validate_every_n_epochs=config.validation.every_n_epochs,
            on_epoch_validation_end=[on_epoch_validation_end],
            epoch_progress=lambda it: tqdm(
                it, desc="Epochs", total=config.trainer.epochs, initial=initial_epoch
            ),
            batch_progress=lambda it: tqdm(it, desc="Batches", leave=False),
            validation_batch_progress=lambda it: tqdm(it, desc="Validation batches", leave=False),
            strict_number_of_instances=config.trainer.strict_number_of_instances,
            validate_before_training=config.validation.validate_before_training,
        ).fit(
            module=nds_module,
            optimizer=optimizer,
            instance_sampler=sampler,
            validation_instance_sampler=val_sampler,
        )
    except Exception as exc:
        # Surface the error instead of silently ending the run, then re-raise below so
        # the process exits non-zero (after the cleanup in `finally`).
        traceback.print_exc()
        error = exc
    finally:
        if config.logging.wandb_project:
            final_checkpoint, final_epoch = _find_latest_checkpoint(run_dir)
            if final_checkpoint is not None:
                upload_final_checkpoint(
                    run_name=config.run_name,
                    checkpoint_path=final_checkpoint,
                    epoch=final_epoch,
                )
            wandb.finish(exit_code=1 if error is not None else 0)

    if error is not None:
        raise error


if __name__ == "__main__":
    app()
