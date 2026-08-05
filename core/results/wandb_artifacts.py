from __future__ import annotations

from pathlib import Path

import wandb


def upload_final_checkpoint(*, run_name: str, checkpoint_path: Path, epoch: int) -> None:
    artifact = wandb.Artifact(
        name=f"checkpoint-{run_name}",
        type="model",
        metadata={"epoch": epoch, "run_name": run_name},
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact, aliases=["final", f"epoch-{epoch:04d}"])


def upload_validation_plots(*, run_name: str, plots_dir: Path) -> None:
    if not plots_dir.exists():
        return
    plots = sorted(plots_dir.glob("*.pdf")) + sorted(plots_dir.glob("*.png"))
    if not plots:
        return
    artifact = wandb.Artifact(
        name=f"val-plots-{run_name}",
        type="validation-plots",
        metadata={"run_name": run_name},
    )
    for plot_path in plots:
        artifact.add_file(str(plot_path))
    wandb.log_artifact(artifact)


def upload_eval_results(
    *,
    run_name: str,
    eval_parquet_path: Path,
    eval_id: str,
    metadata: dict | None = None,
    link_checkpoint_alias: str | None = "final",
) -> None:
    if link_checkpoint_alias is not None:
        wandb.use_artifact(f"checkpoint-{run_name}:{link_checkpoint_alias}")
    artifact = wandb.Artifact(
        name=f"eval-{run_name}",
        type="eval-results",
        metadata=metadata or {},
    )
    artifact.add_file(str(eval_parquet_path))
    wandb.log_artifact(artifact, aliases=[eval_id])


def upload_eval_dir(
    *,
    run_name: str,
    eval_dir: Path,
    metadata: dict | None = None,
) -> None:
    if not eval_dir.exists():
        return
    artifact = wandb.Artifact(
        name=f"eval-{run_name}",
        type="eval-results",
        metadata=metadata or {},
    )
    artifact.add_dir(str(eval_dir))
    wandb.log_artifact(artifact)


def upload_eval_plots(*, run_name: str, eval_id: str, plots_dir: Path) -> None:
    if not plots_dir.exists():
        return
    plots = sorted(plots_dir.glob("*.pdf"))
    if not plots:
        return
    artifact = wandb.Artifact(
        name=f"eval-plots-{run_name}",
        type="eval-plots",
    )
    for plot_path in plots:
        artifact.add_file(str(plot_path))
    wandb.log_artifact(artifact, aliases=[eval_id])
