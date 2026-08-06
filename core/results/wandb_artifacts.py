from __future__ import annotations

from pathlib import Path

import wandb


def upload_final_checkpoint(*, run_name: str, checkpoint_path: Path, epoch: int) -> None:
    artifact = wandb.Artifact(
        name=run_name,
        type="model",
        metadata={"epoch": epoch, "run_name": run_name},
    )
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact, aliases=["final", f"epoch-{epoch:04d}"])
