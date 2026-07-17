from __future__ import annotations

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Configuration for logging and experiment tracking (training and evaluation)."""

    wandb_project: str = Field(
        default="nds",
        title="W&B project (empty to skip)",
        description="W&B project name (empty to skip).",
    )
    # TODO: add wandb run_id
