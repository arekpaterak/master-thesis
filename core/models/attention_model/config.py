from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, Field, model_validator

from core.models.optimizer import OptimizerConfig
from core.problems import ProblemConfig, TSPConfig


class ModelParams(BaseModel):
    """Configuration for the model architecture."""

    hidden_dim: int = Field(default=128, description="Hidden dimension.")
    num_heads: int = Field(default=8, description="Number of attention heads.")
    num_layers: int = Field(default=3, description="Number of attention layers.")


class TrainerParams(BaseModel):
    """Configuration for the training process."""

    epochs: int = Field(default=1, description="Number of training epochs.")
    batches_per_epoch: int = Field(default=500, description="Batches per epoch.")
    batch_size: int = Field(default=256, description="Training batch size.")
    checkpoint_every_n_epochs: int = Field(
        default=10, description="Save a checkpoint every N epochs (always saves the last epoch)."
    )
    init_weights: str | None = Field(
        default=None, description="Path to a checkpoint to load model weights from (new run)."
    )


class LoggingConfig(BaseModel):
    """Configuration for logging and experiment tracking."""

    wandb_project: str = Field(
        default="attention-model", description="W&B project name (empty to skip)."
    )


class TrainConfig(BaseModel):
    """Complete configuration for Attention Model training."""

    problem: ProblemConfig = Field(default_factory=TSPConfig)
    model: ModelParams = Field(default_factory=ModelParams)
    # The attention model trains without weight decay (matches the repro).
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig(weight_decay=0.0))
    trainer: TrainerParams = Field(default_factory=TrainerParams)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    run_name: str = Field(
        default="",
        title="Run name",
    )

    @model_validator(mode="after")
    def _assign_run_name(self) -> Self:
        """Generate a unique timestamped run name once, unless one was supplied."""
        if not self.run_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_name = f"{self.problem.name}{self.problem.size}-{timestamp}"
        return self
