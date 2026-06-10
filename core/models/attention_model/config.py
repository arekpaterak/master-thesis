from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field, computed_field

from core.problems import ProblemConfig, TSPConfig


class ModelParams(BaseModel):
    """Configuration for the model architecture."""

    hidden_dim: int = Field(default=128, description="Hidden dimension.")
    num_heads: int = Field(default=8, description="Number of attention heads.")
    num_layers: int = Field(default=3, description="Number of attention layers.")


class OptimizerParams(BaseModel):
    """Configuration for the optimizer."""

    learning_rate: float = Field(default=1e-4, description="Learning rate.")


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
    optimizer: OptimizerParams = Field(default_factory=OptimizerParams)
    trainer: TrainerParams = Field(default_factory=TrainerParams)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @computed_field
    @property
    def run_name(self) -> str:
        config_dict = self.model_dump(exclude={"logging", "run_name"})
        config_json = json.dumps(config_dict, sort_keys=True)
        h = hashlib.md5(config_json.encode()).hexdigest()[:6]
        return f"{self.problem.name}{self.problem.size}-{h}"
