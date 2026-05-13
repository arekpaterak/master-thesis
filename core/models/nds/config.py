from __future__ import annotations

import hashlib
import json
import math

from pydantic import BaseModel, Field, computed_field


class ProblemParams(BaseModel):
    """Configuration for the optimization problem."""

    name: str = Field(default="cvrp", description="Problem to solve.")
    size: int = Field(default=20, description="Number of customers.")


class ModelParams(BaseModel):
    """Configuration for the model architecture."""

    hidden_dim: int = Field(default=128, description="Hidden dimension.")
    num_heads: int = Field(default=8, description="Number of attention heads.")
    ff_dim: int = Field(default=512, description="Feed-forward hidden dimension.")
    poly_dim: int = Field(default=256, description="PolyNet hidden dimension.")
    seed_dim: int = Field(default=10, description="Seed dimension.")
    with_neighbours_encoding: bool = Field(default=True, description="Use neighbours encoding.")
    with_tour_encoding: bool | None = Field(
        default=None,
        description="Use tour encoding. None means problem-specific default (True for CVRP, False for TSP).",
    )


class OptimizerParams(BaseModel):
    """Configuration for the optimizer."""

    learning_rate: float = Field(default=1e-4, description="Learning rate.")
    weight_decay: float = Field(default=1e-6, description="Weight decay.")


class TrainerParams(BaseModel):
    """Configuration for the training process."""

    epochs: int = Field(default=1, description="Number of training epochs.")
    instances_per_epoch: int = Field(default=1500, description="Instances per epoch.")
    batch_size: int = Field(default=128, description="Training batch size.")

    @computed_field
    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.instances_per_epoch / self.batch_size)

    rollouts_per_instance: int = Field(default=128, description="Rollouts per instance.")
    num_initial_improvement_steps: int = Field(
        default=10, description="Initial improvement steps before training."
    )
    num_main_improvement_steps: int = Field(default=50, description="Number of improvement steps.")
    deconstruction_size: int = Field(
        default=5, description="Number of nodes removed per iteration."
    )


class LoggingConfig(BaseModel):
    """Configuration for logging and experiment tracking."""

    wandb_project: str = Field(default="nds", description="W&B project name (empty to skip).")


class ValidationConfig(BaseModel):
    """Configuration for validation during training."""

    num_instances: int = Field(default=1000, description="Instances to evaluate.")
    batch_size: int = Field(default=128, description="Batch size.")
    num_improvement_steps: int = Field(
        default=60,
        description="LNS steps.",
    )
    every_n_epochs: int = Field(default=1, description="Run validation every N training epochs.")
    seed: int = Field(default=42, description="Fixed seed for reproducible validation instances.")
    plot_enabled: bool = Field(default=True, description="Plot LNS progress for one instance.")
    plot_every_n_improvement_steps: int = Field(
        default=5, description="Capture solution snapshot every N LNS steps."
    )
    plot_dir: str = Field(default="plots", description="Directory to save plots.")
    plot_save_to_disk: bool = Field(default=True, description="Save plots.")
    plot_show_in_marimo: bool = Field(default=True, description="Display plots inline in marimo.")


class TrainConfig(BaseModel):
    """Complete configuration for NDS model training."""

    problem: ProblemParams = Field(default_factory=ProblemParams)
    model: ModelParams = Field(default_factory=ModelParams)
    optimizer: OptimizerParams = Field(default_factory=OptimizerParams)
    trainer: TrainerParams = Field(default_factory=TrainerParams)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @computed_field
    @property
    def run_name(self) -> str:
        config_dict = self.model_dump(exclude={"logging", "validation", "run_name"})

        config_json = json.dumps(config_dict, sort_keys=True)
        h = hashlib.md5(config_json.encode()).hexdigest()[:7]

        return f"{self.problem.name}{self.problem.size}-{h}"
