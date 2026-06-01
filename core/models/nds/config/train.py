from __future__ import annotations

import hashlib
import json
import math

from pydantic import BaseModel, Field, computed_field, model_validator

from core.models.nds.config.common import ProblemParams, SearchParams
from core.models.nds.decoders.config import AutoregressiveDecoderConfig, DecoderConfig


class OriginalPretrainedConfig(BaseModel):
    """Configuration for initialising from an original NDS pretrained checkpoint."""

    path: str = Field(description="Path to the pretrained checkpoint file.")
    load_decoder: bool = Field(default=False, description="Also load decoder weights.")


class ModelParams(BaseModel):
    """Configuration for the model architecture and weight initialisation."""

    hidden_dim: int = Field(default=128, description="Hidden dimension.")
    num_heads: int = Field(default=8, description="Number of attention heads.")
    ff_dim: int = Field(default=512, description="Feed-forward hidden dimension.")
    poly_dim: int = Field(default=256, description="PolyNet hidden dimension.")
    seed_dim: int = Field(default=10, description="Seed dimension.")
    with_neighbours_encoding: bool = Field(default=True, description="Use neighbours encoding.")
    with_route_encoding: bool | None = Field(
        default=None,
        description="Use route encoding. None means problem-specific default (True for CVRP, False for TSP).",
    )
    decoder: DecoderConfig = Field(default_factory=AutoregressiveDecoderConfig)

    init_weights: str | None = Field(
        default=None, description="Path to a checkpoint to load model weights from (new run)."
    )
    original_pretrained_weights: OriginalPretrainedConfig | None = Field(
        default=None, description="Original NDS pretrained checkpoint to initialise from."
    )
    freeze_encoder: bool = Field(
        default=False, description="Freeze encoder weights during training."
    )

    @model_validator(mode="after")
    def _validate_weights_mutually_exclusive(self) -> ModelParams:
        if self.init_weights is not None and self.original_pretrained_weights is not None:
            raise ValueError("init_weights and original_pretrained_weights are mutually exclusive.")
        return self


class OptimizerParams(BaseModel):
    """Configuration for the optimizer."""

    learning_rate: float = Field(default=1e-4, description="Learning rate.")
    weight_decay: float = Field(default=1e-6, description="Weight decay.")


class TrainerParams(BaseModel):
    """Configuration for the training loop (excluding search and model weights)."""

    epochs: int = Field(default=1, description="Number of training epochs.")
    instances_per_epoch: int = Field(default=1500, description="Instances per epoch.")
    batch_size: int = Field(default=128, description="Training batch size.")

    @computed_field
    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.instances_per_epoch / self.batch_size)

    num_main_improvement_steps: int = Field(
        default=100, description="Number of gradient-bearing improvement steps per batch."
    )
    checkpoint_every_n_epochs: int = Field(
        default=10, description="Save a checkpoint every N epochs (always saves the last epoch)."
    )


class LoggingConfig(BaseModel):
    """Configuration for logging and experiment tracking."""

    wandb_project: str = Field(default="nds", description="W&B project name (empty to skip).")


class ValidationConfig(BaseModel):
    """Configuration for validation during training."""

    num_instances: int = Field(default=1000, description="Instances to evaluate.")
    batch_size: int = Field(default=10, description="Batch size.")
    every_n_epochs: int = Field(default=10, description="Run validation every N training epochs.")
    seed: int = Field(default=42, description="Fixed seed for reproducible validation instances.")
    plot_enabled: bool = Field(default=True, description="Plot LNS progress for one instance.")
    plot_every_n_improvement_steps: int = Field(
        default=5, description="Capture solution snapshot every N outer SA iterations."
    )
    plot_save_to_disk: bool = Field(default=True, description="Save plots.")
    search: SearchParams = Field(
        default_factory=lambda: SearchParams(
            num_improvement_steps=50, num_rollouts=100, n_repairs=5
        ),
        description="Search-loop knobs used during validation.",
    )


class TrainConfig(BaseModel):
    """Complete configuration for NDS model training."""

    problem: ProblemParams = Field(default_factory=ProblemParams)
    model: ModelParams = Field(default_factory=ModelParams)
    optimizer: OptimizerParams = Field(default_factory=OptimizerParams)
    trainer: TrainerParams = Field(default_factory=TrainerParams)
    search: SearchParams = Field(
        default_factory=SearchParams,
        description="Search-loop knobs used during the training initial phase and main loop.",
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @computed_field
    @property
    def run_name(self) -> str:
        config_dict = self.model_dump(exclude={"logging", "validation", "run_name"})

        config_json = json.dumps(config_dict, sort_keys=True)
        h = hashlib.md5(config_json.encode()).hexdigest()[:7]

        return f"{self.problem.name}{self.problem.size}-{h}"
