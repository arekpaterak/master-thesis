from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, computed_field, model_validator

from core.lns.acceptance_criterion import (
    AcceptanceCriterionConfig,
    GreedyAcceptanceConfig,
    UnconditionalAcceptanceConfig,
)
from core.lns.config import SearchParams
from core.logging_config import LoggingConfig
from core.models.nds.baseline import (
    BaselineConfig,
    SharedBaselineConfig,
    build_baseline,
)
from core.models.nds.decoders.config import AutoregressiveDecoderConfig, DecoderConfig
from core.models.nds.encoders.config import EncoderConfig, PositionalEncodingConfig
from core.models.nds.policy_loss import BestRolloutLossConfig, PolicyLossConfig
from core.models.nds.pretrained.config import (
    OriginalPretrainedConfig,
    validate_encoder_matches_original_nds,
)
from core.models.nds.reward import AbsoluteRewardConfig, RewardConfig
from core.models.optimizer import OptimizerConfig
from core.models.runtime import AmpConfig, CompileConfig
from core.problems import CVRPConfig, ProblemConfig


class _ModelConfigBase(BaseModel):
    """Fields shared by every model variant.

    ``encoder_mode`` is the discriminator selecting a concrete variant; see
    :data:`ModelParams`.
    """

    hidden_dim: int = Field(default=128, title="Hidden dim", description="Hidden dimension.")
    num_heads: int = Field(default=8, title="Num. heads", description="Number of attention heads.")
    ff_dim: int = Field(default=512, title="FF dim", description="Feed-forward hidden dimension.")
    poly_dim: int = Field(default=256, title="Poly dim", description="PolyNet hidden dimension.")
    seed_dim: int = Field(default=10, title="Seed dim", description="Seed dimension.")
    positional_encoding: PositionalEncodingConfig | None = Field(
        default=None, title="Positional encoding"
    )
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=AutoregressiveDecoderConfig)
    init_weights: str | None = Field(
        default=None,
        title="Checkpoint path",
        description="Path to a checkpoint to load model weights from.",
    )


class JointModelConfig(_ModelConfigBase):
    """NDS joint encoder: re-encodes the instance and solution at every step.

    Positional encoding, if set, is injected before the encoder. This is the only
    variant that can initialise from an original NDS pretrained checkpoint.
    """

    encoder_mode: Literal["joint"] = "joint"
    with_neighbours_encoding: bool = Field(
        default=True, title="With neighbours encoding", description="Use neighbours encoding."
    )
    with_route_encoding: bool | None = Field(
        default=None,
        description="Use route encoding. None means problem-specific default (True for CVRP, False for TSP).",
    )
    original_pretrained_weights: OriginalPretrainedConfig | None = Field(
        default=None, description="Original NDS pretrained checkpoint to initialise from."
    )
    freeze_encoder: bool = Field(
        default=False, title="Freeze encoder", description="Freeze encoder weights during training."
    )

    @model_validator(mode="after")
    def _validate_weights_mutually_exclusive(self) -> Self:
        if self.init_weights is not None and self.original_pretrained_weights is not None:
            raise ValueError("init_weights and original_pretrained_weights are mutually exclusive.")
        return self

    @model_validator(mode="after")
    def _validate_encoder_for_original_pretrained(self) -> Self:
        if self.original_pretrained_weights is not None:
            validate_encoder_matches_original_nds(self.encoder)
        return self


class SplitModelConfig(_ModelConfigBase):
    """LNS-native split encoder: the instance is encoded once and the solution enters
    via positional encodings fused before the decoder.

    The solution-aware neighbours/route encoding layers are always disabled, and there
    is no pretrained-NDS initialisation (its weights are entangled with those layers),
    so the corresponding joint-only fields are absent here.
    """

    encoder_mode: Literal["split"] = "split"
    encode_once_in_training: bool = Field(
        default=True,
        title="Encode once in training",
        description="Run the instance encoder once per training step and reuse the "
        "embeddings across all improvement steps (exact, with bounded memory). "
        "False recomputes the encoder every step (current behavior).",
    )
    with_fusion_neighbours_encoding: bool = Field(
        default=True,
        title="Fusion neighbours encoding",
        description="Apply a neighbours encoding layer after positional fusion (before route and attention).",
    )
    with_fusion_route_encoding: bool | None = Field(
        default=None,
        title="Fusion route encoding",
        description="Apply a route encoding layer after positional fusion (before attention). "
        "CVRP only. None means problem-specific default (True for CVRP, False for TSP).",
    )
    num_fusion_attention_layers: int = Field(
        default=0,
        title="Fusion attention layers",
        description="Self-attention layers applied to all nodes after positional fusion. "
        "0 = disabled.",
    )


ModelConfig = Annotated[JointModelConfig | SplitModelConfig, Field(discriminator="encoder_mode")]


# TODO: Move outside of nds
class TrainerParams(BaseModel):
    """Configuration for the training loop (excluding search and model weights)."""

    epochs: int = Field(default=1, title="Epochs", description="Number of training epochs.")
    instances_per_epoch: int = Field(
        default=1500, title="Instances per epoch", description="Instances per epoch."
    )
    batch_size: int = Field(default=64, title="Batch size", description="Training batch size.")

    @computed_field
    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.instances_per_epoch / self.batch_size)

    # TODO: It should not be under TrainerParams
    num_main_improvement_steps: int = Field(
        default=100,
        title="Main improvement steps",
        description="Number of the main improvement steps per batch.",
    )
    checkpoint_every_n_epochs: int = Field(
        default=1,
        title="Checkpoint every N epochs",
        description="Save a checkpoint every N epochs (always saves the last epoch).",
    )
    strict_number_of_instances: bool = Field(
        default=False,
        title="Strict instance count",
        description=(
            "If True, training/validation process exactly the configured "
            "instance count (final batch may be smaller). If False (default), "
            "the trainer rounds the batch count up so every batch has the same "
            "size"
        ),
    )
    # TODO: It should not be under TrainerParams
    training_rollout_selection: Literal["reward", "objective"] = Field(
        default="objective",
        description="Whether to pick the best candidate rollout by argmax(reward) or argmin(objective).",
    )
    # TODO: It should not be under TrainerParams
    training_acceptance: AcceptanceCriterionConfig = Field(
        default_factory=GreedyAcceptanceConfig,
        description="Acceptance criterion used for the main training loop.",
    )

    # TODO: It should not be under TrainerParams
    @model_validator(mode="after")
    def _validate_training_acceptance(self) -> Self:
        match self.training_acceptance:
            case UnconditionalAcceptanceConfig() | GreedyAcceptanceConfig():
                pass
            case _:
                raise ValueError("Unsupported training acceptance criterion.")
        return self


class ValidationConfig(BaseModel):
    """Configuration for validation during training."""

    num_instances: int = Field(
        default=1000, title="Num. instances", description="Instances to evaluate."
    )
    batch_size: int = Field(default=20, title="Batch size", description="Batch size.")
    every_n_epochs: int = Field(
        default=1, title="Every N epochs", description="Run validation every N training epochs."
    )
    seed: int = Field(
        default=42, title="Seed", description="Fixed seed for reproducible validation instances."
    )
    dataset_path: str | None = Field(
        default=None,
        description=(
            "If set, validate on instances loaded from this benchmark file instead "
            "of sampling them randomly. The first num_instances are used."
        ),
    )
    plot_enabled: bool = Field(
        default=True, title="Plot enabled", description="Plot LNS progress for one instance."
    )
    plot_every_n_improvement_steps: int = Field(
        default=5,
        title="Plot every N steps",
        description="Capture solution snapshot every N main search iterations.",
    )
    plot_save_to_disk: bool = Field(default=True, title="Save to disk", description="Save plots.")
    validate_before_training: bool = Field(
        default=False,
        title="Validate before training",
        description="Run a validation pass before the first training epoch.",
    )
    search: SearchParams = Field(
        default_factory=lambda: SearchParams(
            num_improvement_steps=50,
            num_rollouts=100,
            acceptance=GreedyAcceptanceConfig(),
        ),
    )


class TrainConfig(BaseModel):
    """Complete configuration for NDS model training."""

    problem: ProblemConfig = Field(default_factory=CVRPConfig)
    model: ModelConfig = Field(default_factory=JointModelConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loss: PolicyLossConfig = Field(default_factory=BestRolloutLossConfig)
    baseline: BaselineConfig = Field(default_factory=SharedBaselineConfig)
    reward: RewardConfig = Field(default_factory=AbsoluteRewardConfig)
    trainer: TrainerParams = Field(default_factory=TrainerParams)
    search: SearchParams = Field(
        default_factory=lambda: SearchParams(acceptance=GreedyAcceptanceConfig()),
    )
    amp: AmpConfig = Field(default_factory=AmpConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    run_name: str = Field(
        default="",
        title="Run name",
    )

    @model_validator(mode="before")
    @classmethod
    def _default_encoder_mode(cls, data):
        """Default the model discriminator to 'joint' for configs predating encoder_mode."""
        if isinstance(data, dict):
            model = data.get("model")
            if isinstance(model, dict) and "encoder_mode" not in model:
                model["encoder_mode"] = "joint"
        return data

    @model_validator(mode="after")
    def _validate_rollouts_for_baseline(self) -> Self:
        """The training rollout count must support the chosen baseline."""
        required = build_baseline(self.baseline).min_rollouts()
        if self.search.num_rollouts < required:
            raise ValueError(
                f"baseline '{self.baseline.type}' requires search.num_rollouts >= "
                f"{required}, got {self.search.num_rollouts}."
            )
        return self

    @model_validator(mode="after")
    def _assign_run_name(self) -> Self:
        """Generate a unique timestamped run name once, unless one was supplied."""
        if not self.run_name:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_name = f"{self.problem.name}{self.problem.size}-{timestamp}"
        return self
