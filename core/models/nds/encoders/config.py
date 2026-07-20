from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EncoderConfig(BaseModel):
    """Architecture choices for the transformer encoder layers."""

    norm_type: Literal["instance", "layer", "rms"] = Field(
        default="instance",
        title="Norm type",
        description="Normalization. 'instance' = InstanceNorm across nodes (AM default), "
        "'layer' = LayerNorm across features, 'rms' = RMSNorm across features.",
    )
    pre_norm: bool = Field(
        default=False,
        title="Pre-norm",
        description="If True, normalize before each sublayer (pre-norm). Default is post-norm.",
    )
    with_final_norm: bool | None = Field(
        default=None,
        title="Norm after all layers",
        description="Final normalization before the encoder output. None resolves based on pre_norm "
        "(on when pre_norm is enabled, off otherwise).",
    )
    ff_activation: Literal["relu", "swiglu"] = Field(
        default="relu",
        title="FF activation",
        description="Feed-forward variant. 'relu' = 2-layer ReLU MLP, 'swiglu' = SwiGLU.",
    )
    encoding_order: Literal["route_neighbours", "neighbours_route"] = Field(
        default="route_neighbours",
        title="Encoding order",
        description="Order of the structural encoding layers.",
    )
    route_aggregation: Literal["mean", "sum"] = Field(
        default="sum",
        title="Route aggregation",
        description="Pooling used to aggregate customer embeddings into a per-route "
        "embedding in the route encoding layer.",
    )

    @model_validator(mode="after")
    def _resolve_with_final_norm(self) -> EncoderConfig:
        if self.with_final_norm is None:
            self.with_final_norm = self.pre_norm
        return self


class JointEncoderConfig(EncoderConfig):
    """Encoder architecture for the joint variant, which re-encodes the solution
    every step via the structural neighbours/route encoding layers."""

    with_neighbours_encoding: bool = Field(
        default=True, title="With neighbours encoding", description="Use neighbours encoding."
    )
    with_route_encoding: bool | None = Field(
        default=None,
        description="Use route encoding. None means problem-specific default (True for CVRP, False for TSP).",
    )
    num_pre_attention_layers: int = Field(
        default=2,
        title="Pre-attention layers",
        description="Self-attention layers before the neighbours/route encoding layers.",
    )
    num_post_attention_layers: int = Field(
        default=2,
        title="Post-attention layers",
        description="Self-attention layers after the neighbours/route encoding layers.",
    )


class SplitEncoderConfig(EncoderConfig):
    """Encoder architecture for the split (LNS-native) variant, which applies the
    structural neighbours/route encoding layers after positional fusion instead."""

    with_neighbours_encoding: bool = Field(
        default=True,
        title="Fusion neighbours encoding",
        description="Apply a neighbours encoding layer after positional fusion (before route and attention).",
    )
    with_route_encoding: bool | None = Field(
        default=None,
        title="Fusion route encoding",
        description="Apply a route encoding layer after positional fusion (before attention). "
        "CVRP only. None means problem-specific default (True for CVRP, False for TSP).",
    )
    num_solution_attention_layers: int = Field(
        default=0,
        title="Solution attention layers",
        description="Self-attention layers applied to all nodes after positional fusion. "
        "0 = disabled.",
    )
    num_instance_attention_layers: int = Field(
        default=4,
        title="Instance attention layers",
        description="Self-attention layers in the static instance encoder. The structural "
        "neighbours/route encoding layers are always disabled there, so there is no "
        "pre/post split -- this is one contiguous block.",
    )


class PositionalEncodingConfig(BaseModel):
    """Hierarchical positional encoding (IPE + XPE) settings.

    Where the encoding is injected relative to the encoder follows from the model
    structure, not from this config: the 'joint' encoder injects it before the encoder
    (over its inputs), while the 'split' (LNS-native) model injects it after the encoder
    just before the decoder. Disable positional encoding entirely by leaving the
    surrounding field unset (``None``).
    """

    dim: int = Field(
        default=16,
        title="Per-component encoding dimension",
        description="Per-component encoding dimension (must be even).",
    )
    direction_aware: bool = Field(
        default=False,
        title="Direction aware",
        description="If True, use sin/cos pairs (direction-sensitive). False = cosine-only "
        "(reversal-invariant).",
    )
    frequency_schedule: Literal["integer", "geometric"] = Field(
        default="integer",
        title="Frequency schedule",
        description="Frequency schedule. 'integer' = Fourier harmonics (exact reversal "
        "invariance for the cosine-only variant); 'geometric' = lambda^(-2k/D).",
    )
