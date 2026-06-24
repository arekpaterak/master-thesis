from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AutoregressiveDecoderConfig(BaseModel):
    type: Literal["ar"] = "ar"
    use_hidden_decoding_state: bool = Field(
        default=True,
        title="Use hidden decoding state",
        description="Use hidden decoding state, updated with a GRU.",
    )


class SimpleOnePassDecoderConfig(BaseModel):
    type: Literal["simple"] = "simple"


class SlotDecoderConfig(BaseModel):
    type: Literal["slot"] = "slot"
    k: int = Field(
        default=15,
        title="Slot queries (k)",
        description="Number of slot queries (should match deconstruction_size).",
    )


class ProximityDecoderConfig(BaseModel):
    type: Literal["proximity"] = "proximity"
    sigma: float = Field(default=0.1, title="Sigma")


class ProximityCategoricalSelectorConfig(BaseModel):
    type: Literal["proximity_categorical"] = "proximity_categorical"
    sigma_init: float = Field(
        default=0.1,
        title="Sigma init",
        description="Initial RBF bandwidth (destroy radius); refined per rollout during training.",
    )
    anisotropic: bool = Field(
        default=False,
        title="Anisotropic sigma",
        description="Predict a per-axis sigma (axis-aligned elliptical kernel) instead of a single radius.",
    )


DecoderConfig = Annotated[
    AutoregressiveDecoderConfig
    | SimpleOnePassDecoderConfig
    | SlotDecoderConfig
    | ProximityDecoderConfig
    | ProximityCategoricalSelectorConfig,
    Field(discriminator="type"),
]
