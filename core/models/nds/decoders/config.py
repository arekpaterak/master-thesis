from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AutoregressiveDecoderConfig(BaseModel):
    type: Literal["ar"] = "ar"
    use_hidden_decoding_state: bool = Field(default=True)


class SimpleOnePassDecoderConfig(BaseModel):
    type: Literal["simple"] = "simple"


class SlotDecoderConfig(BaseModel):
    type: Literal["slot"] = "slot"


class ProximityDecoderConfig(BaseModel):
    type: Literal["proximity"] = "proximity"
    sigma: float = Field(default=0.1)


DecoderConfig = Annotated[
    AutoregressiveDecoderConfig
    | SimpleOnePassDecoderConfig
    | SlotDecoderConfig
    | ProximityDecoderConfig,
    Field(discriminator="type"),
]
