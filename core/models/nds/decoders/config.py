from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AutoregressiveDecoderConfig(BaseModel):
    type: Literal["ar"] = "ar"
    use_hidden_decoding_state: bool = Field(default=True)


class GumbelDecoderConfig(BaseModel):
    type: Literal["gumbel"] = "gumbel"


class SlotDecoderConfig(BaseModel):
    type: Literal["slot"] = "slot"


class ProximityDecoderConfig(BaseModel):
    type: Literal["proximity"] = "proximity"
    sigma: float = Field(default=0.1)


DecoderConfig = Annotated[
    AutoregressiveDecoderConfig | GumbelDecoderConfig | SlotDecoderConfig | ProximityDecoderConfig,
    Field(discriminator="type"),
]
