from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class _DecoderConfigBase(BaseModel):
    with_poly: bool = Field(
        default=True,
        title="With PolyNet",
        description=(
            "Build the PolyNet seed-conditioning head used for rollout diversity. "
            "Disable it for a single-rollout, seed-independent policy (required by the "
            "greedy rollout baseline)."
        ),
    )


class AutoregressiveDecoderConfig(_DecoderConfigBase):
    type: Literal["ar"] = "ar"
    use_hidden_decoding_state: bool = Field(
        default=True,
        title="Use hidden decoding state",
        description="Use hidden decoding state, updated with a GRU.",
    )


class SimpleOnePassDecoderConfig(_DecoderConfigBase):
    type: Literal["simple"] = "simple"


class SinkhornDeduplicationConfig(BaseModel):
    iterations: int = Field(
        default=20,
        title="Sinkhorn iterations",
        description="Number of Sinkhorn row/column iterations.",
    )
    tau: float = Field(
        default=0.05,
        title="Sinkhorn temperature",
        description="Sinkhorn temperature; smaller sharpens toward a hard distinct assignment.",
    )


# None means no deduplication: slots are sampled independently (duplicates possible).
DeduplicationConfig = SinkhornDeduplicationConfig | None


class SlotDecoderConfig(_DecoderConfigBase):
    type: Literal["slot"] = "slot"
    num_slots: int = Field(
        default=15,
        title="Slot queries",
        description="Number of slot queries (should match deconstruction_size).",
    )
    slot_self_attention: Literal["none", "pre", "post"] = Field(
        default="post",
        title="Slot self-attention",
        description=(
            "Where to apply self-attention among slot queries: disabled, before node "
            "cross-attention, or after node cross-attention."
        ),
    )
    deduplication: DeduplicationConfig = Field(
        default_factory=SinkhornDeduplicationConfig,
        title="Slot deduplication",
        description=(
            "How to couple slots so they select distinct nodes. null samples each slot "
            "independently (duplicates possible); a Sinkhorn config normalizes the "
            "per-slot score matrix so slots compete for distinct nodes."
        ),
    )


DecoderConfig = Annotated[
    AutoregressiveDecoderConfig | SimpleOnePassDecoderConfig | SlotDecoderConfig,
    Field(discriminator="type"),
]
